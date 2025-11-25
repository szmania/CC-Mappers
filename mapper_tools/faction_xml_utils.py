import os
import xml.etree.ElementTree as ET
import random
from collections import defaultdict, Counter
import re
import Levenshtein

from mapper_tools import shared_utils
from mapper_tools import ck3_to_attila_mappings as mappings

def remove_factions_not_in_cultures(root, culture_factions, screen_name_to_faction_key_map):
    """
    Removes factions from the XML tree that are not present in the provided
    set of faction names from Cultures.xml.
    """
    factions_to_remove = []
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue
        if faction_name not in culture_factions:
            factions_to_remove.append(faction)

    for faction in factions_to_remove:
        root.remove(faction)
        print(f"  - Removed faction '{faction.get('name')}' as it is not found in Cultures.xml.")

    return len(factions_to_remove)

def remove_excluded_factions(root, excluded_factions, screen_name_to_faction_key_map):
    """
    Removes factions from the XML tree that are in the excluded_factions set.
    """
    if not excluded_factions:
        return 0

    factions_to_remove = []
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name in excluded_factions:
            factions_to_remove.append(faction)

    for faction in factions_to_remove:
        root.remove(faction)
        print(f"  - Removed explicitly excluded faction: '{faction.get('name')}'.")

    return len(factions_to_remove)

def remove_factions_in_main_mod(root, main_mod_faction_maa_map, screen_name_to_faction_key_map):
    """
    Removes factions from the XML tree if they are present in the main mod's
    faction map and have no new MenAtArm types defined in the current file.
    Returns the count of removed factions and a set of their names for sync purposes.
    """
    if not main_mod_faction_maa_map:
        return 0, set()

    factions_to_remove = []
    removed_faction_names = set()
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name in main_mod_faction_maa_map:
            # Check if this faction has any MenAtArm tags in the current file
            if not faction.findall('MenAtArm'):
                factions_to_remove.append(faction)
                removed_faction_names.add(faction_name)

    for faction in factions_to_remove:
        root.remove(faction)
        print(f"  - Removed faction '{faction.get('name')}' as it is already in the main mod and has no new MenAtArm types.")

    return len(factions_to_remove), removed_faction_names

def sync_factions_from_cultures(root, culture_factions, explicitly_removed_factions=None):
    """
    Ensures that every faction in culture_factions exists in the XML tree.
    If a faction is missing, it's created by copying the 'Default' faction.
    """
    if explicitly_removed_factions is None:
        explicitly_removed_factions = set()

    existing_factions = {faction.get('name') for faction in root.findall('Faction')}
    default_faction = root.find("./Faction[@name='Default']")

    if default_faction is None:
        print("  -> ERROR: 'Default' faction not found. Cannot add missing factions.")
        return 0

    factions_added = 0
    for faction_name in sorted(list(culture_factions)):
        if faction_name not in existing_factions and faction_name not in explicitly_removed_factions:
            new_faction = ET.fromstring(ET.tostring(default_faction))
            new_faction.set('name', faction_name)
            root.append(new_faction)
            print(f"  - Added missing faction '{faction_name}' by copying 'Default'.")
            factions_added += 1

    return factions_added

def validate_and_fix_faction_names(root, faction_key_to_screen_name_map, unit_to_faction_key_map, culture_factions):
    """
    Validates that all faction names in the XML are canonical. If a non-canonical
    name is found, it attempts to fix it.
    """
    if not faction_key_to_screen_name_map:
        return 0, {}

    factions_fixed = 0
    faction_name_map = {}
    valid_faction_names = set(faction_key_to_screen_name_map.values())

    for faction in root.findall('Faction'):
        current_name = faction.get('name')
        if current_name == "Default":
            continue

        if current_name not in valid_faction_names:
            # Attempt to find the correct name via fuzzy matching
            best_match = shared_utils.find_best_fuzzy_match(current_name, valid_faction_names)
            if best_match and best_match in culture_factions:
                faction.set('name', best_match)
                faction_name_map[current_name] = best_match
                factions_fixed += 1
            else:
                print(f"  -> WARNING: Faction '{current_name}' is not a valid faction name and no good fuzzy match was found. It will be removed.")
                root.remove(faction)

    return factions_fixed, faction_name_map

def merge_duplicate_factions(root, screen_name_to_faction_key_map):
    """
    Merges factions that have the same faction key but different screen names.
    """
    factions_by_key = defaultdict(list)
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        faction_key = screen_name_to_faction_key_map.get(faction_name)
        if faction_key:
            factions_by_key[faction_key].append(faction)

    merged_count = 0
    for key, factions in factions_by_key.items():
        if len(factions) > 1:
            # Keep the first one, merge others into it
            primary_faction = factions[0]
            print(f"  - Merging {len(factions) - 1} duplicate faction(s) for key '{key}' into '{primary_faction.get('name')}'.")
            for i in range(1, len(factions)):
                duplicate_faction = factions[i]
                for child in duplicate_faction:
                    primary_faction.append(child)
                root.remove(duplicate_faction)
                merged_count += 1

    return merged_count

def ensure_max_attributes(root, ck3_maa_definitions, unit_categories, unit_to_class_map):
    """
    Ensures that all non-siege MenAtArm units have a 'max' attribute corresponding
    to their unit type (e.g., INFANTRY, CAVALRY). Removes 'max' from siege units.
    """
    tags_updated = 0
    tags_removed = 0

    for element in root.findall('.//MenAtArm'):
        maa_definition_name = element.get('type')
        unit_key = element.get('key')

        if not maa_definition_name:
            continue

        # Skip if the unit key is missing, as we can't determine its category
        if not unit_key:
            # This will be caught by the final validation pass
            if 'max' not in element.attrib and element.get('siege') != 'true':
                print(f"  - WARNING: <MenAtArm type='{maa_definition_name}'> is missing 'key' and 'max' attributes. This will be handled by the final fix pass.")
            continue

        # Determine if it's a siege unit.
        internal_type = ck3_maa_definitions.get(maa_definition_name)
        is_siege_by_ck3_type = (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) is None) or \
                               (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type) is None)
        unit_class = unit_to_class_map.get(unit_key)
        is_siege_by_attila_class = (unit_class == 'art_siege')
        is_siege = is_siege_by_ck3_type or is_siege_by_attila_class

        if is_siege:
            if 'max' in element.attrib:
                del element.attrib['max']
                print(f"  - Removed 'max' attribute from <MenAtArm type='{maa_definition_name}'> as it is a siege unit.")
                tags_removed += 1
        else:  # Not a siege unit, must have 'max'
            # Determine the correct max value
            max_value = mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) or \
                        (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type))

            if not max_value:
                # Fallback to unit's specific category if mapping is missing
                specific_category = unit_categories.get(unit_key)
                max_value = mappings.ATTILA_CATEGORY_TO_MAX_VALUE.get(specific_category, "INFANTRY")

            if element.get('max') != max_value:
                element.set('max', max_value)
                print(f"  - Set max='{max_value}' for <MenAtArm type='{maa_definition_name}'>.")
                tags_updated += 1

    if tags_updated > 0 or tags_removed > 0:
        print(f"Updated {tags_updated} tags with a 'max' attribute and removed {tags_removed} unnecessary 'max' attributes.")
    return tags_updated + tags_removed

def ensure_siege_attributes(root, ck3_maa_definitions, unit_to_class_map, no_siege=False):
    """
    Ensures that siege units have siege="true" and non-siege units do not.
    """
    tags_updated = 0
    tags_removed = 0

    if no_siege:
        for element in root.findall('.//MenAtArm[@siege]'):
            del element.attrib['siege']
            tags_removed += 1
        if tags_removed > 0:
            print(f"Removed 'siege' attribute from {tags_removed} MenAtArm tags due to --no-siege flag.")
        return tags_removed

    for element in root.findall('.//MenAtArm'):
        maa_definition_name = element.get('type')
        unit_key = element.get('key')

        if not maa_definition_name:
            continue

        internal_type = ck3_maa_definitions.get(maa_definition_name)
        # A unit is considered siege if its CK3 type is mapped as siege,
        # OR if its assigned Attila unit has the 'art_siege' class.
        is_siege_by_ck3_type = (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) is None) or \
                               (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type) is None)

        unit_class = unit_to_class_map.get(unit_key)
        is_siege_by_attila_class = (unit_class == 'art_siege')

        is_siege = is_siege_by_ck3_type or is_siege_by_attila_class
        if is_siege:
            if element.get('siege') != 'true':
                element.set('siege', 'true')
                print(f"  - Added siege='true' to <MenAtArm type='{maa_definition_name}'>.")
                tags_updated += 1
        else:
            if 'siege' in element.attrib:
                del element.attrib['siege']
                print(f"  - Removed siege='true' from non-siege <MenAtArm type='{maa_definition_name}'>.")
                tags_removed += 1

    if tags_updated > 0 or tags_removed > 0:
        print(f"Updated {tags_updated} tags with a 'siege' attribute and removed it from {tags_removed} tags.")
    return tags_updated + tags_removed

def ensure_siege_engine_per_unit_attributes(root):
    """
    Ensures that siege units have siege_engine_per_unit="1" and non-siege units do not.
    """
    tags_updated = 0
    tags_removed = 0

    for element in root.findall('.//MenAtArm'):
        is_siege = element.get('siege') == 'true'

        if is_siege:
            if element.get('siege_engine_per_unit') != '1':
                element.set('siege_engine_per_unit', '1')
                tags_updated += 1
        else:
            if 'siege_engine_per_unit' in element.attrib:
                del element.attrib['siege_engine_per_unit']
                tags_removed += 1

    if tags_updated > 0 or tags_removed > 0:
        print(f"Updated {tags_updated} tags with a 'siege_engine_per_unit' attribute and removed it from {tags_removed} tags.")
    return tags_updated + tags_removed

def ensure_num_guns_attributes(root, unit_to_num_guns_map):
    """
    Adds the 'num_guns' attribute to units that have it defined in the TSV files.
    """
    if not unit_to_num_guns_map:
        return 0

    tags_updated = 0
    tags_removed = 0

    # Find all tags that can have a 'key' attribute
    elements_with_key = root.findall('.//*[@key]')

    for element in elements_with_key:
        unit_key = element.get('key')
        num_guns = unit_to_num_guns_map.get(unit_key)

        if num_guns is not None and num_guns > 0:
            if element.get('num_guns') != str(num_guns):
                element.set('num_guns', str(num_guns))
                tags_updated += 1
        else:
            if 'num_guns' in element.attrib:
                del element.attrib['num_guns']
                tags_removed += 1

    if tags_updated > 0 or tags_removed > 0:
        print(f"Updated {tags_updated} tags with a 'num_guns' attribute and removed it from {tags_removed} tags.")
    return tags_updated + tags_removed

def ensure_subculture_attributes(root, screen_name_to_faction_key_map, faction_to_subculture_map, no_subculture=False, most_common_faction_key=None):
    """
    Ensures that all factions have a 'subculture' attribute.
    """
    if no_subculture:
        tags_removed = 0
        for faction in root.findall('.//Faction[@subculture]'):
            del faction.attrib['subculture']
            tags_removed += 1
        if tags_removed > 0:
            print(f"Removed 'subculture' attribute from {tags_removed} factions due to --no-subculture flag.")
        return tags_removed

    tags_updated = 0
    default_subculture = None
    if most_common_faction_key:
        default_subculture = faction_to_subculture_map.get(most_common_faction_key)

    if not default_subculture:
        print("  -> WARNING: Could not determine a default subculture. Subculture attribute may not be added for all factions.")

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        faction_key = screen_name_to_faction_key_map.get(faction_name)
        subculture = faction_to_subculture_map.get(faction_key)

        if subculture:
            if faction.get('subculture') != subculture:
                faction.set('subculture', subculture)
                tags_updated += 1
        elif default_subculture:
            if faction.get('subculture') != default_subculture:
                faction.set('subculture', default_subculture)
                print(f"  - Faction '{faction_name}' missing subculture. Applying default: '{default_subculture}'.")
                tags_updated += 1

    if tags_updated > 0:
        print(f"Added/updated 'subculture' attribute for {tags_updated} factions.")
    return tags_updated

def ensure_default_faction_is_first(root):
    """
    Finds the 'Default' faction and moves it to be the first element under the root.
    """
    default_faction = root.find("./Faction[@name='Default']")
    if default_faction is not None:
        # Check if it's already the first element
        if root[0] != default_faction:
            root.remove(default_faction)
            root.insert(0, default_faction)
            print("Moved 'Default' faction to be the first element.")
            return 1
    return 0

def create_default_faction_if_missing(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, unit_to_class_map=None, unit_to_description_map=None, unit_stats_map=None, excluded_units_set=None, is_submod_mode=False):
    """
    Checks if the 'Default' faction exists. If not, creates it and populates it
    with a representative set of units.
    """
    if root.find("./Faction[@name='Default']") is not None:
        return 0

    print("Creating 'Default' faction as it is missing...")
    default_faction = ET.Element('Faction', name="Default")

    if not is_submod_mode:
        # Add General units
        general_candidates = [u for u in general_units if u in template_faction_unit_pool]
        if not general_candidates:
            general_candidates = [u for u in template_faction_unit_pool if unit_categories.get(u) == 'cavalry']
        if general_candidates:
            # Simplified: just pick one for all ranks for the template
            chosen_general = random.choice(general_candidates)
            for rank in ['1', '2', '3']:
                ET.SubElement(default_faction, 'General', {'key': chosen_general, 'rank': rank})

        # Add Knights units
        knights_roles = mappings.TAG_ROLE_MAPPING.get('Knights', [])
        knights_candidates = []
        for role in knights_roles:
            units_in_role = categorized_units.get(role, [])
            knights_candidates.extend([u for u in units_in_role if u in template_faction_unit_pool])
        if not knights_candidates:
            knights_candidates = [u for u in template_faction_unit_pool if unit_categories.get(u) == 'cavalry']
        if knights_candidates:
            # Simplified: just pick one for all ranks for the template
            chosen_knight = random.choice(knights_candidates)
            for rank in ['1', '2', '3']:
                ET.SubElement(default_faction, 'Knights', {'key': chosen_knight, 'rank': rank})

        # Add Levies
        levy_candidates = [u for u in template_faction_unit_pool if unit_categories.get(u) in ['inf_melee', 'inf_ranged']]
        if levy_candidates:
            num_levies = min(len(levy_candidates), 4)
            chosen_levies = random.sample(levy_candidates, num_levies)
            for levy_unit in chosen_levies:
                ET.SubElement(default_faction, 'Levies', {'percentage': '25', 'key': levy_unit, 'max': 'LEVY'})

    # Add MenAtArm units for all defined types
    if ck3_maa_definitions:
        for maa_type in sorted(ck3_maa_definitions.keys()):
            # Create a placeholder tag. The key will be filled in by the main processing loop.
            ET.SubElement(default_faction, 'MenAtArm', {'type': maa_type})

    root.insert(0, default_faction)
    return 1

def sync_faction_structure_from_default(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, screen_name_to_faction_key_map, faction_key_to_units_map, unit_to_class_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map=None, unit_to_description_map=None, unit_stats_map=None, main_mod_faction_maa_map=None, excluded_units_set=None, faction_pool_cache=None, faction_to_heritage_map=None, heritage_to_factions_map=None):
    """
    Ensures all non-Default factions have a complete set of MenAtArm tags
    as defined in the Default faction.
    """
    default_faction = root.find("./Faction[@name='Default']")
    if default_faction is None:
        print("  -> ERROR: 'Default' faction not found. Cannot sync structure.")
        return 0

    default_maa_types = {el.get('type') for el in default_faction.findall('MenAtArm')}
    tags_added = 0

    for faction in root.findall('Faction'):
        if faction.get('name') == "Default":
            continue

        existing_maa_types = {el.get('type') for el in faction.findall('MenAtArm')}
        missing_types = default_maa_types - existing_maa_types

        # Also consider types from the main mod as "existing"
        if main_mod_faction_maa_map:
            main_mod_types = main_mod_faction_maa_map.get(faction.get('name'), set())
            missing_types = missing_types - main_mod_types

        if missing_types:
            for maa_type in sorted(list(missing_types)):
                # Find the corresponding element in Default to copy attributes from
                default_element = default_faction.find(f"./MenAtArm[@type='{maa_type}']")
                if default_element is not None:
                    new_element = ET.fromstring(ET.tostring(default_element))
                    # Clear the key so it gets repopulated for this specific faction
                    if 'key' in new_element.attrib:
                        del new_element.attrib['key']
                    faction.append(new_element)
                    tags_added += 1

    if tags_added > 0:
        print(f"Synced {tags_added} missing MenAtArm tags from 'Default' to other factions.")
    return tags_added

def reorganize_faction_children(root):
    """
    Reorganizes the children of each Faction element into a specific, consistent order.
    Garrison tags are sorted by their 'level' attribute.
    """
    # Define the desired order of tags
    tag_order = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']
    order_map = {tag: i for i, tag in enumerate(tag_order)}
    reorganized_count = 0

    for faction in root.findall('Faction'):
        children = list(faction)
        # Check if sorting is needed to avoid unnecessary operations
        if not any(child.tag in order_map for child in children):
            continue

        # Define a key function for sorting
        def sort_key(child):
            tag_order_val = order_map.get(child.tag, len(tag_order))

            # Secondary sort for Garrison tags by level
            if child.tag == 'Garrison':
                level = child.get('level', '999') # Default to a high number if level is missing
                try:
                    level_val = int(level)
                except (ValueError, TypeError):
                    level_val = 999 # Fallback for non-integer or None levels
                return (tag_order_val, level_val)

            # For other tags, no secondary sort is needed
            return (tag_order_val, 0)

        # Use a stable sort to preserve relative order of same-tag elements
        sorted_children = sorted(children, key=sort_key)

        # Only reorganize if the order has actually changed
        # A simple list comparison is sufficient here.
        if children != sorted_children:
            # Clear existing children and append sorted ones
            for child in children:
                faction.remove(child)
            for child in sorted_children:
                faction.append(child)
            reorganized_count += 1

    if reorganized_count > 0:
        print(f"Reorganized tags for {reorganized_count} factions into a consistent order.")
    return reorganized_count

def reorder_attributes_in_all_tags(root):
    """
    Reorders the attributes of all tags within the XML tree to a consistent order.
    """
    # Define the desired order of attributes for each tag
    attribute_order_map = {
        'Factions': ['submod_tag', 'submod_addon_tag'],
        'Faction': ['name', 'subculture'],
        'General': ['key', 'rank'],
        'Knights': ['key', 'rank'],
        'Levies': ['key', 'percentage', 'max'],
        'MenAtArm': ['type', 'key', 'max', 'siege', 'siege_engine_per_unit', 'num_guns'],
        'Garrison': ['key', 'percentage', 'max', 'level']
    }
    reordered_count = 0

    for tag_name, order in attribute_order_map.items():
        for element in root.iter(tag_name):
            current_attrs = dict(element.attrib)
            # Create a new ordered dictionary of attributes
            new_attrs = {key: current_attrs[key] for key in order if key in current_attrs}
            # Add any other attributes not in the defined order (preserves them)
            for key, value in current_attrs.items():
                if key not in new_attrs:
                    new_attrs[key] = value

            # Only update if the order has changed
            if list(element.attrib.keys()) != list(new_attrs.keys()):
                element.attrib.clear()
                element.attrib.update(new_attrs)
                reordered_count += 1

    if reordered_count > 0:
        print(f"Reordered attributes for {reordered_count} tags into a consistent order.")
    return reordered_count

def rename_porcentage_to_percentage(root):
    """
    Finds all tags with a 'porcentage' attribute and renames it to 'percentage'.
    """
    count = 0
    for element in root.findall(".//*[@porcentage]"):
        value = element.get('porcentage')
        del element.attrib['porcentage']
        element.set('percentage', value)
        count += 1
    if count > 0:
        print(f"Renamed 'porcentage' to 'percentage' for {count} tags.")
    return count

def remove_excluded_unit_keys(root, excluded_units_set):
    """
    Removes the 'key' attribute from any unit tag if the key is in the excluded_units_set.
    This forces the script to find a new replacement for that unit.
    """
    if not excluded_units_set:
        return 0

    count = 0
    for element in root.findall(".//*[@key]"):
        if element.get('key') in excluded_units_set:
            del element.attrib['key']
            count += 1
    if count > 0:
        print(f"Removed 'key' attribute from {count} tags for excluded units to force re-evaluation.")
    return count

def remove_stale_unit_keys(root, all_valid_units):
    """
    Removes the 'key' attribute from any unit tag if the key is not in the set of all_valid_units.
    This forces the script to find a new replacement for the stale unit.
    """
    if not all_valid_units:
        return 0

    count = 0
    for element in root.findall(".//*[@key]"):
        unit_key = element.get('key')
        if unit_key not in all_valid_units:
            del element.attrib['key']
            count += 1
    if count > 0:
        print(f"Removed {count} stale unit keys that no longer exist in the database. They will be replaced.")
    return count

def remove_zero_percentage_tags(root):
    """
    Removes any <Levies> or <Garrison> tags that have a percentage of "0" after normalization.
    """
    tags_removed = 0
    for faction in root.findall('Faction'):
        for tag_name in ['Levies', 'Garrison']:
            for element in list(faction.findall(tag_name)):
                if element.get('percentage') == '0':
                    faction.remove(element)
                    tags_removed += 1
    if tags_removed > 0:
        print(f"Removed {tags_removed} zero-percentage levy/garrison tags after final normalization.")
    return tags_removed

def remove_invalid_men_at_arm_tags(root):
    """
    Removes any <MenAtArm> tags that are missing the required 'type' attribute.
    """
    tags_removed = 0
    for faction in root.findall('Faction'):
        for element in list(faction.findall('MenAtArm')):
            if 'type' not in element.attrib or not element.get('type'):
                faction.remove(element)
                tags_removed += 1
    if tags_removed > 0:
        print(f"Removed {tags_removed} invalid <MenAtArm> tags that were missing the 'type' attribute.")
    return tags_removed

def remove_duplicate_men_at_arm_tags(root):
    """
    Removes duplicate <MenAtArm> tags (based on 'type') within each faction.
    It keeps the first occurrence and removes subsequent ones.
    """
    tags_removed = 0
    for faction in root.findall('Faction'):
        seen_maa_types = set()
        elements_to_remove = []
        for element in faction.findall('MenAtArm'):
            maa_type = element.get('type')
            if maa_type:
                if maa_type in seen_maa_types:
                    elements_to_remove.append(element)
                    tags_removed += 1
                else:
                    seen_maa_types.add(maa_type)

        for el in elements_to_remove:
            faction.remove(el)

    if tags_removed > 0:
        print(f"Removed {tags_removed} duplicate <MenAtArm> tags to ensure one per type.")
    return tags_removed

def remove_core_unit_tags(root, factions_to_strip):
    """
    Removes General, Knights, Levies, and Garrison tags from a specified set of factions.
    Used for submod generation mode.
    """
    tags_removed = 0
    if not factions_to_strip:
        return 0

    for faction in root.findall('Faction'):
        if faction.get('name') in factions_to_strip:
            for tag_name in ['General', 'Knights', 'Levies', 'Garrison']:
                for element in list(faction.findall(tag_name)):
                    faction.remove(element)
                    tags_removed += 1
    if tags_removed > 0:
        print(f"Removed {tags_removed} core unit tags (General, Knights, Levies, Garrison) from factions already present in the main mod.")
    return tags_removed

def remove_duplicate_ranked_units(root):
    """
    Ensures that for each faction, each rank for General/Knights has only one unit,
    and each unit is only used for one rank. It prioritizes keeping the highest rank for a unit.
    This function runs in two stages to handle different types of duplicates that can occur after merging factions.
    """
    tags_removed = 0
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        for tag_name in ['General', 'Knights']:
            # --- Stage 1: Ensure only one unit per rank ---
            # This handles cases where a merge results in multiple units for the same rank (e.g., two rank="1" generals).
            ranks_to_elements = defaultdict(list)
            for element in faction.findall(tag_name):
                rank = element.get('rank')
                if rank:
                    ranks_to_elements[rank].append(element)

            for rank, elements in ranks_to_elements.items():
                if len(elements) > 1:
                    # We have multiple units for the same rank. Keep one, remove others.
                    # Strategy: Keep the one with a 'key' attribute if possible, otherwise the first one.
                    element_to_keep = None
                    for el in elements:
                        if el.get('key'):
                            element_to_keep = el
                            break
                    if not element_to_keep:
                        element_to_keep = elements[0] # Fallback to the first element found

                    for el_to_remove in elements:
                        if el_to_remove is not element_to_keep:
                            faction.remove(el_to_remove)
                            tags_removed += 1
                            print(f"  - Removed duplicate <{tag_name}> for rank {rank} in faction '{faction_name}' (Unit: '{el_to_remove.get('key', 'N/A')}'). Keeping unit '{element_to_keep.get('key', 'N/A')}'.")

            # --- Stage 2: Ensure each unit is used for only one rank (the highest) ---
            # This handles cases where the same unit is assigned to multiple ranks (e.g., unit_A is both rank 1 and 2).
            keys_to_elements = defaultdict(list)
            for element in faction.findall(tag_name):
                key = element.get('key')
                if key:
                    keys_to_elements[key].append(element)

            for key, elements in keys_to_elements.items():
                if len(elements) > 1:
                    # Sort by rank descending (e.g., 3, 2, 1)
                    elements.sort(key=lambda el: int(el.get('rank', 0)), reverse=True)
                    # Keep the first one (highest rank) and remove the rest
                    for el_to_remove in elements[1:]:
                        faction.remove(el_to_remove)
                        tags_removed += 1
                        print(f"  - Removed duplicate <{tag_name}> unit '{key}' from rank {el_to_remove.get('rank')} in faction '{faction_name}' (already present at a higher rank).")

    if tags_removed > 0:
        print(f"Removed {tags_removed} duplicate/conflicting ranked units.")
    return tags_removed

def _get_all_tiered_pools(faction_name, screen_name_to_faction_key_map, faction_key_to_units_map,
                              faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                              culture_to_faction_map, log_prefix="", excluded_units_set=None,
                              faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None):
    """
    Gets the working pool of units for a faction for all 7 fallback tiers.
    Returns a list of pools and a corresponding list of log strings.
    """
    tiered_pools = []
    log_strings = []
    faction_key = screen_name_to_faction_key_map.get(faction_name)

    # Tier 1: Faction-specific units
    pool1 = set()
    if faction_key and faction_key in faction_key_to_units_map:
        pool1 = set(faction_key_to_units_map[faction_key])
        if excluded_units_set:
            pool1 = pool1 - excluded_units_set
    tiered_pools.append(pool1)
    log_strings.append(f"'{faction_name}' (Tier 1: Faction-specific)")

    # Tier 2: Heritage-level fallback
    pool2 = set()
    primary_heritage = None
    if faction_to_heritage_map and heritage_to_factions_map:
        heritage = faction_to_heritage_map.get(faction_name)
        primary_heritage = heritage
        if heritage:
            sibling_faction_names = heritage_to_factions_map.get(heritage, [])
            for sibling_name in sibling_faction_names:
                sibling_key = screen_name_to_faction_key_map.get(sibling_name)
                if sibling_key:
                    sibling_units = faction_key_to_units_map.get(sibling_key)
                    if sibling_units:
                        pool2.update(sibling_units)
            if excluded_units_set:
                pool2 = pool2 - excluded_units_set
    tiered_pools.append(pool2)
    log_strings.append(f"'{faction_name}' (Tier 2: Heritage '{primary_heritage or 'N/A'}')")

    # Tier 3: Related Heritage Fallback
    pool3 = set()
    all_related_heritages = set()
    if primary_heritage and faction_to_heritages_map and heritage_to_factions_map:
        sibling_factions = heritage_to_factions_map.get(primary_heritage, [])
        for sibling in sibling_factions:
            heritages_for_sibling = faction_to_heritages_map.get(sibling, [])
            all_related_heritages.update(heritages_for_sibling)
        all_related_heritages.discard(primary_heritage)
        if 'Default' in all_related_heritages:
            all_related_heritages.discard('Default')
        if all_related_heritages:
            factions_from_related_heritages = set()
            for rel_heritage in all_related_heritages:
                factions_from_related_heritages.update(heritage_to_factions_map.get(rel_heritage, []))
            for faction_in_pool in factions_from_related_heritages:
                faction_key_rel = screen_name_to_faction_key_map.get(faction_in_pool)
                if faction_key_rel:
                    units = faction_key_to_units_map.get(faction_key_rel)
                    if units:
                        pool3.update(units)
            if excluded_units_set:
                pool3 = pool3 - excluded_units_set
    tiered_pools.append(pool3)
    log_strings.append(f"'{faction_name}' (Tier 3: Related Heritages [{len(all_related_heritages)}])")

    # Tier 4: Subculture-level fallback
    pool4 = set()
    subculture = None
    if faction_to_subculture_map and subculture_to_factions_map:
        subculture = faction_to_subculture_map.get(faction_key)
        if subculture:
            sibling_faction_keys = subculture_to_factions_map.get(subculture, [])
            for sibling_key in sibling_faction_keys:
                sibling_units = faction_key_to_units_map.get(sibling_key)
                if sibling_units:
                    pool4.update(sibling_units)
            if excluded_units_set:
                pool4 = pool4 - excluded_units_set
    tiered_pools.append(pool4)
    log_strings.append(f"'{faction_name}' (Tier 4: Subculture '{subculture or 'N/A'}')")

    # Tier 5: Heritage's Subcultures Fallback
    pool5 = set()
    subcultures_from_heritage = set()
    if primary_heritage and heritage_to_factions_map and faction_to_subculture_map and subculture_to_factions_map:
        sibling_factions_in_heritage = heritage_to_factions_map.get(primary_heritage, [])
        for sibling_faction_name in sibling_factions_in_heritage:
            sibling_faction_key = screen_name_to_faction_key_map.get(sibling_faction_name)
            if sibling_faction_key:
                subculture_for_sibling = faction_to_subculture_map.get(sibling_faction_key)
                if subculture_for_sibling:
                    subcultures_from_heritage.add(subculture_for_sibling)
        if subcultures_from_heritage:
            for subculture_from_heritage in subcultures_from_heritage:
                factions_in_subculture = subculture_to_factions_map.get(subculture_from_heritage, [])
                for faction_key_in_subculture in factions_in_subculture:
                    units = faction_key_to_units_map.get(faction_key_in_subculture)
                    if units:
                        pool5.update(units)
            if excluded_units_set:
                pool5 = pool5 - excluded_units_set
    tiered_pools.append(pool5)
    log_strings.append(f"'{faction_name}' (Tier 5: Heritage's Subcultures [{len(subcultures_from_heritage)}])")

    # Tier 6: Related Heritages' Subcultures Fallback
    pool6 = set()
    subcultures_from_related_heritages = set()
    if all_related_heritages:
        factions_from_related_heritages = set()
        for rel_heritage in all_related_heritages:
            factions_from_related_heritages.update(heritage_to_factions_map.get(rel_heritage, []))
        for faction_name_in_related in factions_from_related_heritages:
            faction_key_rel = screen_name_to_faction_key_map.get(faction_name_in_related)
            if faction_key_rel:
                subculture_rel = faction_to_subculture_map.get(faction_key_rel)
                if subculture_rel:
                    subcultures_from_related_heritages.add(subculture_rel)
        if subcultures_from_related_heritages:
            for subculture_rel in subcultures_from_related_heritages:
                factions_in_subculture = subculture_to_factions_map.get(subculture_rel, [])
                for faction_key_in_subculture in factions_in_subculture:
                    units = faction_key_to_units_map.get(faction_key_in_subculture)
                    if units:
                        pool6.update(units)
            if excluded_units_set:
                pool6 = pool6 - excluded_units_set
    tiered_pools.append(pool6)
    log_strings.append(f"'{faction_name}' (Tier 6: Related Heritages' Subcultures [{len(subcultures_from_related_heritages)}])")

    # Tier 7: Culture-level fallback
    pool7 = set()
    faction_culture = None
    if culture_to_faction_map:
        for culture, factions in culture_to_faction_map.items():
            if faction_name in factions:
                faction_culture = culture
                break
        if faction_culture:
            sibling_faction_names = culture_to_faction_map.get(faction_culture, [])
            for sibling_name in sibling_faction_names:
                sibling_key = screen_name_to_faction_key_map.get(sibling_name)
                if sibling_key:
                    sibling_units = faction_key_to_units_map.get(sibling_key)
                    if sibling_units:
                        pool7.update(sibling_units)
            if excluded_units_set:
                pool7 = pool7 - excluded_units_set
    tiered_pools.append(pool7)
    log_strings.append(f"'{faction_name}' (Tier 7: Culture '{faction_culture or 'N/A'}')")

    # Log a summary of the tiers
    summary = []
    for i, pool in enumerate(tiered_pools):
        if pool:
            summary.append(f"Tier {i+1} ({len(pool)} units)")
    if summary:
        print(f"  {log_prefix} Pool generation summary for '{faction_name}': {', '.join(summary)}.")
    else:
        print(f"  {log_prefix} Pool generation summary for '{faction_name}': All tiers are empty.")

    return tiered_pools, log_strings

def _get_faction_working_pool(tiered_pools, log_strings, required_classes=None, unit_to_class_map=None, min_pool_size=1):
    """
    Processes a list of tiered pools to get the final working pool by accumulating tiers
    until a stopping condition is met.
    """
    accumulated_pool = set()
    final_log_string = ""

    for i, pool in enumerate(tiered_pools):
        if not pool:
            continue

        accumulated_pool.update(pool)
        final_log_string = log_strings[i]

        # If searching for specific classes, stop when one is found.
        if required_classes and unit_to_class_map:
            if any(unit_to_class_map.get(unit) in required_classes for unit in accumulated_pool):
                print(f"      -> INFO: Found matching unit class in {final_log_string}. Using accumulated pool of {len(accumulated_pool)} units.")
                return accumulated_pool, final_log_string
        # Otherwise, if not class-searching, stop when a minimum size is reached.
        elif len(accumulated_pool) >= min_pool_size:
            print(f"      -> INFO: Reached minimum pool size of {min_pool_size} at {final_log_string}. Using accumulated pool of {len(accumulated_pool)} units.")
            return accumulated_pool, final_log_string

    # If the loop completes without returning, we've exhausted all tiers.
    if accumulated_pool:
        print(f"      -> INFO: Exhausted all tiers for {final_log_string}. Using final accumulated pool of {len(accumulated_pool)} units.")
        return accumulated_pool, final_log_string
    
    # If all tiers were empty.
    print(f"      -> WARNING: All tiers were empty. Returning empty pool.")
    return set(), log_strings[0] if log_strings else "Unknown Faction"
