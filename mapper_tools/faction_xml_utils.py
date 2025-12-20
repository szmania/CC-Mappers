import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import re
import Levenshtein
import os
import json
import random
import hashlib

from mapper_tools import shared_utils
from mapper_tools import ck3_to_attila_mappings as mappings
from mapper_tools import unit_selector
from mapper_tools import unit_management


def conditionally_remove_procedural_keys(root, llm_helper, tier, faction_pool_cache, screen_name_to_faction_key_map,
                                         faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map,
                                         faction_key_to_screen_name_map, culture_to_faction_map, excluded_units_set,
                                         faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map,
                                         unit_to_class_map):
    """
    Removes keys from units that were likely assigned procedurally and are not in the LLM cache,
    forcing them to be re-evaluated.
    """
    removed_count = 0
    if not llm_helper:
        return 0

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        # Get the full working pool for this faction (without global exclusions yet)
        working_pool, _, _ = get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, set(), faction_to_heritage_map, heritage_to_factions_map,
            faction_to_heritages_map
        )

        for element in faction.iter():
            if 'key' in element.attrib and element.tag in ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']:
                unit_key = element.get('key')
                if unit_key in excluded_units_set:
                    continue # Don't remove keys for explicitly excluded units

                # Check if the unit is in the faction's working pool
                if unit_key not in working_pool:
                    # This unit is not in the faction's valid pool, so it's likely a placeholder or invalid.
                    # We should remove its key to force re-evaluation.
                    del element.attrib['key']
                    removed_count += 1
                    continue

                # Check if the unit is in the LLM cache for this specific request type
                # This is a simplified check, a more robust one would involve reconstructing the LLM request ID
                # For now, we assume if it's in the working pool, and not in the LLM cache, it's procedural.
                if not llm_helper.is_unit_in_cache(unit_key):
                    del element.attrib['key']
                    removed_count += 1

    if removed_count > 0:
        print(f"Removed {removed_count} procedural unit keys to force re-evaluation.")
    return removed_count


def create_default_faction_if_missing(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, unit_to_class_map=None, unit_to_description_map=None, unit_stats_map=None, excluded_units_set=None, is_submod_mode=False):
    """
    Ensures a 'Default' faction exists in the XML. If not, it creates one and populates it
    with a representative set of units.
    """
    default_faction = root.find("Faction[@name='Default']")
    if default_faction is None:
        print("\n--- Creating missing 'Default' faction ---")
        default_faction = ET.SubElement(root, 'Faction', name='Default')
        default_created = 1

        # Add a representative set of MenAtArm units to the Default faction
        # Prioritize units from the template_faction_unit_pool
        default_maa_units = set()
        if template_faction_unit_pool:
            # Select a diverse set of units from the template pool
            selected_from_template = random.sample(list(template_faction_unit_pool), min(50, len(template_faction_unit_pool)))
            default_maa_units.update(selected_from_template)

        # Ensure some general units are included if not already
        if general_units:
            selected_generals = random.sample(list(general_units), min(5, len(general_units)))
            default_maa_units.update(selected_generals)

        # Add some random units from all_units if the default_maa_units is still small
        if len(default_maa_units) < 50 and all_units:
            remaining_slots = 50 - len(default_maa_units)
            additional_units = random.sample(list(all_units - default_maa_units), min(remaining_slots, len(all_units - default_maa_units)))
            default_maa_units.update(additional_units)

        # Add MenAtArm tags for each selected unit
        for unit_key in sorted(list(default_maa_units)):
            # Try to infer a type, or use a generic one
            maa_type = mappings.ATTILA_CLASS_TO_CK3_TYPE.get(unit_to_class_map.get(unit_key)) or "generic_men_at_arms"
            ET.SubElement(default_faction, 'MenAtArm', type=maa_type, key=unit_key)

        # Add some generic General, Knights, Levies, Garrison tags
        # These will be filled in by the unit assignment passes
        ET.SubElement(default_faction, 'General', rank='1')
        ET.SubElement(default_faction, 'General', rank='2')
        ET.SubElement(default_faction, 'Knights', rank='1')
        ET.SubElement(default_faction, 'Levies', percentage='100', max='LEVY')
        ET.SubElement(default_faction, 'Garrison', level='1', percentage='100', max='LEVY')

        print("  -> Default faction created with empty unit tags.")
        return default_created
    return 0

def sync_factions_from_cultures(root, culture_factions, explicitly_removed_factions=None, all_faction_elements=None):
    """
    Adds any factions present in culture_factions but missing from the XML.
    """
    added_count = 0
    # Use cached faction elements if provided, otherwise find them normally
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')
    existing_faction_names = {f.get('name') for f in factions_to_iterate}
    if explicitly_removed_factions is None:
        explicitly_removed_factions = set()

    for faction_name in sorted(list(culture_factions)):
        if faction_name not in existing_faction_names and faction_name not in explicitly_removed_factions:
            ET.SubElement(root, 'Faction', name=faction_name)
            print(f"  -> Added missing faction: '{faction_name}' from Cultures.xml.")
            added_count += 1
    return added_count

def create_faction_to_heritages_map(heritage_to_factions_map):
    """
    Creates a map from faction key to a list of heritage names it belongs to.
    (A faction can belong to multiple cultures, and cultures can belong to multiple heritages in some mods).
    This function assumes heritage_to_factions_map is heritage_name -> list of faction_keys.
    """
    faction_to_heritages_map = defaultdict(list)
    for heritage_name, faction_keys in heritage_to_factions_map.items():
        for faction_key in faction_keys:
            faction_to_heritages_map[faction_key].append(heritage_name)
    return faction_to_heritages_map


def ensure_default_faction_is_first(root):
    """
    Ensures the 'Default' faction is the first child element under <Factions>.
    Returns 1 if moved, 0 otherwise.
    """
    default_faction = root.find("Faction[@name='Default']")
    if default_faction is not None and root.getchildren()[0] is not default_faction:
        root.remove(default_faction)
        root.insert(0, default_faction)
        print("Moved 'Default' faction to the top of the XML.")
        return 1
    return 0


def ensure_required_tags_exist(root, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                               faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                               culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                               heritage_to_factions_map, faction_to_heritages_map,
                               general_units, unit_stats_map, unit_categories, unit_to_training_level,
                               faction_elite_units, ck3_maa_definitions, unit_to_class_map, unit_to_description_map,
                               categorized_units, unit_to_tier_map, all_units):
    """
    Ensures every faction has at least one of each required core unit tag.
    If a tag is missing, it creates one and populates it with a suitable unit.
    This is a final pre-validation step to guarantee structural integrity.
    """
    required_tags = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']
    added_count = 0

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if not faction_name:
            continue

        used_units = {el.get('key') for el in faction if el.get('key')}

        for tag_name in required_tags:
            if not faction.find(tag_name):
                print(f"  -> PRE-VALIDATION: Faction '{faction_name}' is missing required <{tag_name}> tag. Attempting to create and populate.")

                # This logic is a simplified version of populate_or_remove_keyless_tags, focused on creation
                new_key = None
                from mapper_tools import unit_selector

                # Get the working pool for this faction
                working_pool, _, _ = get_cached_faction_working_pool(
                    faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                    culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                    heritage_to_factions_map, faction_to_heritages_map
                )
                available_pool = working_pool - used_units

                if tag_name == 'General':
                    # Use the more robust candidate pool function for Generals
                    from mapper_tools import unit_management
                    general_candidate_pool = unit_management._get_candidate_pool_for_tag('General', available_pool, general_units, categorized_units, unit_categories, unit_to_training_level)
                    if general_candidate_pool:
                        # Select the best unit from the candidate pool based on stats
                        general_candidate_pool_with_stats = {unit for unit in general_candidate_pool if unit in available_pool}
                        if general_candidate_pool_with_stats:
                            new_key = unit_selector.select_best_unit_from_pool(general_candidate_pool_with_stats, rank=1, unit_stats_map=unit_stats_map)
                elif tag_name == 'Knights':
                    # Use the more robust candidate pool function for Knights
                    from mapper_tools import unit_management
                    knight_candidate_pool = unit_management._get_candidate_pool_for_tag('Knights', available_pool, general_units, categorized_units, unit_categories, unit_to_training_level)
                    if knight_candidate_pool:
                        # Select the best unit from the candidate pool based on stats
                        knight_candidate_pool_with_stats = {unit for unit in knight_candidate_pool if unit in available_pool}
                        if knight_candidate_pool_with_stats:
                            new_key = unit_selector.select_best_unit_from_pool(knight_candidate_pool_with_stats, rank=1, unit_stats_map=unit_stats_map)
                elif tag_name == 'Levies':
                    faction_elites = faction_elite_units.get(faction_name, set()) if faction_elite_units else set()
                    new_key = unit_selector.find_best_levy_replacement(available_pool, unit_to_training_level, unit_categories, exclude_units=faction_elites)
                elif tag_name == 'Garrison':
                    new_key = unit_selector.find_best_garrison_replacement(available_pool, unit_categories, exclude_units=general_units)
                elif tag_name == 'MenAtArm':
                    # For a missing MAA, we have to pick a generic type and find a unit for it.
                    maa_type = 'heavy_infantry'
                    if 'heavy_infantry' not in ck3_maa_definitions:
                        maa_type = next(iter(ck3_maa_definitions), None)

                    if maa_type:
                        internal_type = ck3_maa_definitions.get(maa_type)
                        expected_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(maa_type) or (mappings.CK3_TYPE_TO_ATTILA_CLASS.get(internal_type) if internal_type else None)
                        if expected_classes:
                            class_pool = {u for u in available_pool if unit_to_class_map.get(u) in expected_classes}
                            if class_pool:
                                new_key = unit_selector.select_best_unit_by_tier(class_pool, unit_to_tier_map)

                # If no ideal unit was found, use the global fallback
                if not new_key:
                    print(f"    -> PRE-VALIDATION: Could not find ideal unit for new <{tag_name}> in faction '{faction_name}'. Attempting global fallback.")
                    global_fallback_pool = all_units - (excluded_units_set if excluded_units_set else set()) - used_units
                    if global_fallback_pool:
                        new_key = random.choice(list(global_fallback_pool))
                        print(f"    -> PRE-VALIDATION: Assigned random global unit '{new_key}' as last resort.")

                if new_key:
                    if tag_name == 'General':
                        ET.SubElement(faction, tag_name, key=new_key, rank='1')
                    elif tag_name == 'Knights':
                        ET.SubElement(faction, tag_name, key=new_key, rank='1')
                    elif tag_name == 'Levies':
                        ET.SubElement(faction, tag_name, key=new_key, percentage='100', max='LEVY')
                    elif tag_name == 'Garrison':
                        ET.SubElement(faction, tag_name, key=new_key, level='1', percentage='100', max='LEVY')
                    elif tag_name == 'MenAtArm':
                        maa_type = 'heavy_infantry'
                        if 'heavy_infantry' not in ck3_maa_definitions:
                            maa_type = next(iter(ck3_maa_definitions), "generic_men_at_arms")
                        # For MenAtArm, 'type' should come before 'key'
                        ET.SubElement(faction, tag_name, type=maa_type, key=new_key)

                    used_units.add(new_key)
                    added_count += 1
                    print(f"    -> Successfully added and populated <{tag_name}> for '{faction_name}'.")
                else:
                    print(f"    -> CRITICAL: Could not find any unit to populate missing <{tag_name}> for '{faction_name}'. This will likely cause validation to fail.")

        if added_count > 0:
            print(f"  -> PRE-VALIDATION: Added and populated {added_count} missing required unit tags.")

    return added_count


def ensure_subculture_attributes(root, screen_name_to_faction_key_map, faction_to_subculture_map, no_subculture, most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, all_faction_elements=None): # Added all_faction_elements
    """
    Ensures all factions have a 'subculture' attribute. If missing, attempts to assign one
    based on existing mappings or a fallback to the most common subculture.
    Returns the count of subcultures added/updated.
    """
    if no_subculture:
        print("Skipping subculture attribute assignment due to --no-subculture flag.")
        return 0

    added_count = 0
    # Determine which list of factions to iterate over
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    for faction in factions_to_iterate:
        added_count += _ensure_subculture_attributes_for_faction(
            faction, screen_name_to_faction_key_map, faction_to_subculture_map,
            most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map
        )
    return added_count


def fix_duplicate_garrison_units(root, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                                 faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                 culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                                 heritage_to_factions_map, faction_to_heritages_map, unit_categories, general_units):
    """
    Identifies and fixes duplicate Garrison units within each faction, replacing them with suitable alternatives.
    """
    total_fixed_count = 0
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue
        total_fixed_count += _fix_duplicate_garrison_units_for_faction(
            faction, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map, unit_categories, general_units
        )
    return total_fixed_count


def fix_duplicate_levy_units(root, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                             faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                             culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                             heritage_to_factions_map, faction_to_heritages_map, unit_to_training_level,
                             unit_categories, faction_elite_units):
    """
    Identifies and fixes duplicate Levy units within each faction, replacing them with suitable alternatives.
    """
    total_fixed_count = 0
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue
        total_fixed_count += _fix_duplicate_levy_units_for_faction(
            faction, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map, unit_to_training_level,
            unit_categories, faction_elite_units # faction_elite_units will be defaultdict(set) here
        )
    return total_fixed_count


def get_all_tiered_pools(faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                         faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                         culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                         heritage_to_factions_map, faction_to_heritages_map):
    """
    Generates and returns a list of tiered unit pools for a given faction, from most specific to most general.
    This function does NOT apply excluded_units_set filtering, as that should be done dynamically.
    """
    faction_key = screen_name_to_faction_key_map.get(faction_name)
    if not faction_key:
        return [], [] # Cannot determine pools without a faction key

    # Create a cache key for the unfiltered tiered pools
    cache_key = f"{faction_name}_unfiltered_tiered_pools"

    if cache_key in faction_pool_cache:
        return faction_pool_cache[cache_key]

    tiered_pools = []
    tiered_log_strings = []

    # Tier 1: Faction-specific units
    faction_specific_pool = faction_key_to_units_map.get(faction_key, set())
    if faction_specific_pool:
        tiered_pools.append(faction_specific_pool)
        tiered_log_strings.append(f"Tier 1: Faction-specific ({faction_name})")

    # Tier 2: Subculture-specific units
    subculture_name = faction_to_subculture_map.get(faction_key)
    if subculture_name:
        subculture_factions = subculture_to_factions_map.get(subculture_name, [])
        subculture_pool = set()
        for sc_faction_key in subculture_factions:
            subculture_pool.update(faction_key_to_units_map.get(sc_faction_key, set()))
        if subculture_pool:
            tiered_pools.append(subculture_pool)
            tiered_log_strings.append(f"Tier 2: Subculture-specific ({subculture_name})")

    # Tier 3: Heritage-specific units
    heritages = faction_to_heritages_map.get(faction_key, [])
    heritage_pool = set()
    for heritage_name in heritages:
        heritage_factions = heritage_to_factions_map.get(heritage_name, [])
        for h_faction_key in heritage_factions:
            heritage_pool.update(faction_key_to_units_map.get(h_faction_key, set()))
    if heritage_pool:
        tiered_pools.append(heritage_pool)
        tiered_log_strings.append(f"Tier 3: Heritage-specific ({', '.join(heritages)})")

    # Tier 4: Culture-specific units (from cultures.xml)
    culture_names = culture_to_faction_map.get(faction_name) # This map is faction_name -> list of cultures
    culture_pool = set()
    if culture_names:
        for culture_name in culture_names:
            factions_in_culture = culture_to_faction_map.get(culture_name, []) # This map is culture_name -> list of faction_names
            for c_faction_name in factions_in_culture:
                c_faction_key = screen_name_to_faction_key_map.get(c_faction_name)
                if c_faction_key:
                    culture_pool.update(faction_key_to_units_map.get(c_faction_key, set()))
    if culture_pool:
        tiered_pools.append(culture_pool)
        tiered_log_strings.append(f"Tier 4: Culture-specific (from Cultures.xml)")

    # Tier 5: Global pool (all units)
    # This is typically passed as `all_units` to the main function, but for caching,
    # we need to ensure it's available. For now, assume it's passed in.
    # For now, this function only builds pools based on faction_key_to_units_map.
    # The global pool will be added by the caller if needed.

    # Cache the result
    faction_pool_cache[cache_key] = (tiered_pools, tiered_log_strings)

    return tiered_pools, tiered_log_strings


def get_cached_faction_working_pool(faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                    culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                                    heritage_to_factions_map, faction_to_heritages_map, log_prefix="",
                                    required_classes=None, unit_to_class_map=None): # Added required_classes and unit_to_class_map
    """
    Retrieves or generates the tiered unit pools for a faction, applying exclusions.
    Caches the unfiltered tiered pools.
    Returns the combined working pool, log string, and the unfiltered tiered pools.
    """
    # 1. Get unfiltered pools (this is cached internally by get_all_tiered_pools)
    unfiltered_tiered_pools, tiered_log_strings = get_all_tiered_pools(
        faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
        faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
        culture_to_faction_map, set(), faction_to_heritage_map,
        heritage_to_factions_map, faction_to_heritages_map
    )

    # 2. Create a cache key for the *filtered* result
    cache_key_parts = [faction_name]
    if excluded_units_set:
        # Use a hash of the sorted set to keep the key length manageable
        excl_hash = hashlib.sha1(','.join(sorted(list(excluded_units_set))).encode()).hexdigest()[:8]
        cache_key_parts.append(f"excl_{excl_hash}")
    if required_classes and unit_to_class_map:
        cache_key_parts.append("classes:" + ",".join(sorted(required_classes)))
    cache_key = "|".join(cache_key_parts)

    # 3. Check if the filtered result is already cached
    if cache_key in faction_pool_cache:
        working_pool, log_string_for_pool = faction_pool_cache[cache_key]
    else:
        # 4. If not, create the filtered pool and cache it
        working_pool = set()
        filtered_log_strings = []
        for i, pool in enumerate(unfiltered_tiered_pools):
            filtered_pool = pool
            if excluded_units_set:
                filtered_pool = filtered_pool - excluded_units_set
            if filtered_pool:
                working_pool.update(filtered_pool)
                if i < len(tiered_log_strings):
                    filtered_log_strings.append(tiered_log_strings[i])

        # Apply required_classes filtering if specified
        if required_classes and unit_to_class_map:
            initial_size = len(working_pool)
            filtered_by_class_pool = {
                unit_key for unit_key in working_pool
                if unit_to_class_map.get(unit_key) in required_classes
            }
            working_pool = filtered_by_class_pool
            if len(working_pool) < initial_size:
                print(f"    -> {log_prefix} Further filtered pool by required classes {required_classes}. Reduced from {initial_size} to {len(working_pool)} units.")

        log_string_for_pool = "; ".join(filtered_log_strings)
        faction_pool_cache[cache_key] = (working_pool, log_string_for_pool)

    # 5. Construct final log string and return
    log_faction_str = f"{log_prefix} Faction '{faction_name}' (Pools: {log_string_for_pool})"
    return working_pool, log_faction_str, unfiltered_tiered_pools


def get_conscript_keys_in_faction(faction_element: ET.Element) -> set[str]:
    """
    Collects keys from 'conscript' unit tags: Levies, Garrison.
    These keys can be shared between Levies and Garrisons, but not with professional units.

    Args:
        faction_element (ET.Element): The <Faction> XML element.

    Returns:
        set[str]: A set of all 'conscript' unit keys currently used in the faction.
    """
    used_keys = set()
    unit_tags = ['Levies', 'Garrison']
    for tag_name in unit_tags:
        for element in faction_element.findall(tag_name):
            key = element.get('key')
            if key:
                used_keys.add(key)
    return used_keys


def get_culture_to_faction_map_from_xml(cultures_xml_path):
    """
    Parses the Cultures XML file and returns a map from culture name to a list of faction names.
    """
    culture_to_faction_map = defaultdict(list)
    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for culture_element in root.findall('.//culture'):
            culture_name = culture_element.get('name')
            if culture_name:
                for faction_element in culture_element.findall('faction'):
                    faction_name = faction_element.get('name')
                    if faction_name:
                        culture_to_faction_map[culture_name].append(faction_name)
    except FileNotFoundError:
        print(f"Error: Cultures XML file not found at '{cultures_xml_path}'.")
        return defaultdict(list)
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file {cultures_xml_path}: {e}")
        return defaultdict(list)
    return culture_to_faction_map


def get_faction_heritage_maps_from_xml(cultures_xml_path):
    """
    Parses the Cultures XML file and returns two maps:
    1. faction_to_heritage_map: faction_key -> heritage_name
    2. heritage_to_factions_map: heritage_name -> list of faction_keys
    """
    faction_to_heritage_map = {}
    heritage_to_factions_map = defaultdict(list)
    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for heritage_element in root.findall('.//heritage'):
            heritage_name = heritage_element.get('name')
            if heritage_name:
                for culture_element in heritage_element.findall('culture'):
                    culture_name = culture_element.get('name')
                    if culture_name:
                        for faction_element in culture_element.findall('faction'):
                            faction_key = faction_element.get('key') # Assuming 'key' is used here
                            if faction_key:
                                faction_to_heritage_map[faction_key] = heritage_name
                                heritage_to_factions_map[heritage_name].append(faction_key)
    except FileNotFoundError:
        print(f"Error: Cultures XML file not found at '{cultures_xml_path}'.")
        return {}, defaultdict(list)
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file {cultures_xml_path}: {e}")
        return {}, defaultdict(list)
    return faction_to_heritage_map, defaultdict(list)


def get_factions_from_cultures_xml(cultures_xml_path):
    """
    Parses the Cultures XML file and returns a set of all faction names defined within it.
    """
    factions = set()
    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for culture_element in root.findall('.//culture'):
            for faction_element in culture_element.findall('faction'):
                faction_name = faction_element.get('name')
                if faction_name:
                    factions.add(faction_name)
    except FileNotFoundError:
        print(f"Error: Cultures XML file not found at '{cultures_xml_path}'.")
        return set()
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file {cultures_xml_path}: {e}")
        return set()
    return factions


def get_professional_keys_in_faction(faction_element: ET.Element) -> set[str]:
    """
    Collects keys from 'professional' unit tags: General, Knights, MenAtArm.
    These keys must be unique across the entire faction roster.

    Args:
        faction_element (ET.Element): The <Faction> XML element.

    Returns:
        set[str]: A set of all 'professional' unit keys currently used in the faction.
    """
    used_keys = set()
    unit_tags = ['General', 'Knights', 'MenAtArm']
    for tag_name in unit_tags:
        for element in faction_element.findall(tag_name):
            key = element.get('key')
            if key:
                used_keys.add(key)
    return used_keys


def merge_duplicate_factions(root, screen_name_to_faction_key_map):
    """
    Identifies factions with duplicate names (after fuzzy matching) and merges their contents.
    Keeps the first encountered faction and removes subsequent duplicates.
    """
    merged_count = 0
    seen_factions = {} # {faction_name: first_faction_element}
    factions_to_remove = []

    # Iterate over a copy of the list to allow modification during iteration
    for faction_element in list(root.findall('Faction')):
        faction_name = faction_element.get('name')
        if not faction_name:
            continue

        if faction_name in seen_factions:
            # Duplicate found, merge its children into the first one
            first_faction_element = seen_factions[faction_name]
            print(f"  -> Merging duplicate faction '{faction_name}'.")

            # Collect existing children keys/types in the primary faction to avoid direct duplicates
            existing_children_identifiers = set()
            for child in first_faction_element:
                if child.tag == 'MenAtArm' and child.get('type'):
                    existing_children_identifiers.add((child.tag, child.get('type')))
                elif child.get('key'):
                    existing_children_identifiers.add((child.tag, child.get('key')))
                elif child.tag == 'Garrison' and child.get('level'):
                    existing_children_identifiers.add((child.tag, child.get('level')))
                elif child.tag == 'Levies': # Levies are unique by tag, not key/type
                    existing_children_identifiers.add((child.tag, ''))


            for child_to_move in list(faction_element): # Iterate over a copy
                identifier = None
                if child_to_move.tag == 'MenAtArm' and child_to_move.get('type'):
                    identifier = (child_to_move.tag, child_to_move.get('type'))
                elif child_to_move.get('key'):
                    identifier = (child_to_move.tag, child_to_move.get('key'))
                elif child_to_move.tag == 'Garrison' and child_to_move.get('level'):
                    identifier = (child_to_move.tag, child_to_move.get('level'))
                elif child_to_move.tag == 'Levies':
                    identifier = (child_to_move.tag, '')

                if identifier and identifier not in existing_children_identifiers:
                    first_faction_element.append(child_to_move)
                    existing_children_identifiers.add(identifier)
                else:
                    # If it's a duplicate child, remove it from the source faction
                    # print(f"    - Skipping duplicate child <{child_to_move.tag}> (key/type: {child_to_move.get('key') or child_to_move.get('type')}) in merged faction.")
                    faction_element.remove(child_to_move) # Remove from the source faction

            factions_to_remove.append(faction_element)
            merged_count += 1
        else:
            seen_factions[faction_name] = faction_element

    for faction_element in factions_to_remove:
        root.remove(faction_element)

    if merged_count > 0:
        print(f"Merged {merged_count} duplicate factions.")
    return merged_count


def remove_factions_not_in_cultures(root, culture_factions, screen_name_to_faction_key_map, all_faction_elements=None):
    """
    Removes factions from the XML that are not present in the culture_factions set.
    """
    removed_count = 0
    factions_to_remove = []

    # Use cached faction elements if provided, otherwise find them normally
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    for faction_element in factions_to_iterate:
        faction_name = faction_element.get('name')
        if faction_name == "Default":
            continue # Always keep the Default faction

        if faction_name not in culture_factions:
            # Check if the faction_name is actually a faction_key and its screen name is in culture_factions
            # This handles cases where the XML might use keys instead of screen names
            is_valid_by_key = False
            for db_key, db_screen_name in screen_name_to_faction_key_map.items():
                if faction_name == db_key and db_screen_name in culture_factions:
                    is_valid_by_key = True
                    break
            if not is_valid_by_key:
                factions_to_remove.append(faction_element)
                print(f"  -> Removing faction '{faction_name}' not found in Cultures.xml.")
                removed_count += 1

    for faction_element in factions_to_remove:
        root.remove(faction_element)
    return removed_count


def populate_or_remove_keyless_tags(root, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                    culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                                    heritage_to_factions_map, faction_to_heritages_map,
                                    # unit selection specific args
                                    general_units, unit_stats_map, unit_categories, unit_to_training_level,
                                    faction_elite_units, ck3_maa_definitions, unit_to_class_map, unit_to_description_map,
                                    categorized_units, unit_to_tier_map, all_units):
    """
    Finds unit tags missing a 'key' attribute and attempts to populate them using appropriate unit selectors.
    If a suitable unit cannot be found, the tag is removed to ensure XML validity.
    Returns the number of keys populated and tags removed.
    """
    populated_count = 0
    removed_count = 0
    unit_tags_to_check = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if not faction_name:
            continue

        working_pool, _, _ = get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map
        )
        used_units = {el.get('key') for el in faction if el.get('key')}

        for tag_name in unit_tags_to_check:
            for element in list(faction.findall(tag_name)):
                if 'key' not in element.attrib or not element.get('key'):
                    new_key = None
                    # Import unit_selector locally to avoid circular import issues
                    from mapper_tools import unit_selector

                    if tag_name == 'General':
                        general_rank = element.get('rank')
                        try:
                            rank_int = int(general_rank) if general_rank else 1
                        except ValueError:
                            rank_int = 1
                        # Create a pool of general-eligible units from the working pool
                        general_pool = {unit for unit in working_pool if unit in general_units} - used_units
                        if general_pool:
                            new_key = unit_selector.select_best_unit_from_pool(
                                general_pool, rank=rank_int, unit_stats_map=unit_stats_map
                            )
                    elif tag_name == 'Knights':
                        knight_rank = element.get('rank')
                        try:
                            rank_int = int(knight_rank) if knight_rank else 1
                        except ValueError:
                            rank_int = 1
                        # Prioritize heavy/shock cavalry for knights
                        knight_class_priorities = ['cav_shock', 'cav_heavy', 'cav_melee']
                        for knight_class in knight_class_priorities:
                            class_pool = {unit for unit in working_pool if unit_to_class_map.get(unit) == knight_class}
                            available_pool = class_pool - used_units
                            if available_pool:
                                new_key = unit_selector.select_best_unit_from_pool(
                                    available_pool, rank=rank_int, unit_stats_map=unit_stats_map
                                )
                                if new_key:
                                    break # Use the first found suitable unit
                    elif tag_name == 'Levies':
                        faction_elites = faction_elite_units.get(faction_name, set()) if faction_elite_units else set()
                        new_key = unit_selector.find_best_levy_replacement(
                            working_pool, unit_to_training_level, unit_categories,
                            exclude_units=used_units | faction_elites
                        )
                    elif tag_name == 'Garrison':
                        garrison_level = element.get('level')
                        try:
                            level_int = int(garrison_level) if garrison_level else 1
                        except ValueError:
                            level_int = 1
                        new_key = unit_selector.find_best_garrison_replacement(
                            working_pool, unit_categories,
                            exclude_units=used_units | general_units
                        )
                    elif tag_name == 'MenAtArm':
                        maa_type = element.get('type')
                        if maa_type:
                            # Use high-confidence logic similar to processing_passes.py
                            internal_type = ck3_maa_definitions.get(maa_type)
                            expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(maa_type) or \
                                                      (mappings.CK3_TYPE_TO_ATTILA_CLASS.get(internal_type) if internal_type else None)

                            # Tiered selection logic
                            tiered_candidates = []
                            if working_pool:
                                # Tier 1: Exact class match
                                if expected_attila_classes:
                                    tier1_pool = {u for u in working_pool if unit_to_class_map.get(u) in expected_attila_classes}
                                    if tier1_pool:
                                        tiered_candidates.append(('Class Match', tier1_pool))

                                # Tier 2: Category match
                                attila_roles = mappings.CK3_TYPE_TO_ATTILA_ROLES.get(maa_type) or \
                                               (mappings.CK3_TYPE_TO_ATTILA_ROLES.get(internal_type) if internal_type else [])
                                tier2_pool = set()
                                for role in attila_roles:
                                    tier2_pool.update(categorized_units.get(role, []))
                                tier2_pool &= working_pool
                                if tier2_pool:
                                    tiered_candidates.append(('Role Match', tier2_pool))

                                # Tier 3: Description keywords
                                if unit_to_description_map:
                                    tier3_pool = unit_selector.find_units_by_keywords(working_pool, unit_to_description_map, [maa_type, internal_type])
                                    if tier3_pool:
                                        tiered_candidates.append(('Description Keywords', tier3_pool))

                                # Select from the highest priority tier
                                for tier_name, candidate_pool in tiered_candidates:
                                    filtered_pool = candidate_pool - used_units
                                    if filtered_pool:
                                        new_key = unit_selector.select_best_unit_by_tier(
                                            filtered_pool, unit_to_tier_map, tier=None # Use default tier logic
                                        )
                                        if new_key:
                                            # print(f"    - Found keyless MAA replacement '{new_key}' for type '{maa_type}' (Pool: {tier_name}).")
                                            break # Stop searching tiers once a unit is found

                    # If the ideal search found a unit that's already used, nullify it to trigger the fallback.
                    if new_key and new_key in used_units:
                        new_key = None

                    # If no unique ideal unit was found, attempt a global last-resort fallback.
                    if not new_key:
                        print(f"    -> PRE-VALIDATION: Could not find ideal unit for keyless <{tag_name}> in faction '{faction_name}'. Attempting global fallback.")
                        global_fallback_pool = all_units - (excluded_units_set if excluded_units_set else set()) - used_units
                        if global_fallback_pool:
                            new_key = random.choice(list(global_fallback_pool))
                            print(f"    -> PRE-VALIDATION: Assigned random global unit '{new_key}' as last resort.")

                    # After all attempts, if a key was found, assign it. Otherwise, log a critical failure.
                    if new_key:
                        element.set('key', new_key)
                        used_units.add(new_key)
                        populated_count += 1
                    else:
                        # This is a critical failure state where no units are available at all.
                        # We will remove the tag to prevent a key-validation error, but this will likely cause a missing-element error.
                        # This is unavoidable if all unit pools are empty.
                        print(f"  -> PRE-VALIDATION: CRITICAL - No units available in any pool for <{tag_name}> in faction '{faction_name}'. Removing tag.")
                        faction.remove(element)
                        removed_count += 1

    if populated_count > 0:
        print(f"  -> PRE-VALIDATION: Populated {populated_count} unit elements missing the 'key' attribute.")
    if removed_count > 0:
        print(f"  -> PRE-VALIDATION: Removed {removed_count} keyless unit elements that could not be populated.")

    return populated_count, removed_count


def validate_and_fix_faction_names(root, faction_key_to_screen_name_map, unit_to_faction_key_map, culture_factions):
    """
    Validates faction names in the XML against known faction names from the DB and Cultures.xml.
    Attempts to fix fuzzy matches and removes invalid factions.
    Returns the number of factions fixed and a map of old_name -> new_name.
    """
    fixed_count = 0
    faction_name_map = {}
    valid_db_faction_names = set(faction_key_to_screen_name_map.values())
    all_valid_faction_names = valid_db_faction_names.union(culture_factions)

    factions_to_process = list(root.findall('Faction')) # Create a copy to iterate while modifying

    for faction_element in factions_to_process:
        current_name = faction_element.get('name')
        if not current_name:
            print(f"  -> WARNING: Faction element found without a 'name' attribute. Removing.")
            root.remove(faction_element)
            fixed_count += 1
            continue

        if current_name in all_valid_faction_names:
            continue # Name is already valid

        # Attempt fuzzy matching
        best_match, score = shared_utils.find_best_fuzzy_match(current_name, list(all_valid_faction_names), threshold=0.8)

        if best_match and score > 0.8: # A good enough match
            print(f"  -> INFO: Faction name '{current_name}' fuzzy-matched to '{best_match}' (score: {score:.2f}). Updating.")
            faction_element.set('name', best_match)
            faction_name_map[current_name] = best_match
            fixed_count += 1
        else:
            # If no good match, check if it's a valid key that just doesn't have a screen name
            if current_name in faction_key_to_screen_name_map:
                # It's a valid key, but not a screen name. Keep it as is.
                continue
            # If it's not a valid screen name and not a valid key, remove it.
            print(f"  -> WARNING: Faction '{current_name}' is not a valid faction name or key and could not be fuzzy-matched. Removing.")
            root.remove(faction_element)
            fixed_count += 1

    return fixed_count, faction_name_map


def validate_faction_unit_tags(root, is_submod_mode, no_garrison, ck3_maa_definitions, unit_to_class_map, unit_categories):
    """
    Validates all faction unit tags for structural integrity.
    Checks for missing tags, missing key attributes, and missing max attributes.
    Returns a list of validation failures for the final fix pass.
    """
    validation_failures = []

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if not faction_name or faction_name == "Default":
            continue

        # Skip factions that are managed by main mod in submod mode
        # Note: This would need to be passed in if needed, but for now we check all factions

        # Check for missing required tags
        required_tags = ['General', 'Knights', 'Levies', 'MenAtArm']
        if not no_garrison:
            required_tags.append('Garrison')

        for tag_name in required_tags:
            elements = faction.findall(tag_name)
            if not elements:
                # Missing tag entirely
                validation_failures.append({
                    'faction_element': faction,
                    'tag_name': tag_name,
                    'element': None,
                    'validation_error': 'missing_tag',
                    'unit_role_description': f"A required {tag_name} unit"
                })
            else:
                # Check for missing key attributes
                for element in elements:
                    if 'key' not in element.attrib or not element.get('key'):
                        # Missing key attribute
                        unit_role_description = f"A {tag_name} unit"
                        if tag_name == 'MenAtArm':
                            maa_type = element.get('type')
                            if maa_type:
                                unit_role_description = maa_type

                        validation_failures.append({
                            'faction_element': faction,
                            'tag_name': tag_name,
                            'element': element,
                            'validation_error': 'missing_key_attribute',
                            'unit_role_description': unit_role_description,
                            'rank': element.get('rank'),
                            'level': element.get('level')
                        })

        # Check MenAtArm tags for missing max attributes (non-siege units)
        maa_elements = faction.findall('MenAtArm')
        for element in maa_elements:
            unit_key = element.get('key')
            maa_definition_name = element.get('type')

            if not unit_key or not maa_definition_name:
                # Skip if key or type is missing (already caught above)
                continue

            # Determine if this is a siege unit
            internal_type = ck3_maa_definitions.get(maa_definition_name) if ck3_maa_definitions else None
            is_siege_by_ck3_type = (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) is None) or \
                                   (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type) is None)
            unit_class = unit_to_class_map.get(unit_key) if unit_to_class_map else None
            is_siege_by_attila_class = (unit_class == 'art_siege')
            unit_category = unit_categories.get(unit_key) if unit_categories else None
            is_siege_by_attila_category = (unit_category == 'artillery')
            is_siege = is_siege_by_ck3_type or is_siege_by_attila_class or is_siege_by_attila_category

            # Non-siege units must have 'max' attribute
            if not is_siege and ('max' not in element.attrib or not element.get('max')):
                validation_failures.append({
                    'faction_element': faction,
                    'tag_name': 'MenAtArm',
                    'element': element,
                    'validation_error': 'missing_max_attribute',
                    'unit_role_description': maa_definition_name,
                    'maa_definition_name': maa_definition_name
                })

    return validation_failures


def _ensure_subculture_attributes_for_faction(faction_element, screen_name_to_faction_key_map, faction_to_subculture_map, most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map):
    """
    Ensures a single faction has a 'subculture' attribute.
    Returns 1 if a subculture was added/updated, 0 otherwise.
    """
    faction_name = faction_element.get('name')
    if faction_name == "Default":
        return 0

    current_subculture = faction_element.get('subculture')
    faction_key = screen_name_to_faction_key_map.get(faction_name)

    if faction_key and faction_key in faction_to_subculture_map:
        # Use the subculture from the DB if available
        db_subculture = faction_to_subculture_map[faction_key]
        if current_subculture != db_subculture:
            faction_element.set('subculture', db_subculture)
            # print(f"  -> Set subculture for '{faction_name}' to '{db_subculture}' (from DB).")
            return 1
    elif not current_subculture:
        # Attempt to infer subculture from heritage or culture if not in DB map
        assigned_subculture = None
        if faction_key:
            heritages = faction_to_heritages_map.get(faction_key, [])
            for heritage_name in heritages:
                # Try to find a subculture associated with this heritage
                # This is a heuristic, as there's no direct heritage->subculture map
                # We'll look for factions in this heritage that *do* have a subculture
                for h_faction_key in heritage_to_factions_map.get(heritage_name, []):
                    if h_faction_key in faction_to_subculture_map:
                        assigned_subculture = faction_to_subculture_map[h_faction_key]
                        # print(f"  -> Inferred subculture for '{faction_name}' to '{assigned_subculture}' (from heritage '{heritage_name}').")
                        break
                if assigned_subculture:
                    break

        if not assigned_subculture:
            # Fallback to the most common subculture if all else fails
            if most_common_faction_key and most_common_faction_key in faction_to_subculture_map:
                assigned_subculture = faction_to_subculture_map[most_common_faction_key]
                # print(f"  -> Fallback: Set subculture for '{faction_name}' to '{assigned_subculture}' (most common).")
            elif faction_to_subculture_map:
                # If no most common, pick any available subculture
                assigned_subculture = next(iter(faction_to_subculture_map.values()))
                # print(f"  -> Fallback: Set subculture for '{faction_name}' to '{assigned_subculture}' (any available).")

        if assigned_subculture:
            faction_element.set('subculture', assigned_subculture)
            return 1
    return 0


def _fix_duplicate_garrison_units_for_faction(faction_element, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                                              faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                              culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                                              heritage_to_factions_map, faction_to_heritages_map, unit_categories, general_units):
    """
    Identifies and fixes duplicate Garrison units within a single faction, replacing them with suitable alternatives.
    Returns the number of fixes made for this faction.
    """
    fixed_for_faction = 0
    faction_name = faction_element.get('name')

    garrison_tags = faction_element.findall('Garrison')
    if not garrison_tags:
        return 0

    garrisons_by_level = defaultdict(list)
    for g_tag in garrison_tags:
        level = g_tag.get('level')
        if level:
            garrisons_by_level[level].append(g_tag)

    for level, tags_at_level in garrisons_by_level.items():
        seen_garrison_keys = set()
        duplicates_to_fix = []

        for garrison in tags_at_level:
            key = garrison.get('key')
            if key and key in seen_garrison_keys:
                duplicates_to_fix.append(garrison)
            elif key:
                seen_garrison_keys.add(key)

        if not duplicates_to_fix:
            continue

        print(f"  -> Fixing {len(duplicates_to_fix)} duplicate garrison units at level '{level}' in faction '{faction_name}'.")

        # Get the working pool for this faction
        working_pool, _, _ = get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map, log_prefix=f"(Duplicate Garrisons L{level})"
        )

        # Exclude general units from the garrison pool
        current_garrison_pool = working_pool - general_units

        for duplicate_garrison in duplicates_to_fix:
            original_key = duplicate_garrison.get('key')
            # Try to find a replacement unit
            replacement_unit = unit_selector.find_best_garrison_replacement(
                current_garrison_pool, unit_categories,
                exclude_units=seen_garrison_keys # Exclude already assigned garrisons
            )

            if replacement_unit:
                duplicate_garrison.set('key', replacement_unit)
                seen_garrison_keys.add(replacement_unit) # Add new unit to seen list
                fixed_for_faction += 1
                # print(f"    - Replaced duplicate garrison '{original_key}' with '{replacement_unit}'.")
            else:
                # If no suitable replacement, remove the duplicate tag entirely
                faction_element.remove(duplicate_garrison)
                fixed_for_faction += 1
                # print(f"    - Removed duplicate garrison '{original_key}' (no suitable replacement found).")

        # Ensure minimum of 3 unique garrison units per level
        units_to_add = 3 - len(seen_garrison_keys)
        for _ in range(units_to_add):
            # Find a new unit, excluding already assigned ones
            new_unit = unit_selector.find_best_garrison_replacement(
                current_garrison_pool, unit_categories,
                exclude_units=seen_garrison_keys
            )
            if new_unit:
                # Create new Garrison element
                ET.SubElement(faction_element, 'Garrison', key=new_unit, level=level, percentage='0', max='LEVY')
                seen_garrison_keys.add(new_unit)
                fixed_for_faction += 1
                # print(f"    - Added new garrison unit '{new_unit}' to level '{level}'.")

    return fixed_for_faction


def _fix_duplicate_levy_units_for_faction(faction_element, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                                          faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                          culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                                          heritage_to_factions_map, faction_to_heritages_map, unit_to_training_level,
                                          unit_categories, faction_elite_units):
    """
    Identifies and fixes duplicate Levy units within a single faction, replacing them with suitable alternatives.
    Returns the number of fixes made for this faction.
    """
    fixed_for_faction = 0
    faction_name = faction_element.get('name')

    levy_tags = faction_element.findall('Levies')
    if not levy_tags:
        return 0

    seen_levy_keys = set()
    duplicates_to_fix = []

    for levy in levy_tags:
        key = levy.get('key')
        if key and key in seen_levy_keys:
            duplicates_to_fix.append(levy)
        elif key:
            seen_levy_keys.add(key)

    if not duplicates_to_fix:
        return 0

    print(f"  -> Fixing {len(duplicates_to_fix)} duplicate levy units in faction '{faction_name}'.")

    # Get the working pool for this faction
    working_pool, _, _ = get_cached_faction_working_pool(
        faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
        faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
        culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
        heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Duplicate Levies)"
    )

    # Exclude elite units from the levy pool
    current_levy_pool = working_pool - faction_elite_units

    for duplicate_levy in duplicates_to_fix:
        original_key = duplicate_levy.get('key')
        # Try to find a replacement unit
        replacement_unit = unit_selector.find_best_levy_replacement(
            current_levy_pool, unit_to_training_level, unit_categories,
            exclude_units=seen_levy_keys # Exclude already assigned levies
        )

        if replacement_unit:
            duplicate_levy.set('key', replacement_unit)
            seen_levy_keys.add(replacement_unit) # Add new unit to seen list
            fixed_for_faction += 1
            # print(f"    - Replaced duplicate levy '{original_key}' with '{replacement_unit}'.")
        else:
            # If no suitable replacement, remove the duplicate tag entirely
            faction_element.remove(duplicate_levy)
            fixed_for_faction += 1
            # print(f"    - Removed duplicate levy '{original_key}' (no suitable replacement found).")

    return fixed_for_faction


def _reorder_attributes_in_all_tags_for_element(element):
    """
    Reorders attributes within a single XML element to a consistent order.
    Returns 1 if attributes were reordered, 0 otherwise.
    """
    if not element.attrib:
        return 0

    # Define a desired order for attributes for each tag type
    if element.tag == 'Faction':
        desired_order = ['name', 'subculture']
    elif element.tag == 'General':
        desired_order = ['key', 'rank', 'num_guns']
    elif element.tag == 'Knights':
        desired_order = ['key', 'rank', 'num_guns']
    elif element.tag == 'Levies':
        desired_order = ['key', 'percentage', 'max', 'num_guns']
    elif element.tag == 'Garrison':
        desired_order = ['key', 'level', 'percentage', 'max', 'num_guns']
    elif element.tag == 'MenAtArm':
        desired_order = ['type', 'key', 'max', 'siege', 'siege_engine_per_unit', 'num_guns']
    else: # Generic order for other tags like <Factions>
        desired_order = [
            'name', 'key', 'type', 'rank', 'level', 'percentage', 'max', 'siege',
            'siege_engine_per_unit', 'num_guns', 'subculture', 'submod_tag', 'submod_addon_tag'
        ]

    # Get current attributes and sort them based on desired_order, then alphabetically
    current_attribs = list(element.attrib.items())
    sorted_attribs = sorted(current_attribs, key=lambda item: (
        desired_order.index(item[0]) if item[0] in desired_order else len(desired_order),
        item[0] # Secondary sort by name for attributes not in desired_order
    ))

    # Check if the order has changed
    if list(element.attrib.items()) != sorted_attribs:
        element.attrib.clear() # Clear existing attributes
        for key, value in sorted_attribs:
            element.set(key, value) # Add them back in the desired order
        return 1
    return 0


def _remove_duplicate_ranked_units_for_faction(faction_element):
    """
    Removes duplicate General and Knights units within a single faction based on their 'rank' attribute.
    Returns the count of units removed for this faction.
    """
    removed_for_faction = 0

    # Process Generals
    seen_general_ranks = set()
    generals_to_remove = []
    for general in faction_element.findall('General'):
        rank = general.get('rank')
        if rank:
            if rank in seen_general_ranks:
                generals_to_remove.append(general)
            else:
                seen_general_ranks.add(rank)
    for general in generals_to_remove:
        faction_element.remove(general)
        removed_for_faction += 1

    # Process Knights
    seen_knights_ranks = set()
    knights_to_remove = []
    for knight in faction_element.findall('Knights'):
        rank = knight.get('rank')
        if rank:
            if rank in seen_knights_ranks:
                knights_to_remove.append(knight)
            else:
                seen_knights_ranks.add(rank)
    for knight in knights_to_remove:
        faction_element.remove(knight)
        removed_for_faction += 1

    if removed_for_faction > 0:
        print(f"  -> Removed {removed_for_faction} duplicate ranked units from faction '{faction_element.get('name')}'.")
    return removed_for_faction


def _remove_excluded_unit_keys_for_faction(faction_element, excluded_units_set):
    """
    Removes the 'key' attribute from any unit tags within a single faction
    whose key is in the excluded_units_set.
    Returns the count of keys removed for this faction.
    """
    removed_for_faction = 0
    for element in faction_element.iter():
        if element.tag in ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']:
            unit_key = element.get('key')
            if unit_key and unit_key in excluded_units_set:
                del element.attrib['key']
                removed_for_faction += 1
    return removed_for_faction


def _remove_zero_percentage_tags_for_faction(faction_element):
    """
    Removes <Levies> and <Garrison> tags within a single faction that have a 'percentage' attribute of '0'.
    For Levies, it ensures at least one tag remains to satisfy the schema.
    Returns the count of tags removed for this faction.
    """
    removed_for_faction = 0

    # Handle Levies separately to ensure at least one remains for schema compliance.
    levy_tags = faction_element.findall('Levies')
    if levy_tags:
        levies_to_remove = [tag for tag in levy_tags if tag.get('percentage') == '0']
        # Only remove zero-percentage levies if it won't result in removing ALL levy tags.
        if len(levies_to_remove) < len(levy_tags):
            for tag in levies_to_remove:
                faction_element.remove(tag)
                removed_for_faction += 1

    # Handle Garrisons: these can all be safely removed as they are not required for base schema validity.
    garrison_tags_to_remove = [tag for tag in faction_element.findall('Garrison') if tag.get('percentage') == '0']
    for tag in garrison_tags_to_remove:
        faction_element.remove(tag)
        removed_for_faction += 1

    return removed_for_faction


def remove_core_unit_tags(root, factions_in_main_mod):
    """
    In submod mode, removes General, Knights, Levies, and Garrison tags from factions
    that are also present in the main mod. This ensures the submod only adds MAA.
    """
    removed_count = 0
    tags_to_remove = ['General', 'Knights', 'Levies', 'Garrison']

    for faction_element in root.findall('Faction'):
        faction_name = faction_element.get('name')
        if faction_name == "Default":
            continue

        if faction_name in factions_in_main_mod:
            for tag_name in tags_to_remove:
                for unit_tag in list(faction_element.findall(tag_name)):
                    faction_element.remove(unit_tag)
                    removed_count += 1
            # print(f"  -> Removed core unit tags (General, Knights, Levies, Garrison) from faction '{faction_name}' (present in main mod).")

    if removed_count > 0:
        print(f"Removed {removed_count} core unit tags from submod factions that are present in the main mod.")
    return removed_count


def remove_duplicate_men_at_arm_tags(root):
    """
    Removes duplicate MenAtArm tags within each faction based on their 'type' attribute.
    """
    removed_count = 0
    for faction in root.findall('Faction'):
        seen_maa_types = set()
        maa_tags_to_remove = []
        for maa in faction.findall('MenAtArm'):
            maa_type = maa.get('type')
            if maa_type:
                if maa_type in seen_maa_types:
                    maa_tags_to_remove.append(maa)
                else:
                    seen_maa_types.add(maa_type)
        for maa in maa_tags_to_remove:
            faction.remove(maa)
            removed_count += 1
    return removed_count


def remove_duplicate_ranked_units(root):
    """
    Removes duplicate General and Knights units within each faction based on their 'rank' attribute.
    If multiple units have the same rank, only the first one is kept.
    """
    total_removed_count = 0
    for faction in root.findall('Faction'):
        total_removed_count += _remove_duplicate_ranked_units_for_faction(faction)
    return total_removed_count


def remove_excluded_factions(root, excluded_factions, screen_name_to_faction_key_map, all_faction_elements=None):
    """
    Removes factions from the XML that are in the excluded_factions set.
    """
    removed_count = 0
    factions_to_remove = []
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')
    for faction_element in factions_to_iterate:
        faction_name = faction_element.get('name')
        if faction_name in excluded_factions:
            factions_to_remove.append(faction_element)
            print(f"  -> Removing excluded faction: '{faction_name}'")
            removed_count += 1
        else:
            # Also check if the faction key is in the excluded list (if screen_name_to_faction_key_map is available)
            faction_key = screen_name_to_faction_key_map.get(faction_name)
            if faction_key and faction_key in excluded_factions: # Assuming excluded_factions can also contain keys
                factions_to_remove.append(faction_element)
                print(f"  -> Removing excluded faction (by key): '{faction_name}' (key: {faction_key})")
                removed_count += 1

    for faction_element in factions_to_remove:
        root.remove(faction_element)
    return removed_count


def remove_excluded_unit_keys(root, excluded_units_set):
    """
    Removes the 'key' attribute from any unit tags whose key is in the excluded_units_set.
    This forces re-selection for those slots.
    """
    removed_count = 0
    for faction in root.findall('Faction'):
        removed_count += _remove_excluded_unit_keys_for_faction(faction, excluded_units_set)
    return removed_count


def remove_factions_in_main_mod(root, main_mod_faction_maa_map, all_faction_elements=None):
    """
    In submod mode, removes factions from the submod XML that are present in the main mod
    AND have no new MenAtArm types defined in the submod.
    Returns the count of removed factions and a set of their names for syncing.
    """
    removed_count = 0
    removed_faction_names_for_sync = set()
    factions_to_remove = []

    if not main_mod_faction_maa_map:
        return 0, removed_faction_names_for_sync

    # Use cached faction elements if provided, otherwise find them normally
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    for faction_element in factions_to_iterate:
        faction_name = faction_element.get('name')
        if faction_name == "Default":
            continue

        if faction_name in main_mod_faction_maa_map:
            # Check if this faction has any MenAtArm tags that are *not* in the main mod's definition
            submod_maa_types = {maa.get('type') for maa in faction_element.findall('MenAtArm') if maa.get('type')}
            main_mod_maa_types = main_mod_faction_maa_map.get(faction_name, set())

            new_maa_types_in_submod = submod_maa_types - main_mod_maa_types

            if not new_maa_types_in_submod:
                factions_to_remove.append(faction_element)
                removed_faction_names_for_sync.add(faction_name)
                print(f"  -> Removing faction '{faction_name}' from submod as it's in main mod and has no new MenAtArm types.")
                removed_count += 1

    for faction_element in factions_to_remove:
        root.remove(faction_element)

    return removed_count, removed_faction_names_for_sync


def remove_maa_tags_present_in_main_mod(root, main_mod_faction_maa_map):
    """
    In submod mode, removes MenAtArm tags from submod factions if they are already
    defined in the main mod's Factions.xml.
    """
    removed_count = 0
    if not main_mod_faction_maa_map:
        return 0

    for faction_element in root.findall('Faction'):
        faction_name = faction_element.get('name')
        if faction_name == "Default":
            continue

        main_mod_maa_types = main_mod_faction_maa_map.get(faction_name)
        if main_mod_maa_types:
            maa_tags_to_remove = []
            for maa_tag in faction_element.findall('MenAtArm'):
                maa_type = maa_tag.get('type')
                if maa_type and maa_type in main_mod_maa_types:
                    maa_tags_to_remove.append(maa_tag)

            for maa_tag in maa_tags_to_remove:
                faction_element.remove(maa_tag)
                removed_count += 1
                # print(f"  -> Removed MenAtArm type '{maa_tag.get('type')}' from faction '{faction_name}' (present in main mod).")

    if removed_count > 0:
        print(f"Removed {removed_count} MenAtArm tags from submod factions that are present in the main mod.")
    return removed_count


def remove_zero_percentage_tags(root):
    """
    Removes <Levies> and <Garrison> tags that have a 'percentage' attribute of '0'.
    """
    total_removed_count = 0
    for faction in root.findall('Faction'):
        total_removed_count += _remove_zero_percentage_tags_for_faction(faction)
    return total_removed_count


def reorder_attributes_in_all_tags(root):
    """
    Reorders attributes within all XML tags to a consistent order.
    Returns the count of tags that had their attributes reordered.
    """
    reordered_attr_count = 0
    for element in root.iter():
        reordered_attr_count += _reorder_attributes_in_all_tags_for_element(element)
    return reordered_attr_count


def reorganize_faction_children(root):
    """
    Reorganizes the child elements within each <Faction> tag to a consistent order:
    General, Knights, Levies, Garrison, MenAtArm.
    Also sorts tags within each group by their attributes.
    Returns the count of factions that had their children reorganized.
    """
    reorganized_count = 0
    desired_order = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']

    for faction in root.findall('Faction'):
        current_children = list(faction)
        if not current_children:
            continue

        # Create a dictionary to group children by tag name
        grouped_children = defaultdict(list)
        for child in current_children:
            grouped_children[child.tag].append(child)

        new_children_order = []
        for tag_name in desired_order:
            children_for_tag = grouped_children[tag_name]
            
            # Sort children within each tag group by their attributes
            if tag_name in ['General', 'Knights']:
                # Sort by rank (numeric), then by key for stability
                children_for_tag.sort(key=lambda el: (int(el.get('rank', '0')), el.get('key', '')))
            elif tag_name == 'Garrison':
                # Sort by level (numeric), then by key for stability
                children_for_tag.sort(key=lambda el: (int(el.get('level', '0')), el.get('key', '')))
            elif tag_name == 'MenAtArm':
                # Sort by type for deterministic order
                children_for_tag.sort(key=lambda el: el.get('type', ''))
            elif tag_name == 'Levies':
                # Sort by key for deterministic order
                children_for_tag.sort(key=lambda el: el.get('key', ''))
            
            new_children_order.extend(children_for_tag)
            # Remove these from grouped_children to identify any unexpected tags
            if tag_name in grouped_children:
                del grouped_children[tag_name]

        # Add any remaining (unexpected) tags at the end, sorted by key
        for tag_name in sorted(grouped_children.keys()):
            remaining_children = grouped_children[tag_name]
            remaining_children.sort(key=lambda el: el.get('key', ''))
            new_children_order.extend(remaining_children)

        # Check if the order has changed
        if new_children_order != current_children:
            # Remove all existing children without clearing the faction element itself
            for child in list(faction):
                faction.remove(child)
            # Add children back in the new order
            for child in new_children_order:
                faction.append(child)
            reorganized_count += 1

    if reorganized_count > 0:
        print(f"Reorganized children for {reorganized_count} factions.")
    return reorganized_count


def sync_faction_structure_from_default(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, screen_name_to_faction_key_map, faction_key_to_units_map, unit_to_class_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map=None, unit_to_description_map=None, unit_stats_map=None, main_mod_faction_maa_map=None, excluded_units_set=None, faction_pool_cache=None, faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, unit_to_training_level=None):
    """
    Ensures all factions have the same basic structure (e.g., MenAtArm tags for all CK3 MAA types)
    as the 'Default' faction, or a comprehensive set if Default is empty.
    """
    default_faction = root.find("Faction[@name='Default']")
    if default_faction is None:
        print("WARNING: 'Default' faction not found. Cannot sync structure.")
        return 0

    default_maa_types = {maa.get('type') for maa in default_faction.findall('MenAtArm') if maa.get('type')}
    default_general_ranks = {int(g.get('rank')) for g in default_faction.findall('General') if g.get('rank')}
    default_knights_ranks = {int(k.get('rank')) for k in default_faction.findall('Knights') if k.get('rank')}
    default_garrison_levels = {int(g.get('level')) for g in default_faction.findall('Garrison') if g.get('level')}

    # If Default faction is empty, generate a comprehensive set of MAA types
    if not default_maa_types:
        print("INFO: Default faction has no MenAtArm types. Generating a comprehensive set for syncing.")
        for ck3_maa_type in ck3_maa_definitions.keys():
            default_maa_types.add(ck3_maa_type)
    
    # Ensure basic ranked units and garrisons are covered if missing
    if not default_general_ranks:
        default_general_ranks.update([1, 2])
    if not default_knights_ranks:
        default_knights_ranks.update([1])
    if not default_garrison_levels:
        default_garrison_levels.update([1])

    synced_count = 0
    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        # Sync MenAtArm tags
        current_maa_types = {maa.get('type') for maa in faction.findall('MenAtArm') if maa.get('type')}
        for maa_type in default_maa_types:
            if maa_type not in current_maa_types:
                ET.SubElement(faction, 'MenAtArm', type=maa_type)
                synced_count += 1

        # Sync General tags
        current_general_ranks = {int(g.get('rank')) for g in faction.findall('General') if g.get('rank')}
        for rank in default_general_ranks:
            if rank not in current_general_ranks:
                ET.SubElement(faction, 'General', rank=str(rank))
                synced_count += 1

        # Sync Knights tags
        current_knights_ranks = {int(k.get('rank')) for k in faction.findall('Knights') if k.get('rank')}
        for rank in default_knights_ranks:
            if rank not in current_knights_ranks:
                ET.SubElement(faction, 'Knights', rank=str(rank))
                synced_count += 1

        # Sync Levies tag
        if not faction.find('Levies'):
            # Get the faction's working pool for levy selection
            faction_name = faction.get('name')
            if faction_name and faction_pool_cache is not None and unit_to_training_level is not None:
                try:
                    working_pool, _, _ = get_cached_faction_working_pool(
                        faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                        faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                        culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
                        heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Levy Sync)",
                        required_classes={'inf_spear', 'inf_melee', 'inf_heavy', 'inf_bow', 'inf_sling', 'inf_javelin'}, 
                        unit_to_class_map=unit_to_class_map
                    )
                    
                    # Find a suitable levy unit from the faction's pool
                    levy_unit_key = None
                    if working_pool:
                        from mapper_tools import unit_selector
                        levy_unit_key = unit_selector.find_best_levy_replacement(
                            working_pool, unit_to_training_level, unit_categories
                        )
                    
                    if levy_unit_key:
                        ET.SubElement(faction, 'Levies', key=levy_unit_key, percentage='100', max='LEVY')
                        print(f"  -> Added missing <Levies> tag for faction '{faction_name}' with unit '{levy_unit_key}'.")
                        synced_count += 1
                    else:
                        print(f"  -> WARNING: Could not find a suitable levy unit for faction '{faction_name}'. "
                              f"Adding <Levies> tag without key (will fail validation).")
                        ET.SubElement(faction, 'Levies', percentage='100', max='LEVY')
                        synced_count += 1
                except Exception as e:
                    print(f"  -> ERROR: Failed to add <Levies> tag for faction '{faction_name}': {e}")
                    # Fallback to creating the tag without a key
                    ET.SubElement(faction, 'Levies', percentage='100', max='LEVY')
                    synced_count += 1
            else:
                # Fallback when we don't have the necessary data
                ET.SubElement(faction, 'Levies', percentage='100', max='LEVY')
                synced_count += 1

        # Sync Garrison tags
        current_garrison_levels = {int(g.get('level')) for g in faction.findall('Garrison') if g.get('level')}
        for level in default_garrison_levels:
            if level not in current_garrison_levels:
                ET.SubElement(faction, 'Garrison', level=str(level), percentage='100', max='LEVY')
                synced_count += 1

    if synced_count > 0:
        print(f"Synced {synced_count} MenAtArm, General, Knights, Levies, and Garrison tags from Default faction.")
    return synced_count


def validate_faction_structure(root, is_submod_mode, no_garrison):
    """
    Validates that each faction (except Default) has all required core unit tags.
    In submod mode, this validation is skipped.
    Returns (is_valid, errors) where is_valid is a boolean and errors is a list of strings.
    """
    if is_submod_mode:
        print("\nSkipping faction structure validation in submod mode.")
        return True, []

    required_tags = ['General', 'Knights', 'MenAtArm', 'Levies']
    if not no_garrison:
        required_tags.append('Garrison')

    errors = []
    factions_to_validate = [f for f in root.findall('Faction') if f.get('name') != 'Default']

    for faction in factions_to_validate:
        faction_name = faction.get('name')
        for tag in required_tags:
            if not faction.findall(tag):
                errors.append(f"Faction '{faction_name}' is missing required <{tag}> element(s).")

    if errors:
        return False, errors

    print("\nAll factions have the required core unit structure.")
    return True, []


def validate_levy_garrison_percentages(root: ET.Element) -> tuple[bool, list]:
    """
    Validates that the sum of percentages for Levies and Garrisons (per level) in each faction equals 100.
    
    Args:
        root (ET.Element): The root element of the parsed Factions XML.
        
    Returns:
        tuple[bool, list]: A tuple where the first element is True if all percentages are valid,
                           and the second element is a list of error messages for any invalid sums.
    """
    errors = []
    for faction in root.findall('Faction'):
        faction_name = faction.get('name', 'Unknown Faction')

        # Validate Levies
        levy_tags = faction.findall('Levies')
        if levy_tags: # Only validate if there are levy tags
            total_levy_percentage = 0
            for tag in levy_tags:
                try:
                    perc = int(tag.get('percentage', 0))
                    total_levy_percentage += perc
                except (ValueError, TypeError):
                    # If percentage is invalid, we can't validate the sum correctly.
                    # This is an error state, but let's focus on the sum check.
                    # The schema validation should ideally catch invalid percentage formats.
                    # For now, treat invalid percentage as 0 for sum calculation.
                    pass 
            if total_levy_percentage != 100:
                errors.append(f"Faction '{faction_name}': Sum of Levy percentages is {total_levy_percentage}, expected 100.")

        # Validate Garrisons by level
        garrison_tags = faction.findall('Garrison')
        garrisons_by_level = defaultdict(list)
        for g_tag in garrison_tags:
            level = g_tag.get('level')
            if level:
                garrisons_by_level[level].append(g_tag)

        for level, tags in garrisons_by_level.items():
             if tags: # Only validate if there are garrison tags for this level
                total_garrison_percentage = 0
                for tag in tags:
                    try:
                        perc = int(tag.get('percentage', 0))
                        total_garrison_percentage += perc
                    except (ValueError, TypeError):
                        # Similar to levy, treat invalid percentage as 0 for sum calculation.
                        pass
                if total_garrison_percentage != 100:
                    errors.append(f"Faction '{faction_name}': Sum of Garrison (level {level}) percentages is {total_garrison_percentage}, expected 100.")

    is_valid = len(errors) == 0
    return is_valid, errors
