import xml.etree.ElementTree as ET
import random
from collections import defaultdict, Counter
import re
import Levenshtein

from mapper_tools import shared_utils
from mapper_tools import ck3_to_attila_mappings as mappings
from mapper_tools import unit_selector
from mapper_tools import faction_xml_utils # For _get_faction_working_pool


def _normalize_levy_percentages(levy_tags):
    """
    Normalizes the 'percentage' attribute of a list of tags (Levies or Garrisons) so they sum to 100.
    If existing percentages sum to zero, it falls back to equal distribution.
    Otherwise, it uses existing percentages as weights.
    Returns True if any changes were made, False otherwise.
    """
    if not levy_tags:
        return False

    changes_made = False
    current_percentages = []
    for tag in levy_tags:
        try:
            current_percentages.append(int(tag.get('percentage', '0')))
        except ValueError:
            current_percentages.append(0) # Treat invalid percentages as 0

    total_weight = sum(current_percentages)

    if total_weight == 0:
        # Fallback to equal distribution if no weights are provided or all are zero
        num_tags = len(levy_tags)
        base_percentage = 100 // num_tags
        remainder = 100 % num_tags

        # Sort tags by key to ensure deterministic distribution of remainder
        sorted_tags = sorted(levy_tags, key=lambda x: x.get('key', ''))
        for i, tag in enumerate(sorted_tags):
            percentage = base_percentage
            if i < remainder:
                percentage += 1

            current_percentage_str = tag.get('percentage')
            new_percentage_str = str(percentage)

            if current_percentage_str != new_percentage_str:
                tag.set('percentage', new_percentage_str)
                changes_made = True
    else:
        # Use existing percentages as weights
        # Sort tags by their current percentage (descending) then by key for determinism
        # This ensures remainders are distributed to higher-weighted units first
        sorted_tags_with_weights = sorted(
            [(tag, int(tag.get('percentage', '0'))) for tag in levy_tags],
            key=lambda x: (x[1], x[0].get('key', '')), reverse=True
        )

        # Create mutable list for percentage calculations
        mutable_percentages = []
        current_sum = 0
        for tag, weight in sorted_tags_with_weights:
            if total_weight > 0:
                calculated_percentage = round((weight / total_weight) * 100)
            else:
                calculated_percentage = 0 # Should not happen if total_weight > 0
            mutable_percentages.append([tag, calculated_percentage]) # Use list for mutability
            current_sum += calculated_percentage

        # Ensure minimum 1% for all units
        for item in mutable_percentages:
            if item[1] == 0:
                item[1] = 1
                current_sum += 1

        # Distribute any remainder to ensure sum is exactly 100
        remainder = 100 - current_sum
        if remainder != 0:
            # Sort by percentage (and then by key for determinism) for remainder distribution
            # For positive remainder: give to highest percentages first
            # For negative remainder: take from highest percentages first (but not below 1%)
            mutable_percentages.sort(key=lambda x: (x[1], x[0].get('key', '')), reverse=(remainder > 0))

            for i in range(min(abs(remainder), len(mutable_percentages))):
                if remainder > 0:
                    mutable_percentages[i][1] += 1
                else: # remainder < 0
                    # Ensure we don't reduce below 1%
                    if mutable_percentages[i][1] > 1:
                        mutable_percentages[i][1] -= 1
                    else:
                        # If we can't reduce this unit, stop reducing
                        break

        # Final adjustment to ensure we sum to exactly 100
        final_sum = sum(item[1] for item in mutable_percentages)
        difference = 100 - final_sum

        if difference != 0:
            # Sort again for final adjustment
            mutable_percentages.sort(key=lambda x: (x[1], x[0].get('key', '')), reverse=(difference > 0))

            for item in mutable_percentages:
                if difference == 0:
                    break
                if difference > 0:
                    item[1] += 1
                    difference -= 1
                else: # difference < 0
                    if item[1] > 1:  # Don't go below 1%
                        item[1] -= 1
                        difference += 1

        # --- NEW: Validation and application ---
        final_sum = 0
        for item in mutable_percentages:
            tag, new_percent_val = item[0], item[1]
            final_sum += new_percent_val
            # Ensure no unit has 0% (should not happen with the above logic)
            if new_percent_val < 1:
                new_percent_val = 1
                print(f"  -> CRITICAL: Unit with 0% detected. Setting to minimum 1%.")
            current_percentage_str = tag.get('percentage')
            new_percentage_str = str(new_percent_val)
            if current_percentage_str != new_percentage_str:
                tag.set('percentage', new_percentage_str)
                changes_made = True

        if final_sum != 100 and levy_tags:
            # This block should ideally never be reached with the new logic.
            print(f"  -> CRITICAL VALIDATION ERROR: Levy/Garrison percentages for keys {[t.get('key', 'N/A') for t in levy_tags]} sum to {final_sum}, not 100. This indicates a bug in the normalization logic.")
            # To be safe, force a re-normalization with equal distribution
            num_tags = len(levy_tags)
            if num_tags > 0:
                base_percentage = 100 // num_tags
                remainder = 100 % num_tags
                # Sort by key to be deterministic
                sorted_tags_for_recovery = sorted(levy_tags, key=lambda x: x.get('key', ''))
                for i, tag in enumerate(sorted_tags_for_recovery):
                    percentage = base_percentage
                    if i < remainder:
                        percentage += 1
                    tag.set('percentage', str(percentage))
                print("  -> RECOVERY: Forced equal percentage distribution to resolve error.")
                changes_made = True

    return changes_made

def normalize_all_levy_percentages(root, all_faction_elements=None):
    """
    Normalizes levy and garrison percentages for all factions in the XML.
    Returns total number of changes made (count of factions where normalization occurred).
    """
    total_changes = 0

    if root is not None:
        factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')
    elif all_faction_elements is not None:
        factions_to_iterate = all_faction_elements
    else:
        return 0

    for faction in factions_to_iterate:
        faction_changed = False

        # Normalize Levies
        levy_tags = faction.findall('Levies')
        if _normalize_levy_percentages(levy_tags):
            faction_changed = True

        # Normalize Garrisons by level
        garrison_tags = faction.findall('Garrison')
        garrisons_by_level = defaultdict(list)
        for g_tag in garrison_tags:
            level = g_tag.get('level')
            if level:
                garrisons_by_level[level].append(g_tag)

        for level_garrisons in garrisons_by_level.values():
            if _normalize_levy_percentages(level_garrisons):
                faction_changed = True

        if faction_changed:
            total_changes += 1

    return total_changes

def find_best_unit_match(unit_name_to_find, unit_pool, threshold=0.8):
    """
    Finds the canonical unit key from a unit_pool that best matches a potentially misspelled
    unit_name_to_find.
    """
    best_match, ratio = shared_utils.find_best_fuzzy_match(unit_name_to_find, unit_pool, threshold)
    if best_match:
        print(f"        -> INFO: Fuzzy unit match found for '{unit_name_to_find}'. Using canonical unit '{best_match}' (ratio: {ratio:.2f}).")
        return best_match

    # To replicate old logging, we would need the ratio back from the shared function.
    # For now, this is cleaner.
    return None

def _get_candidate_pool_for_tag(tag_name, working_pool, general_units, categorized_units, unit_categories, unit_to_training_level=None, unit_to_class_map=None):
    """Gets a pool of candidate units for a given tag ('General' or 'Knights')."""
    candidate_pool = []
    if tag_name == 'General':
        # Primary pool for Generals are units flagged as general_unit
        candidate_pool = [u for u in general_units if u in working_pool]

        # NEW: Add units with 'com' (commander) class to the candidate pool
        if unit_to_class_map:
            com_class_units = [u for u in working_pool if unit_to_class_map.get(u) == 'com']
            if com_class_units:
                candidate_pool.extend(com_class_units)
                print(f"        -> INFO: Found {len(com_class_units)} units with 'com' (commander) class for General pool.")

        if candidate_pool:
            return sorted(list(set(candidate_pool)))

        # --- NEW FALLBACKS for General ---
        if unit_to_training_level:
            # Fallback 1: Elite Cavalry
            elite_cav_pool = [
                u for u in working_pool
                if unit_to_training_level.get(u) == 'elite' and unit_categories.get(u) == 'cavalry'
            ]
            if elite_cav_pool:
                print(f"        -> INFO: No specific general units found. Using elite cavalry as fallback.")
                return sorted(list(set(elite_cav_pool)))

            # Fallback 2: Any Elite unit
            any_elite_pool = [
                u for u in working_pool
                if unit_to_training_level.get(u) == 'elite'
            ]
            if any_elite_pool:
                print(f"        -> INFO: No elite cavalry found. Using any elite unit as fallback.")
                return sorted(list(set(any_elite_pool)))

        print(f"        -> INFO: No specific general or elite units found. Falling back to Knights logic for General.")
        # If no specific generals or elite fallbacks, fall through to use the Knights logic

    # Logic for Knights, also serves as fallback for General
    roles = mappings.TAG_ROLE_MAPPING.get('Knights', [])
    for role in roles:
        units_in_role = categorized_units.get(role, [])
        candidate_pool.extend([u for u in units_in_role if u in working_pool])

    # Convert to set and back to list to ensure uniqueness and allow modification
    candidate_pool = list(set(candidate_pool))

    # --- EXISTING FALLBACK LOGIC ---
    if not candidate_pool:
        print(f"        -> INFO: No ideal '{tag_name}' candidates found based on roles. Attempting further fallbacks.")

        # Fallback 1: Any Cavalry
        any_cav_pool = [u for u in working_pool if unit_categories.get(u) == 'cavalry']
        if any_cav_pool:
            print(f"        -> Fallback 1: Using any cavalry unit ({len(any_cav_pool)} candidates).")
            return list(set(any_cav_pool))

        # Fallback 2: Any Melee Infantry
        any_inf_melee_pool = [u for u in working_pool if unit_categories.get(u) == 'inf_melee']
        if any_inf_melee_pool:
            print(f"        -> Fallback 2: Using any melee infantry unit ({len(any_inf_melee_pool)} candidates).")
            return list(set(any_inf_melee_pool))

        # Fallback 3: Any Unit from the working pool
        if working_pool:
            print(f"        -> Fallback 3: Using any unit from the faction's working pool ({len(working_pool)} candidates).")
            return list(set(working_pool))
        else:
            print(f"        -> WARNING: No units available in the working pool for '{tag_name}'.")
            return [] # Return empty list if working_pool is also empty

    return candidate_pool

def _apply_json_unit_override(faction_element, tag_name, json_data, all_units, working_pool, excluded_units_set=None):
    """
    Adds a JSON override for a unit type (General, Knights).
    Handles both string and list-of-ranked-objects formats.
    Returns a tuple: (override_applied, units_added_count).
    """
    # Handle string format
    if isinstance(json_data, str):
        json_unit_name = json_data
        print(f"    -> JSON Override: Proposed {tag_name.lower()} unit from JSON: '{json_unit_name}'.")
        canonical_unit_key = find_best_unit_match(json_unit_name, all_units)

        if not canonical_unit_key:
            print(f"    -> WARNING: JSON override {tag_name.lower()} unit name '{json_unit_name}' is invalid or no good fuzzy match found in all_units. This override will be ignored.")
            return False, 0
        # --- NEW: Check against global exclusion list ---
        elif excluded_units_set and canonical_unit_key in excluded_units_set:
            print(f"    -> WARNING: JSON override unit '{canonical_unit_key}' is in the global exclusion list. This override will be ignored.")
            return False, 0
        # --- END NEW ---
        else:
            # An explicit JSON override bypasses the working_pool.
            for element in list(faction_element.findall(tag_name)): # Remove existing
                faction_element.remove(element)

            # Create three elements, one for each rank
            units_added = 0
            for rank in ['1', '2', '3']:
                ET.SubElement(faction_element, tag_name, {'key': canonical_unit_key, 'rank': rank}) # Add new with rank
                units_added += 1
            print(f"    -> Applied JSON override: Set <{tag_name}> key to '{canonical_unit_key}' for all ranks.")
            return True, units_added


    # Handle list of ranked objects format
    elif isinstance(json_data, list):
        print(f"    -> JSON Override: Found ranked {tag_name.lower()} units for faction.")
        units_added = 0
        json_used_units = set() # Track units used within this JSON override list

        # First, remove all existing tags of this type
        for element in list(faction_element.findall(tag_name)):
            faction_element.remove(element)

        for item in json_data:
            if not isinstance(item, dict):
                print(f"    -> WARNING: Invalid item in JSON {tag_name.lower()} list (expected a dictionary): {item}. Skipping.")
                continue

            # This now supports both [{"rank_1": "unit1"}, {"rank_2": "unit2"}] and [{"rank_1": "unit1", "rank_2": "unit2"}]
            for rank_key, unit_name in item.items():
                # Parse rank
                rank_match = re.match(r'rank_(\d+)', rank_key)
                if not rank_match:
                    print(f"    -> WARNING: Invalid rank key '{rank_key}' in JSON {tag_name.lower()} list. Skipping.")
                    continue
                rank = rank_match.group(1)

                # Validate unit
                canonical_unit_key = find_best_unit_match(unit_name, all_units)
                if not canonical_unit_key:
                    print(f"    -> WARNING: JSON override {tag_name.lower()} unit name '{unit_name}' (rank {rank}) is invalid or no good fuzzy match found. Skipping.")
                    continue
                # --- NEW: Check against global exclusion list ---
                if excluded_units_set and canonical_unit_key in excluded_units_set:
                    print(f"    -> WARNING: JSON override unit '{canonical_unit_key}' (rank {rank}) is in the global exclusion list. Skipping.")
                    continue
                # --- NEW: Check for uniqueness within JSON override ---
                if canonical_unit_key in json_used_units:
                    print(f"    -> WARNING: JSON override unit '{canonical_unit_key}' (rank {rank}) is a duplicate within the JSON override list. Skipping to maintain diversity.")
                    continue
                # --- END NEW ---
                # An explicit JSON override bypasses the working_pool.
                # Add new element
                ET.SubElement(faction_element, tag_name, {'key': canonical_unit_key, 'rank': rank})
                json_used_units.add(canonical_unit_key) # Add to used units for this JSON override
                units_added += 1
                print(f"    -> Applied JSON override: Set <{tag_name}> (rank {rank}) key to '{canonical_unit_key}'.")

        if units_added > 0:
            return True, units_added
        else:
            # If the list was empty or all items were invalid, we should fall back.
            print(f"    -> WARNING: JSON override for {tag_name.lower()} was a list, but resulted in no valid units. Falling back to procedural logic.")
            return False, 0

    # If json_data is None or any other type, there's no override to apply.
    if json_data is not None:
        print(f"    -> WARNING: JSON override data for {tag_name.lower()} is of an unexpected type ({type(json_data)}). Ignoring override.")

    return False, 0

def populate_ranked_units(faction_element, tag_name, candidate_pool, unit_stats_map, faction_name, log_faction_str, tier=None, unit_to_tier_map=None, json_excluded_units=None):
    """
    Removes existing tags and populates new ones with ranks based on unit quality.
    Ensures each rank gets a unique unit if possible.
    Returns (units_added_count, failures_list).
    """
    failures = []
    units_added = 0
    used_units = set() # NEW: Keep track of units already assigned to a rank

    # --- NEW: Safeguard filter ---
    if json_excluded_units:
        original_pool_size = len(candidate_pool)
        candidate_pool = [u for u in candidate_pool if u not in json_excluded_units]
        if len(candidate_pool) < original_pool_size:
            print(f"      -> INFO: Safeguard filter removed {original_pool_size - len(candidate_pool)} excluded units from <{tag_name}> candidate pool.")
    # --- END NEW ---

    # 1. Remove existing tags of this type
    for element in list(faction_element.findall(tag_name)):
        faction_element.remove(element)

    if not candidate_pool:
        # If no candidate units are found, generate a failure for each rank
        for rank in [1, 2, 3]:
            description = f"A rank {rank} {tag_name.lower()} unit"
            if tag_name == 'General':
                description = f"A rank {rank} general's bodyguard unit"
            elif tag_name == 'Knights':
                description = f"A rank {rank} heavy cavalry unit"

            failures.append({
                'faction_element': faction_element,
                'tag_name': tag_name,
                'rank': str(rank), # Explicitly add rank
                'unit_role_description': description,
                'tier': tier,
                'unit_stats_map': unit_stats_map, # Keep for context
                'excluded_by_json': json_excluded_units,
            })
            print(f"      -> No candidate units found for <{tag_name}> rank {rank} for faction {log_faction_str}. Queued for LLM.")
        return 0, failures

    # Apply tier filtering if a specific tier is requested
    filtered_candidate_pool = []
    if tier is not None and unit_to_tier_map:
        for unit_key in candidate_pool:
            unit_tier = unit_to_tier_map.get(unit_key)
            if unit_tier == tier:
                filtered_candidate_pool.append(unit_key)

        if filtered_candidate_pool:
            print(f"      -> INFO: Filtered <{tag_name}> candidates for tier '{tier}'. Using {len(filtered_candidate_pool)} units.")
            candidate_pool = filtered_candidate_pool
        else:
            print(f"      -> WARNING: No <{tag_name}> units found for tier '{tier}' in faction {log_faction_str}. Falling back to all candidates.")

    # 2. Score and sort candidates
    candidates_with_scores = []
    for unit_key in candidate_pool:
        stats = unit_stats_map.get(unit_key)
        if stats:
            score = unit_selector._calculate_quality_score(stats)
            candidates_with_scores.append((score, unit_key))

    if not candidates_with_scores:
        # If no units have stats, we can't rank them. Generate a failure for each rank.
        for rank in [1, 2, 3]:
            description = f"A rank {rank} {tag_name.lower()} unit"
            if tag_name == 'General':
                description = f"A rank {rank} general's bodyguard unit"
            elif tag_name == 'Knights':
                description = f"A rank {rank} heavy cavalry unit"

            failures.append({
                'faction_element': faction_element, # Reference the faction element
                'tag_name': tag_name,
                'rank': str(rank), # Explicitly add rank
                'unit_role_description': description,
                'tier': tier,
                'unit_stats_map': unit_stats_map,
                'excluded_by_json': json_excluded_units,
            })
            print(f"      -> WARNING: No units in the candidate pool for <{tag_name}> had stats for quality ranking for faction {log_faction_str}. Queued for LLM.")
        return 0, failures

    candidates_with_scores.sort(key=lambda x: x[0])
    num_candidates = len(candidates_with_scores)

    # 3. Divide into three quality tiers
    tier_pools = {1: [], 2: [], 3: []}
    if num_candidates == 1:
        tier_pools[1] = candidates_with_scores
        tier_pools[2] = candidates_with_scores
        tier_pools[3] = candidates_with_scores
    elif num_candidates == 2:
        tier_pools[1] = [candidates_with_scores[0]]
        tier_pools[2] = [candidates_with_scores[1]]
        tier_pools[3] = [candidates_with_scores[1]]
    else: # 3 or more
        tier1_end_idx = max(1, num_candidates // 3)
        tier2_end_idx = max(tier1_end_idx + 1, (num_candidates * 2) // 3)
        tier_pools[1] = candidates_with_scores[0:tier1_end_idx]
        tier_pools[2] = candidates_with_scores[tier1_end_idx:tier2_end_idx]
        tier_pools[3] = candidates_with_scores[tier2_end_idx:]

    # 4. Assign units to ranks
    for rank in [1, 2, 3]:
        pool_for_rank = tier_pools[rank]

        # Filter out already used units
        available_for_rank = [item for item in pool_for_rank if item[1] not in used_units]

        if not available_for_rank:
            # Fallback: if specific tier pool is exhausted, try other tiers for unique units
            # Prioritize higher quality for higher ranks, lower for lower ranks
            if rank == 1: # Try to find a unique unit from any tier, prioritizing lower quality
                available_for_rank = [item for item in candidates_with_scores if item[1] not in used_units]
                available_for_rank.sort(key=lambda x: x[0]) # Sort by score ascending
            elif rank == 2: # Try to find a unique unit from any tier, middle quality
                available_for_rank = [item for item in candidates_with_scores if item[1] not in used_units]
                # No specific sort, just pick from remaining
            elif rank == 3: # Try to find a unique unit from any tier, prioritizing higher quality
                available_for_rank = [item for item in candidates_with_scores if item[1] not in used_units]
                available_for_rank.sort(key=lambda x: x[0], reverse=True) # Sort by score descending

            if available_for_rank:
                print(f"      -> INFO: Tier pool for rank {rank} exhausted of unique units. Falling back to general unique pool.")

        if available_for_rank:
            # Randomly select from the chosen tier to provide variety
            score, unit_key = random.choice(available_for_rank)
            ET.SubElement(faction_element, tag_name, {'key': unit_key, 'rank': str(rank)})
            units_added += 1
            used_units.add(unit_key) # Add to used units
            print(f"      -> Procedural: Set <{tag_name}> (rank {rank}) key to '{unit_key}' (Score: {score}).")
        else:
            # This case should be covered by the initial check for `if not candidate_pool`, but as a safeguard:
            description = f"A rank {rank} {tag_name.lower()} unit"
            failures.append({
                'faction_element': faction_element,
                'tag_name': tag_name,
                'rank': str(rank),
                'unit_role_description': description,
                'tier': tier,
                'unit_stats_map': unit_stats_map,
                'excluded_by_json': json_excluded_units,
            })
            print(f"      -> No unique candidate units found for <{tag_name}> rank {rank} for faction {log_faction_str}. Queued for LLM.")

    return units_added, failures

def manage_faction_levies(faction, working_pool, unit_to_training_level, unit_categories, log_faction_str, unit_to_class_map, json_group_data, all_units, tier, faction_elite_units=None, faction_name=None, excluded_units_set=None, destructive_on_failure=True, levy_composition_override=None, faction_culture_map=None):
    """
    Manages the levy units for a single faction.
    - Applies JSON overrides or a provided levy_composition_override if present.
    - If no composition is provided, generates a special 'LevyComposition' failure for LLM.
    - Finds suitable low-quality units if needed.
    - Normalizes percentages.
    - Returns a list of failures for LLM processing if units cannot be found.
    """
    changes = False
    failures = []

    current_excluded_units = set(excluded_units_set) if excluded_units_set else set()

    # Define training level mapping for numerical comparison
    TRAINING_LEVEL_MAP = {
        "mob": 0,
        "rabble": 1,
        "poorly_trained": 2,
        "trained": 3,
        "well_trained": 4,
        "elite": 5
    }

    # 1. Determine the levy composition to use
    levy_composition = None
    if levy_composition_override:
        levy_composition = levy_composition_override
        print(f"  -> Using provided levy_composition_override for {log_faction_str}: {levy_composition}")
    elif json_group_data and "levy_composition" in json_group_data:
        levy_composition = json_group_data["levy_composition"]
        print(f"  -> Using JSON-defined levy_composition for {log_faction_str}: {levy_composition}")

    if not levy_composition:
        # If no composition is found, this is now an LLM request.

        # NEW: If destructive, remove old tags before queuing for LLM
        if destructive_on_failure:
            for levy_tag in list(faction.findall('Levies')):
                faction.remove(levy_tag)
                changes = True
            print(f"  -> Removed existing levy tags for {log_faction_str} before queuing for LLM composition generation.")

        # First, determine available levy categories based on the working pool.
        available_levy_categories = set()
        for unit_key in working_pool:
            unit_class = unit_to_class_map.get(unit_key)
            for category, classes in mappings.LEVY_CATEGORY_TO_CLASSES.items():
                if unit_class in classes:
                    available_levy_categories.add(category)

        if not available_levy_categories:
            print(f"    -> WARNING: No suitable levy units found in working pool for {log_faction_str} to determine available categories. Using default categories for LLM request.")
            available_levy_categories = set(mappings.LEVY_CATEGORY_TO_CLASSES.keys())

        failures.append({
            'faction_element': faction,
            'tag_name': 'LevyComposition', # Special tag for LLM to generate composition
            'unit_role_description': f"Levy composition for faction '{faction_name}'.",
            'tier': tier,
            'available_levy_categories': sorted(list(available_levy_categories)),
            'excluded_units_set': list(excluded_units_set) if excluded_units_set else []
        })
        print(f"  -> No levy composition found for {log_faction_str}. Queued for LLM to generate composition.")
        return changes, failures

    # 2. If a composition was found (either override or JSON), proceed with unit selection
    # --- NEW: Identify and remove existing levies with excluded units ---
    invalid_levies_removed = False
    for levy_tag in list(faction.findall('Levies')): # Iterate over a copy
        key = levy_tag.get('key')
        if key and current_excluded_units and key in current_excluded_units:
            faction.remove(levy_tag)
            changes = True
            invalid_levies_removed = True
    if invalid_levies_removed:
        print(f"    -> Removed existing levy units that are now excluded by JSON rules for {log_faction_str}.")
    # --- END NEW ---

    # --- NEW: Exclude units already used as MenAtArms in this faction ---
    maa_keys_in_faction = {el.get('key') for el in faction.findall('MenAtArm') if el.get('key')}
    if maa_keys_in_faction:
        original_pool_size = len(working_pool)
        working_pool = set(working_pool) - maa_keys_in_faction # FIX: Ensure working_pool is a set
        if len(working_pool) < original_pool_size:
            print(f"    -> Excluded {original_pool_size - len(working_pool)} units already used as MenAtArms from levy consideration.")
    # --- END NEW ---

    # --- NEW: Apply current_excluded_units filter ---
    if current_excluded_units:
        original_pool_size = len(working_pool)
        working_pool = {u for u in working_pool if u not in current_excluded_units}
        if len(working_pool) < original_pool_size:
            print(f"    -> Excluded {original_pool_size - len(working_pool)} units from working pool based on core/unit exclusions.")
    # --- END NEW ---

    # NEW: Determine the pool for selection. If working_pool is empty, we cannot proceed.
    pool_for_selection = working_pool
    if not pool_for_selection:
        print(f"    -> WARNING: Working pool for {log_faction_str} is empty. Cannot select any levy units.")
        return changes, failures

    # Categorize the available pool_for_selection into levy categories
    categorized_levy_pool = defaultdict(list)
    for unit_key in pool_for_selection:
        unit_class = unit_to_class_map.get(unit_key)
        for category, classes in mappings.LEVY_CATEGORY_TO_CLASSES.items():
            if unit_class in classes:
                categorized_levy_pool[category].append(unit_key)

    # Filter out elite units from the levy pool
    if faction_elite_units:
        for category in categorized_levy_pool:
            categorized_levy_pool[category] = [u for u in categorized_levy_pool[category] if u not in faction_elite_units]

    selected_units_for_composition = {}
    used_units = set()

    # Sort composition by percentage descending to prioritize higher percentages
    sorted_composition = sorted(levy_composition.items(), key=lambda item: item[1], reverse=True)

    for category, percentage in sorted_composition:
        if category not in categorized_levy_pool or not categorized_levy_pool[category]:
            print(f"    -> WARNING: No units available for levy category '{category}' in {log_faction_str}. Skipping.")
            continue

        available_for_category = [u for u in categorized_levy_pool[category] if u not in used_units]
        if available_for_category:
            # Prioritize units with the lowest training level for levies
            training_level_candidates = []
            for unit_key in available_for_category:
                # Default to a high training level for units without defined level
                training_level_str = unit_to_training_level.get(unit_key, "elite")  # Default to highest if missing
                training_level_num = TRAINING_LEVEL_MAP.get(training_level_str, 5)  # Default to 5 (elite) if unknown

                # Only include units with training level <= 3 (mob, rabble, poorly_trained, trained)
                if training_level_num <= 3:
                    training_level_candidates.append((training_level_num, unit_key))

            # Sort by training level (ascending) to get lowest first
            training_level_candidates.sort(key=lambda x: x[0])

            if training_level_candidates:
                # Get all candidates with the lowest training level
                lowest_level = training_level_candidates[0][0]
                best_candidates = [unit_key for level, unit_key in training_level_candidates if level == lowest_level]

                # Randomly choose one of the best candidates
                chosen_unit = random.choice(best_candidates)
                selected_units_for_composition[chosen_unit] = percentage
                used_units.add(chosen_unit)
                print(f"    -> Selected '{chosen_unit}' (training level: {lowest_level}) for category '{category}' with {percentage}% weight.")
            else:
                print(f"    -> WARNING: No suitable units found after training level sorting for category '{category}' in {log_faction_str}. Skipping.")
        else:
            print(f"    -> WARNING: No unique units available for levy category '{category}' in {log_faction_str}. Skipping.")

    # Ensure a minimum number of levy slots (e.g., 3-5) are filled.
    # If we have fewer than 3 unique units selected, try to fill more from the remaining pool.
    min_levy_slots = random.randint(3, 5)
    while len(selected_units_for_composition) < min_levy_slots and len(used_units) < len(pool_for_selection):
        remaining_pool = [u for u in pool_for_selection if u not in used_units and (not faction_elite_units or u not in faction_elite_units)]
        if remaining_pool:
            chosen_unit = random.choice(remaining_pool)
            selected_units_for_composition[chosen_unit] = 0 # Assign 0% initially, will be normalized
            used_units.add(chosen_unit)
            print(f"    -> Added additional unit '{chosen_unit}' to fill minimum levy slots.")
        else:
            break # No more units to add

    if not selected_units_for_composition:
        print(f"    -> ERROR: Could not select any levy units for {log_faction_str} even with composition. No changes will be made to existing levy tags.")
        if destructive_on_failure:
            print(f"    -> Destructive mode enabled: Removing all existing levy tags for {log_faction_str} due to selection failure.")
            for levy_tag in list(faction.findall('Levies')):
                faction.remove(levy_tag)
                changes = True
        return changes, failures

    # If we get here, we have a new set of units to apply.
    # NOW we can remove the old tags.
    for levy_tag in list(faction.findall('Levies')):
        faction.remove(levy_tag)
        changes = True

    # Create XML elements based on selected units and their percentages
    for unit_key, percentage in selected_units_for_composition.items():
        ET.SubElement(faction, 'Levies', {'key': unit_key, 'percentage': str(percentage), 'max': 'LEVY'})
        changes = True

    # Normalize percentages
    final_levies = faction.findall('Levies')
    if final_levies:
        if _normalize_levy_percentages(final_levies):
            changes = True

    return changes, failures

def manage_faction_garrisons(faction, working_pool, unit_to_training_level, unit_categories, log_faction_str, general_units, unit_to_class_map, tier, unit_to_tier_map, excluded_units_set=None, destructive_on_failure=True, faction_to_heritage_map=None, heritage_to_factions_map=None, screen_name_to_faction_key_map=None, faction_key_to_units_map=None, json_group_data=None, all_units=None):
    """
    Manages the garrison units for a single faction.
    - Ensures each fortification level (1-4) has an appropriate number of units.
    - Finds suitable low-to-mid quality units.
    - Returns a list of failures for LLM processing if units cannot be found.
    """
    changes = False
    failures = []
    json_override_applied = False

    # NEW: Keep track of all units used for garrisons within this faction
    all_used_garrison_keys = {g.get('key') for g in faction.findall('Garrison') if g.get('key')}

    # 1. Handle JSON Overrides
    json_override_data = json_group_data.get("garrisons") if json_group_data else None
    if json_override_data:
        if isinstance(json_override_data, dict):
            print(f"  -> Applying JSON override for garrisons in {log_faction_str}.")
            # Remove all existing garrisons
            for garrison_tag in list(faction.findall('Garrison')):
                faction.remove(garrison_tag)
                changes = True
            all_used_garrison_keys.clear() # Clear as we are regenerating

            for level, unit_list in json_override_data.items():
                if str(level) not in ['1', '2', '3', '4']:
                    print(f"    -> WARNING: JSON garrison override has invalid level '{level}'. Skipping.")
                    continue
                if not isinstance(unit_list, list):
                    print(f"    -> WARNING: JSON garrison override for level '{level}' is not a list. Skipping.")
                    continue

                for unit_name in unit_list:
                    best_match, _ = find_best_unit_match(unit_name, all_units)
                    if best_match:
                        if excluded_units_set and best_match in excluded_units_set:
                            print(f"    -> WARNING: JSON override garrison unit '{best_match}' is excluded. Skipping.")
                            continue
                        if best_match in all_used_garrison_keys: # NEW: Check for uniqueness even in JSON overrides
                            print(f"    -> WARNING: JSON override garrison unit '{best_match}' for level '{level}' is already used in this faction. Skipping to maintain diversity.")
                            continue
                        ET.SubElement(faction, 'Garrison', {'key': best_match, 'percentage': '0', 'max': 'LEVY', 'level': str(level)})
                        all_used_garrison_keys.add(best_match) # NEW: Add to used keys
                        changes = True
                    else:
                        print(f"    -> WARNING: JSON override garrison unit '{unit_name}' not found. Skipping.")
            json_override_applied = True
        else:
            print(f"  -> WARNING: JSON 'garrisons' override for {log_faction_str} is not a dictionary. Ignoring.")

    # 2. Procedural Logic
    if not json_override_applied:
        # --- NEW: Identify and remove existing garrisons with excluded units ---
        invalid_garrisons_removed = False
        for g in list(faction.findall('Garrison')): # Iterate over a copy
            key = g.get('key')
            if key and excluded_units_set and key in excluded_units_set:
                faction.remove(g)
                changes = True
                invalid_garrisons_removed = True
        if invalid_garrisons_removed:
            print(f"    -> Removed existing garrison units that are now excluded by JSON rules for {log_faction_str}.")
        # --- END NEW ---

        # --- NEW: Remove existing garrisons with missing 'key' attributes ---
        keyless_garrisons_removed = False
        for g in list(faction.findall('Garrison')):
            if not g.get('key'):
                faction.remove(g)
                changes = True
                keyless_garrisons_removed = True
        if keyless_garrisons_removed:
            print(f"    -> Removed existing garrison units with missing 'key' attributes for {log_faction_str}.")
        # --- END NEW ---

        # Rebuild `all_used_garrison_keys` after removals for safety
        if invalid_garrisons_removed or keyless_garrisons_removed:
            all_used_garrison_keys = {g.get('key') for g in faction.findall('Garrison') if g.get('key')}


        garrisons_by_level = defaultdict(list)
        for g in faction.findall('Garrison'):
            level = g.get('level')
            key = g.get('key')
            if level and key:
                garrisons_by_level[level].append(g)

        target_units_per_level = {'1': 3, '2': 3, '3': 3, '4': 3}

        # Candidate pool for garrisons: low-to-mid quality infantry/spearmen, not generals.
        garrison_candidate_pool = []
        if working_pool:
            garrison_candidate_pool = [
                unit for unit in working_pool
                if (not general_units or unit not in general_units) and ('inf' in (unit_categories.get(unit) or ''))
            ]

        if not garrison_candidate_pool and working_pool:
            # Fallback to any non-general unit
            print(f"    -> No infantry candidates for garrisons in {log_faction_str}. Using any non-general unit.")
            garrison_candidate_pool = [u for u in working_pool if not general_units or u not in general_units]

        for level, target_count in target_units_per_level.items():
            existing_units = garrisons_by_level[level]
            num_to_add = target_count - len(existing_units)

            if num_to_add <= 0:
                continue  # Enough units for this level

            if destructive_on_failure:
                print(f"  -> Faction {log_faction_str} has insufficient garrisons for level {level}. Removing existing and regenerating.")
                for tag in existing_units:
                    faction.remove(tag)
                    if tag.get('key'): # Remove from used keys if it had one
                        all_used_garrison_keys.discard(tag.get('key'))
                num_to_add = target_count
                changes = True
                existing_units = [] # Reset after removal

            # NEW: Filter candidate pool for uniqueness
            available_candidates_for_level = [u for u in garrison_candidate_pool if u not in all_used_garrison_keys]

            if len(available_candidates_for_level) >= num_to_add:
                # Prioritize low-to-mid quality units for garrisons
                training_level_candidates = []
                for unit_key in available_candidates_for_level:
                    # Default to a high training level for units without defined level
                    training_level = unit_to_training_level.get(unit_key, 99)
                    training_level_candidates.append((training_level, unit_key))

                # Sort by training level (ascending) to get lowest first
                training_level_candidates.sort(key=lambda x: x[0])

                # Take a slice representing low-to-mid quality units
                # Use at least num_to_add units, but no more than half of available candidates
                slice_size = max(num_to_add, len(training_level_candidates) // 2)
                low_mid_quality_pool = [unit_key for _, unit_key in training_level_candidates[:slice_size]]

                # If we don't have enough in our quality pool, use all available
                if len(low_mid_quality_pool) < num_to_add:
                    low_mid_quality_pool = available_candidates_for_level

                # Select `num_to_add` unique units from the quality pool
                chosen_units = random.sample(low_mid_quality_pool, num_to_add)
                for chosen_unit in chosen_units:
                    ET.SubElement(faction, 'Garrison', {'key': chosen_unit, 'percentage': '0', 'max': 'LEVY', 'level': level})
                    all_used_garrison_keys.add(chosen_unit) # Add to used keys
                    changes = True
            else:
                # Not enough unique candidates. Fill what we can and queue failures for the rest.
                if available_candidates_for_level:
                    chosen_units = random.sample(available_candidates_for_level, len(available_candidates_for_level))
                    for chosen_unit in chosen_units:
                        ET.SubElement(faction, 'Garrison', {'key': chosen_unit, 'percentage': '0', 'max': 'LEVY', 'level': level})
                        all_used_garrison_keys.add(chosen_unit) # Add to used keys
                        changes = True
                    print(f"    -> Added {len(chosen_units)} unique garrison units for {log_faction_str} level {level}. Still need {num_to_add - len(chosen_units)} more.")

                # Queue failures for the remaining slots
                for i in range(num_to_add - len(available_candidates_for_level)):
                    failures.append({
                        'faction_element': faction,
                        'tag_name': 'Garrison',
                        'level': level,
                        'unit_role_description': f"A garrison unit for a level {level} fortification (slot {len(existing_units) + len(available_candidates_for_level) + i + 1}).",
                        'tier': tier,
                        'unit_categories': unit_categories,
                        'unit_to_class_map': unit_to_class_map,
                        'garrison_slot': len(existing_units) + len(available_candidates_for_level) + i + 1
                    })
                    print(f"    -> No suitable unique garrison candidates for {log_faction_str} level {level}. Queued for LLM.")

    # 3. Normalize percentages for each level
    final_garrisons_by_level = defaultdict(list)
    for g_tag in faction.findall('Garrison'):
        level = g_tag.get('level')
        if level:
            final_garrisons_by_level[level].append(g_tag)

    for level, tags in final_garrisons_by_level.items():
        if _normalize_levy_percentages(tags): # Re-using the levy normalization logic
            changes = True

    return changes, failures

def _manage_single_faction_generals_and_knights(faction, faction_pool_cache, categorized_units, general_units, unit_stats_map, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map, tier, unit_to_tier_map,
                                    faction_to_json_map, all_units, unit_to_training_level, global_excluded_units_set, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, faction_culture_map, is_submod_mode, factions_in_main_mod, unit_to_class_map=None):
    """
    Manages General and Knights units for a single faction.
    Returns (units_added_count, list of failures).
    """
    from mapper_tools import faction_json_utils # Late import to avoid circular dependency

    units_added_count = 0
    failures = []

    faction_name = faction.get('name')
    # The Default faction and submod-managed factions are handled by the calling loop.

    print(f"  -> Processing General/Knights for faction: '{faction_name}'")

    # --- Faction-specific Excluded Units ---
    base_excluded_units = set(global_excluded_units_set) if global_excluded_units_set else set()
    json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None
    json_exclusions = faction_json_utils.get_json_general_exclusions_for_faction(
        faction_name, json_group_data, faction_culture_map, all_units, log_context="Generals/Knights"
    )
    base_excluded_units.update(json_exclusions)

    for tag_name in ['General', 'Knights']:
        print(f"    -> Managing <{tag_name}> units...")

        # --- NEW: Implement two-pool exclusions based on unit type ---
        current_pass_exclusions = set(base_excluded_units)  # Start with base exclusions

        # Get all currently used professional and conscript keys in the faction
        professional_keys = faction_xml_utils.get_professional_keys_in_faction(faction)
        conscript_keys = faction_xml_utils.get_conscript_keys_in_faction(faction)

        # Get keys of units for the tag currently being processed
        current_tag_keys = {el.get('key') for el in faction.findall(tag_name) if el.get('key')}

        # Combine all used keys and subtract current tag keys (allow re-selection of same units)
        all_used_keys = (professional_keys | conscript_keys) - current_tag_keys
        current_pass_exclusions.update(all_used_keys)
        print(f"      -> Excluding {len(all_used_keys)} units already used in faction for <{tag_name}> pass.")

        # Get working pool with current pass exclusions
        working_pool, log_faction_str, _ = faction_xml_utils.get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, current_pass_exclusions,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix=f"({tag_name})"
        )

        # 1. Handle JSON Overrides
        json_override_data = json_group_data.get(tag_name.lower()) if json_group_data else None
        override_applied, units_added = _apply_json_unit_override(faction, tag_name, json_override_data, all_units, working_pool, excluded_units_set=current_pass_exclusions)
        if override_applied:
            units_added_count += units_added
            continue # Move to next tag if override was successful

        # 2. Procedural Generation if no override
        existing_tags = faction.findall(tag_name)
        if existing_tags and all(el.get('key') for el in existing_tags) and len(existing_tags) >= 3:
            print(f"      -> INFO: Faction already has {len(existing_tags)} valid <{tag_name}> tags. Skipping procedural generation.")
            continue

        candidate_pool = _get_candidate_pool_for_tag(tag_name, working_pool, general_units, categorized_units, unit_categories, unit_to_training_level, unit_to_class_map=unit_to_class_map)

        # Pass current pass exclusions to populate_ranked_units
        units_added, single_tag_failures = populate_ranked_units(faction, tag_name, candidate_pool, unit_stats_map, faction_name, log_faction_str, tier=tier, unit_to_tier_map=unit_to_tier_map, json_excluded_units=current_pass_exclusions)
        units_added_count += units_added
        failures.extend(single_tag_failures)

    return units_added_count, failures

def manage_all_generals_and_knights(root, categorized_units, general_units, unit_stats_map, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool, faction_to_subculture_map=None, subculture_to_factions_map=None, faction_key_to_screen_name_map=None, culture_to_faction_map=None, tier=None, unit_to_tier_map=None,
                                    faction_to_json_map=None, all_units=None, unit_to_training_level=None, excluded_units_set=None, faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, faction_culture_map=None, is_submod_mode=False, factions_in_main_mod=None, all_faction_elements=None, unit_to_class_map=None):
    """
    Iterates through all factions and ensures they have ranked General and Knights units.
    Handles JSON overrides and procedural generation.
    Returns (units_added_count, list of failures).
    """
    print("\nManaging General and Knights units for all factions...")
    if is_submod_mode:
        print("  -> Submod mode: Will only process factions not present in the main mod.")

    total_units_added = 0
    all_failures = []
    faction_pool_cache = {} # Local cache for this function run

    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    for faction in factions_to_iterate:
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        # In submod mode, skip factions that are already in the main mod.
        if is_submod_mode and factions_in_main_mod and faction_name in factions_in_main_mod:
            continue

        units_added, failures = _manage_single_faction_generals_and_knights(
            faction, faction_pool_cache, categorized_units, general_units, unit_stats_map, unit_categories,
            screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map,
            subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map, tier,
            unit_to_tier_map, faction_to_json_map, all_units, unit_to_training_level, excluded_units_set,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, faction_culture_map,
            is_submod_mode, factions_in_main_mod, unit_to_class_map=unit_to_class_map
        )
        total_units_added += units_added
        all_failures.extend(failures)

    return total_units_added, all_failures

def find_and_apply_high_confidence_replacement(element, log_faction_str, working_pool, tier, unit_variant_map, ck3_maa_definitions, unit_to_class_map, unit_to_description_map, threshold=0.90):
    """
    Attempts to find a high-confidence replacement for a MenAtArm unit based on Attila unit class.
    This is the primary, high-precision matching strategy.
    Returns True if a replacement was made, False otherwise.
    """
    tag_name = element.tag
    if tag_name != 'MenAtArm':
        return False # This function only handles MenAtArm

    maa_definition_name = element.get('type')
    internal_type = None
    if maa_definition_name and ck3_maa_definitions:
        internal_type = ck3_maa_definitions.get(maa_definition_name)

    unit_type_info = f" (internal type: {internal_type})" if internal_type else ""
    # print(f"    -> For MenAtArm definition '{maa_definition_name}'{unit_type_info}:") # Moved to calling function

    found_key = None
    # First, try with the specific MAA definition name
    if maa_definition_name:
        found_key = unit_selector._find_replacement_for_maa(maa_definition_name, working_pool, unit_to_class_map, tier, unit_variant_map, unit_to_description_map, threshold=threshold)
        if found_key:
            # print(f"    -> Strategy: Ideal Match (Class '{mappings.CK3_TYPE_TO_ATTILA_CLASS.get(maa_definition_name)}').")
            pass

    # If not found, fall back to the generic internal type
    if not found_key and internal_type:
        found_key = unit_selector._find_replacement_for_maa(internal_type, working_pool, unit_to_class_map, tier, unit_variant_map, unit_to_description_map, threshold=threshold)
        if found_key:
            # print(f"    -> Strategy: Ideal Match (Class '{mappings.CK3_TYPE_TO_ATTILA_CLASS.get(internal_type)}').")
            pass

    if found_key:
        element.set('key', found_key)
        return True
    return False

def find_and_apply_class_based_low_confidence_replacement(element, log_faction_str, working_pool, tier, unit_variant_map, ck3_maa_definitions, unit_to_class_map, unit_to_description_map):
    """
    Attempts to find a low-confidence replacement for a MenAtArm unit based on Attila unit class,
    but with a lenient fuzzy matching threshold. This is a post-LLM fallback.
    Returns True if a replacement was made, False otherwise.
    """
    tag_name = element.tag
    if tag_name != 'MenAtArm':
        return False

    maa_definition_name = element.get('type')
    internal_type = None
    if maa_definition_name and ck3_maa_definitions:
        internal_type = ck3_maa_definitions.get(maa_definition_name)

    found_key = None
    if internal_type:
        # Use a lenient threshold of 0.0 to get the best possible fuzzy match from the class pool
        found_key = unit_selector._find_replacement_for_maa(internal_type, working_pool, unit_to_class_map, tier, unit_variant_map, unit_to_description_map, threshold=0.0)

    if found_key:
        element.set('key', found_key)
        return True
    return False

def find_and_apply_best_effort_replacement(root, faction_name, element, log_faction_str, categorized_units, faction_unit_pool, unit_categories, tier, unit_variant_map, ck3_maa_definitions, unit_to_description_map, tag_name, unit_role_description, unit_to_class_map, rank=None, level=None, unit_stats_map=None, levy_slot=None, global_unit_pool=None, excluded_units_set=None):
    """
    Helper function to find a suitable replacement unit using broader search strategies
    (role-based, category-based, or any from faction pool) and apply it to the element.
    This is used as a fallback for MenAtArm or as the primary strategy for other unit types (e.g., General, Knights, Levies, Garrison).
    Returns True if a replacement was made, False otherwise.
    """
    # Find the faction element to get existing units
    faction_element = None
    if root is not None:
        for f in root.findall('Faction'):
            if f.get('name') == faction_name:
                faction_element = f
                break
    else:
        # If root is None, we can't find the faction element directly
        # This can happen during parallel processing
        faction_element = None

    # NEW: Get existing keys for the current tag_name within this faction
    existing_keys_for_tag = set()
    if faction_element:
        for el in faction_element.findall(tag_name):
            key = el.get('key')
            if key:
                existing_keys_for_tag.add(key)

    # Combine global exclusions with existing keys for this tag
    combined_exclusions_for_selection = set(excluded_units_set) if excluded_units_set else set()
    combined_exclusions_for_selection.update(existing_keys_for_tag)

    # --- Safeguard filter for the faction-specific pool ---
    if faction_unit_pool:
        original_pool_size = len(faction_unit_pool)
        # Ensure faction_unit_pool is a set for subtraction
        faction_unit_pool = set(faction_unit_pool) - combined_exclusions_for_selection # Use combined exclusions
        if len(faction_unit_pool) < original_pool_size:
            print(f"      -> INFO: Safeguard filter removed {original_pool_size - len(faction_unit_pool)} excluded units from best-effort candidate pool.")

    # --- Filter global_unit_pool once at the start if it exists ---
    filtered_global_pool = None
    if global_unit_pool:
        filtered_global_pool = global_unit_pool
        if combined_exclusions_for_selection: # Use combined exclusions
            original_global_size = len(filtered_global_pool)
            # Ensure it's a set for subtraction
            filtered_global_pool = set(filtered_global_pool) - combined_exclusions_for_selection # Use combined exclusions
            if len(filtered_global_pool) < original_global_size:
                 print(f"      -> INFO: Safeguard filter removed {original_global_size - len(filtered_global_pool)} excluded units from global fallback pool.")

    unit_type_info = ""
    found_key = None

    if tag_name == 'MenAtArm':
        maa_definition_name = element.get('type') if element else unit_role_description # Use unit_role_description if element is None (for missing MAA type)
        internal_type = None
        if maa_definition_name and ck3_maa_definitions:
            internal_type = ck3_maa_definitions.get(maa_definition_name)

        unit_type_info = f" (internal type: {internal_type})" if internal_type else ""
        print(f"    -> For MenAtArm definition '{maa_definition_name}'{unit_type_info}: (Best-effort fallback)")

        max_attr = element.get('max') if element else None
        # Prioritize specific MAA definition, then fallback to internal type
        target_category_key = max_attr.upper() if max_attr else (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) or mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type))
        target_roles = (mappings.CK3_TYPE_TO_ATTILA_ROLES.get(maa_definition_name) or mappings.CK3_TYPE_TO_ATTILA_ROLES.get(internal_type, mappings.CK3_TYPE_TO_ATTILA_ROLES['default']))

        # --- Search in faction_unit_pool first ---
        found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, target_category_key, tier, unit_variant_map, maa_type=internal_type, unit_to_description_map=unit_to_description_map)

        # --- If not found, search in filtered_global_pool for thematic units ---
        if not found_key and global_unit_pool:
            failure_data_for_global_search = {
                'maa_definition_name': maa_definition_name,
                'internal_type': internal_type
            }
            global_candidates = find_global_thematic_candidates(
                failure_data_for_global_search,
                global_unit_pool,
                unit_to_class_map,
                unit_to_description_map,
                ck3_maa_definitions
            )
            if global_candidates:
                print(f"      -> Global Thematic Fallback: Found {len(global_candidates)} candidates for '{maa_definition_name}'.")
                available_global_candidates = [u for u in global_candidates if u not in combined_exclusions_for_selection]
                if available_global_candidates:
                    found_key = random.choice(available_global_candidates)
                    print(f"        -> Selected global thematic unit '{found_key}'.")
                else:
                    print(f"        -> All global thematic candidates were already used or excluded.")

    elif tag_name == 'Levies':
        target_roles = mappings.TAG_ROLE_MAPPING["Levies"]
        # Try LEVY category first
        found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, 'LEVY', tier, unit_variant_map, unit_to_description_map=unit_to_description_map)
        # Fallback to INFANTRY category if no LEVY unit found
        if not found_key:
            print(f"      -> No suitable Levies unit in 'LEVY' category for {log_faction_str}. Falling back to 'INFANTRY'.")
            found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, 'INFANTRY', tier, unit_variant_map, unit_to_description_map=unit_to_description_map)

        # REMOVED: Global fallback for Levies.

    elif tag_name == 'Garrison':
        # Garrisons are general infantry
        target_roles = mappings.TAG_ROLE_MAPPING["Garrison"]

        # --- Search in faction_unit_pool first ---
        # Attempt 1: LEVY category
        found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, 'LEVY', tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)

        # Attempt 2: INFANTRY category
        if not found_key:
            print(f"      -> No suitable Garrison unit in 'LEVY' category for {log_faction_str}. Falling back to 'INFANTRY'.")
            found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, 'INFANTRY', tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)

        # Attempt 3: Any role-matching unit
        if not found_key:
            print(f"      -> No suitable Garrison unit in 'INFANTRY' category for {log_faction_str}. Falling back to any role-matching unit.")
            found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, None, tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)

        # Attempt 4: Ultimate Fallback (any INFANTRY, RANGED, or CAVALRY)
        if not found_key:
            print(f"      -> ULTIMATE FALLBACK (Faction Pool): No role-matching unit found. Searching for any INFANTRY, RANGED, or CAVALRY unit.")
            for category in ['INFANTRY', 'RANGED', 'CAVALRY']:
                found_key = unit_selector.find_unit_for_role(None, categorized_units, faction_unit_pool, unit_categories, category, tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)
                if found_key:
                    print(f"        -> Found unit in '{category}' category.")
                    break

        # REMOVED: Global fallback for Garrison.

    elif tag_name == 'General' or tag_name == 'Knights':
        # Generals and Knights are typically cavalry. First, try to find a suitable cavalry unit.
        target_category_key_cav = 'CAVALRY'
        target_roles_cav = mappings.TAG_ROLE_MAPPING["General_Knights"]
        found_key = unit_selector.find_unit_for_role(target_roles_cav, categorized_units, faction_unit_pool, unit_categories, target_category_key_cav, tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)

        # REMOVED: Global fallback for Generals/Knights cavalry.

        # If no cavalry is found, fall back to high-quality infantry.
        if not found_key:
            print(f"      -> No suitable cavalry found for <{tag_name}> in faction {log_faction_str}. Falling back to high-quality infantry.")
            target_category_key_inf = 'INFANTRY'
            # Use the new mapping for infantry fallback roles
            target_roles_inf = mappings.TAG_ROLE_MAPPING.get("General_Knights_Infantry_Fallback", mappings.CK3_TYPE_TO_ATTILA_ROLES['heavy_infantry'])
            found_key = unit_selector.find_unit_for_role(target_roles_inf, categorized_units, faction_unit_pool, unit_categories, target_category_key_inf, tier, unit_variant_map, unit_to_description_map=unit_to_description_map, rank=rank, unit_stats_map=unit_stats_map)
            # REMOVED: Global fallback for Generals/Knights infantry.
    else: # Default fallback for any other tag that might need a unit
        max_attr = element.get('max') if element else None
        target_category_key = max_attr.upper() if max_attr else None
        target_roles = mappings.TAG_ROLE_MAPPING.get(tag_name, mappings.TAG_ROLE_MAPPING["default"])
        found_key = unit_selector.find_unit_for_role(target_roles, categorized_units, faction_unit_pool, unit_categories, target_category_key, tier, unit_variant_map, unit_to_description_map=unit_to_description_map)
        # REMOVED: Global fallback for other tags.

    # --- NEW: Final fallback to a random levy unit from the same faction ---
    if not found_key and tag_name != 'MenAtArm':
        print(f"      -> ULTIMATE FALLBACK (Levy): All other strategies failed for <{tag_name}>. Searching for any levy unit from the same faction.")
        # Robustly find the faction element to avoid XPath injection issues with quotes
        # faction_element is already defined at the start of the function
        if faction_element is not None:
            faction_levy_units = [el.get('key') for el in faction_element.findall('Levies') if el.get('key')]
            if faction_levy_units:
                # --- NEW: Filter against excluded units ---
                if combined_exclusions_for_selection: # Use combined exclusions
                    faction_levy_units = [u for u in faction_levy_units if u not in combined_exclusions_for_selection]
                # --- END NEW ---
                if faction_levy_units: # Check if list is not empty after filtering
                    found_key = random.choice(sorted(faction_levy_units))
                    if found_key:
                        print(f"        -> Found and assigned random levy unit '{found_key}'.")

    # REMOVED: Absolute final fallback to a random unit from the global pool.
    # REMOVED: Absolute LAST RESORT fallback, ignoring faction-specific exclusions.

    if found_key:
        if element is None: # Create the element if it was missing
            new_element_attrs = {'key': found_key}
            if tag_name == 'MenAtArm':
                new_element_attrs['type'] = unit_role_description # unit_role_description holds the maa_type here
            if rank:
                new_element_attrs['rank'] = str(rank)
            if level:
                new_element_attrs['level'] = str(level)
            if tag_name == 'Levies' or tag_name == 'Garrison':
                new_element_attrs['percentage'] = '0' # Will be normalized later
                new_element_attrs['max'] = 'LEVY'
            # Find faction element to append to
            # faction_element is already defined at the start of the function
            if faction_element is not None:
                ET.SubElement(faction_element, tag_name, new_element_attrs)
                print(f"    -> Created missing <{tag_name}> tag and set key to '{found_key}'.")
            else:
                # If we can't find the faction element (e.g., during parallel processing with root=None)
                # and we need to create a new element, we can't do it here
                # In this case, we assume the element already exists and we're just setting its key
                if element is not None:
                    element.set('key', found_key)
                    print(f"    -> Set key to '{found_key}' on existing element.")
                else:
                    print(f"    -> ERROR: Could not find faction '{faction_name}' and no element provided to set key '{found_key}'.")
                    return False
        else:
            element.set('key', found_key)
        return True
    return False

def find_global_thematic_candidates(failure_data, all_units, unit_to_class_map, unit_to_description_map, ck3_maa_definitions):
    """
    Finds a pool of thematically appropriate units from the entire global unit pool.
    This is used as a final fallback for very specific MAA types (e.g., elephants, cataphracts).
    """
    maa_definition_name = failure_data['maa_definition_name']
    internal_type = failure_data.get('internal_type')

    # Only run this for specific, thematic MAA types, not generic ones.
    if internal_type in ['heavy_infantry', 'light_infantry', 'heavy_cavalry', 'light_cavalry', 'pikemen', 'archers', 'crossbowmen', 'skirmishers', 'siege_weapon']:
        return []

    print(f"    -> Global Thematic Search: Searching for best global match for '{maa_definition_name}'...")

    # Use a lenient threshold to find any thematically similar unit in the entire game
    # This is essentially a Levenshtein search on the MAA type against all unit keys and descriptions.
    candidates = []
    cleaned_maa_type = maa_definition_name.lower().replace("_", " ")
    for unit_key in all_units:
        cleaned_unit_key = unit_key.lower().replace("_", " ")
        ratio_key = Levenshtein.ratio(cleaned_maa_type, cleaned_unit_key)

        ratio_desc = 0
        description = unit_to_description_map.get(unit_key, "").lower()
        if description:
            ratio_desc = Levenshtein.ratio(cleaned_maa_type, description)

        # Use a low threshold to cast a wide net for potential thematic matches
        if max(ratio_key, ratio_desc) > 0.4:
            candidates.append(unit_key)

    return candidates
