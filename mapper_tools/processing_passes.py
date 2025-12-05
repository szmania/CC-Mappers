import random
from mapper_tools import faction_xml_utils
from mapper_tools import unit_management
from mapper_tools import faction_json_utils
from mapper_tools import llm_orchestrator
from mapper_tools import ck3_to_attila_mappings as mappings
from mapper_tools import unit_selector

def run_high_confidence_unit_pass(root, tier, unit_variant_map, ck3_maa_definitions, unit_to_class_map, unit_to_description_map,
                                   screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map,
                                   subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map,
                                   faction_culture_map, categorized_units, unit_categories, unit_stats_map, all_units,
                                   excluded_units_set=None, faction_pool_cache=None,
                                   faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, first_pass_threshold=0.90, llm_helper=None, faction_to_json_map=None):
    """
    Runs the first, high-confidence procedural pass to replace missing unit keys.
    This pass focuses on precise matches using Attila unit classes.
    It only processes MenAtArm tags. Other tags (General, Knights, etc.) are handled by dedicated management functions.
    """
    print("\nRunning high-confidence procedural pass for MenAtArm units...")
    replacements_made = 0
    failures = []
    cache_modified_in_pass = False


    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        print(f"  -> Processing faction: '{faction_name}'")

        # Process only MenAtArm tags in this pass
        for element in faction.findall('MenAtArm'):
            # We process all MenAtArm tags to ensure they have the best possible key, even if one is already present.
            maa_definition_name = element.get('type')
            internal_type = ck3_maa_definitions.get(maa_definition_name) if ck3_maa_definitions else None
            unit_type_info = f" (internal type: {internal_type})" if internal_type else ""
            print(f"    -> Processing <MenAtArm type='{maa_definition_name}'>{unit_type_info}...")

            # --- Get JSON exclusions for this specific MAA type ---
            json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None
            excluded_by_json = faction_json_utils.get_json_exclusions_for_maa(maa_definition_name, json_group_data, faction_culture_map, all_units=all_units)

            # --- Combine global and MAA-specific exclusions before getting the unit pool ---
            combined_exclusions = set(excluded_units_set) if excluded_units_set else set()
            combined_exclusions.update(excluded_by_json)

            # --- NEW: Exclude units already used as Levies in this faction ---
            levy_keys_in_faction = {el.get('key') for el in faction.findall('Levies') if el.get('key')}
            if levy_keys_in_faction:
                combined_exclusions.update(levy_keys_in_faction)
                print(f"    -> Excluding {len(levy_keys_in_faction)} units already used as Levies from MenAtArm consideration.")
            # --- END NEW ---

            # --- Get required classes for this specific MAA type ---
            required_classes = None
            if maa_definition_name:
                required_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(maa_definition_name)
            if not required_classes and internal_type:
                required_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(internal_type)

            # Get the working pool, correctly filtered with the combined exclusions
            working_pool, log_faction_str, tiered_pools = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, combined_exclusions,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(High-Confidence)",
                required_classes=required_classes, unit_to_class_map=unit_to_class_map
            )

            # Get the FULL working pool for cache validation purposes, also correctly filtered
            full_validation_pool, _, _ = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, combined_exclusions,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Cache Validation)",
                required_classes=None, unit_to_class_map=None
            )

            # Tier 0: High-Priority JSON Map Override (runs for ALL factions)
            if faction_json_utils.find_and_apply_json_maa_override(element, json_group_data, f"'{faction_name}'", full_json_map=faction_culture_map, all_units=all_units, ck3_maa_definitions=ck3_maa_definitions, excluded_by_json=excluded_by_json, excluded_units_set=excluded_units_set):
                replacements_made += 1
                continue # Move to the next MenAtArm element

            # --- REFACTORED: Cache-First Logic ---
            if llm_helper:
                req_id = f"{faction_name}|MenAtArm|{maa_definition_name}|tier_{tier if tier is not None else 'any'}"
                if not llm_helper.force_refresh and req_id in llm_helper.cache:
                    cached_entry = llm_helper.cache.get(req_id)
                    if isinstance(cached_entry, dict):
                        chosen_unit = cached_entry.get("chosen_unit")

                        # Perform all validation checks
                        is_stale = chosen_unit not in full_validation_pool
                        is_recache_flagged = chosen_unit in llm_helper.units_to_recache
                        is_globally_excluded = llm_helper.excluded_units_set and chosen_unit in llm_helper.excluded_units_set
                        is_json_excluded = excluded_by_json and chosen_unit in excluded_by_json

                        is_valid = chosen_unit and chosen_unit in all_units and not is_stale and not is_recache_flagged and not is_globally_excluded and not is_json_excluded

                        if is_valid:
                            element.set('key', chosen_unit)
                            print(f"      -> SUCCESS: Replaced unit with '{chosen_unit}' from LLM cache.")
                            replacements_made += 1
                            continue  # Valid cache entry found, move to next element

                        # If we are here, the cache entry is invalid.
                        if chosen_unit:
                            # Log the reason for eviction
                            if is_stale:
                                print(f"      -> INFO: Evicting stale LLM cache entry for '{req_id}'. Cached unit '{chosen_unit}' is no longer in the faction's (filtered) full potential unit pool.")
                            elif is_json_excluded:
                                print(f"      -> INFO: Evicting stale LLM cache entry for '{req_id}'. Cached unit '{chosen_unit}' is now excluded by a JSON rule.")
                            elif is_globally_excluded:
                                print(f"      -> INFO: Evicting stale LLM cache entry for '{req_id}'. Cached unit '{chosen_unit}' is in the global exclusion list.")
                            elif is_recache_flagged:
                                print(f"      -> INFO: Evicting LLM cache entry for '{req_id}'. Unit '{chosen_unit}' is marked for recache.")

                        # Evict from cache
                        with llm_helper.cache_lock:
                            if req_id in llm_helper.cache:
                                del llm_helper.cache[req_id]
                                cache_modified_in_pass = True

                        # If the reason for eviction was staleness, exclusion, or a recache flag, queue for LLM and skip procedural logic.
                        if is_stale or is_globally_excluded or is_json_excluded or is_recache_flagged:
                            print(f"      -> Queuing for LLM pass due to stale cache, exclusion, or recache flag.")
                            failures.append({
                                'element': element,
                                'faction_element': faction,
                                'tag_name': 'MenAtArm',
                                'maa_definition_name': maa_definition_name,
                                'internal_type': internal_type,
                                'unit_role_description': maa_definition_name,
                                'excluded_by_json': excluded_by_json,
                                'tier': tier,
                            })
                            continue  # Skip to next MenAtArm element
            # --- END REFACTORED ---

            # If JSON override did not apply, check for a working pool before proceeding with procedural passes.
            if not working_pool:
                print(f"    -> WARNING: No unit pool for faction {log_faction_str}. Cannot process procedurally. Queuing for LLM pass.")
                # Only add to failures if the key is still missing after the JSON pass
                if not element.get('key'):
                     failures.append({
                        'element': element,
                        'faction_element': faction,
                        'tag_name': 'MenAtArm',
                        'maa_definition_name': maa_definition_name,
                        'internal_type': internal_type,
                        'unit_role_description': maa_definition_name,
                        'excluded_by_json': excluded_by_json,
                        'tier': tier,
                    })
                continue # Move to the next MenAtArm element

            # --- START MODIFIED LOGIC ---
            # Tier 1: High-Confidence Class-based Match
            replacement_successful = unit_management.find_and_apply_high_confidence_replacement(
                element, log_faction_str, working_pool, tier, unit_variant_map,
                ck3_maa_definitions, unit_to_class_map, unit_to_description_map, threshold=first_pass_threshold
            )

            if replacement_successful:
                replacements_made += 1
                print(f"    -> SUCCESS: Replaced unit procedurally with high-confidence match.")
            else:
                # If high-confidence procedural pass fails, add it to the list for the LLM pass
                print(f"    -> FAILED: No high-confidence match found. Queuing for LLM pass.")
                failures.append({
                    'element': element,
                    'faction_element': faction,
                    'tag_name': 'MenAtArm',
                    'maa_definition_name': maa_definition_name,
                    'internal_type': internal_type,
                    'unit_role_description': maa_definition_name,
                    'excluded_by_json': excluded_by_json,
                    'tier': tier,
                })
            # --- END MODIFIED LOGIC ---

    if cache_modified_in_pass and llm_helper:
        print("High-confidence pass evicted stale entries. Saving LLM cache...")
        llm_helper.save_cache()

    print(f"High-confidence pass complete. Made {replacements_made} replacements, {len(failures)} units require LLM intervention.")
    return replacements_made, failures

def run_low_confidence_unit_pass(root, failures, ck3_maa_definitions, unit_to_class_map, unit_variant_map, unit_to_description_map, categorized_units, unit_categories, unit_stats_map, all_units, excluded_units_set=None, faction_to_heritage_map=None, heritage_to_factions_map=None, screen_name_to_faction_key_map=None, faction_key_to_units_map=None, llm_helper=None, unit_to_training_level=None, faction_elite_units=None, faction_to_json_map=None, faction_culture_map=None, faction_pool_cache=None, faction_to_subculture_map=None, subculture_to_factions_map=None, faction_key_to_screen_name_map=None, culture_to_faction_map=None, faction_to_heritages_map=None):
    """
    Runs the final, low-confidence procedural passes on failures from the LLM step.
    This includes a class-based fallback, followed by a broader role-based fallback.
    Also handles applying LLM-generated levy compositions.
    """
    print(f"\nRunning low-confidence procedural pass for {len(failures)} remaining units...")
    replacements_made = 0
    remaining_failures = []

    # If a faction_pool_cache is not provided, create a local one for this pass.
    if faction_pool_cache is None:
        faction_pool_cache = {}

    # --- Stage 1: Class-based Fallback (more precise) ---
    if failures:
        print(f"\nAttempting class-based fallbacks for {len(failures)} units...")
        for failure_data in failures:
            replacement_successful = False
            element = failure_data.get('element') # Use .get() for safe access
            faction_element = failure_data.get('faction_element')
            if not faction_element:
                print(f"  -> ERROR: Failure data is missing 'faction_element'. Cannot process. Data: {failure_data}")
                remaining_failures.append(failure_data)
                continue

            faction_name = failure_data.get('faction_name') or faction_element.get('name')
            tag_name = failure_data['tag_name']
            tier = failure_data['tier']

            # --- NEW: Get pools dynamically ---
            json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None
            combined_exclusions = set(excluded_units_set) if excluded_units_set else set()
            json_exclusions = faction_json_utils.get_json_general_exclusions_for_faction(
                faction_name, json_group_data, faction_culture_map, all_units, log_context="Low-Confidence"
            )
            combined_exclusions.update(json_exclusions)

            working_pool, log_faction_str, tiered_pools = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, combined_exclusions,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Low-Confidence)"
            )
            # --- END NEW ---

            # --- NEW: Handle LevyComposition failures ---
            if tag_name == 'LevyComposition':
                req_id = f"{faction_name}|LevyComposition|tier_{tier if tier is not None else 'any'}"
                cached_composition = None
                if llm_helper:
                    cached_entry = llm_helper.cache.get(req_id)
                    if isinstance(cached_entry, dict):
                        cached_composition = cached_entry.get("chosen_composition")

                if cached_composition:
                    print(f"  -> Applying LLM-generated levy composition for faction '{faction_name}': {cached_composition}")
                    # Call manage_faction_levies with the LLM-generated composition
                    changes, _ = unit_management.manage_faction_levies(
                        faction_element, working_pool, unit_to_training_level, unit_categories, log_faction_str,
                        unit_to_class_map, faction_to_json_map.get(faction_name) if faction_to_json_map else {}, all_units, tier, # unit_to_tier_map is not used in manage_faction_levies
                        faction_elite_units=faction_elite_units, faction_name=faction_name,
                        excluded_units_set=excluded_units_set, destructive_on_failure=False, # Don't destroy if LLM provided a plan
                        levy_composition_override=cached_composition
                    )
                    if changes:
                        replacements_made += 1
                        print(f"  -> SUCCESS: Applied LLM-generated levy composition for faction '{faction_name}'.")
                    else:
                        print(f"  -> WARNING: LLM-generated levy composition for faction '{faction_name}' resulted in no changes.")
                    replacement_successful = True # Mark as successful even if no changes, as the plan was applied
                else:
                    # LLM failed to provide a composition, apply a balanced default
                    default_composition = {
                        'spear': 25, 'infantry': 25, 'missile': 25, 'cavalry': 25
                    }
                    print(f"  -> LLM failed to provide levy composition for faction '{faction_name}'. Applying default: {default_composition}")
                    changes, _ = unit_management.manage_faction_levies(
                        faction_element, working_pool, unit_to_training_level, unit_categories, log_faction_str,
                        unit_to_class_map, faction_to_json_map.get(faction_name) if faction_to_json_map else {}, all_units, tier, # unit_to_tier_map is not used in manage_faction_levies
                        faction_elite_units=faction_elite_units, faction_name=faction_name,
                        excluded_units_set=excluded_units_set, destructive_on_failure=False,
                        levy_composition_override=default_composition
                    )
                    if changes:
                        replacements_made += 1
                        print(f"  -> SUCCESS: Applied default levy composition for faction '{faction_name}'.")
                    else:
                        print(f"  -> WARNING: Default levy composition for faction '{faction_name}' resulted in no changes.")
                    replacement_successful = True # Mark as successful, as a fallback was applied

                if replacement_successful:
                    continue # Move to the next failure, this one is handled
            # --- END NEW ---

            if tag_name == 'MenAtArm':
                maa_definition_name = failure_data['maa_definition_name']
                internal_type = ck3_maa_definitions.get(maa_definition_name)
                found_key = None

                # --- NEW: Gradual Fallback Logic ---
                if internal_type:
                    # Exclude units already used as Levies in this faction
                    levy_keys_in_faction = {el.get('key') for el in faction_element.findall('Levies') if el.get('key')}
                    if levy_keys_in_faction:
                        print(f"    -> (Low-Confidence) Excluding {len(levy_keys_in_faction)} levy units for MAA '{maa_definition_name}'.")

                    accumulated_pool = set()

                    if tiered_pools:
                        for i, tier_pool in enumerate(tiered_pools):
                            accumulated_pool.update(tier_pool)
                            if not accumulated_pool:
                                continue # Skip empty tiers

                            # Create a temporary pool for this tier's search, applying all exclusions
                            current_search_pool = accumulated_pool
                            if excluded_units_set:
                                current_search_pool = current_search_pool - excluded_units_set
                            if levy_keys_in_faction:
                                current_search_pool = current_search_pool - levy_keys_in_faction

                            if not current_search_pool:
                                continue # Skip if exclusions emptied the pool

                            print(f"    -> Low-Confidence Pass (Cultural Tier {i + 1}/7): Class-based search for '{maa_definition_name}' in pool of {len(current_search_pool)} units...")
                            found_key = unit_selector._find_replacement_for_maa(internal_type, current_search_pool, unit_to_class_map, tier, unit_variant_map, unit_to_description_map, threshold=0.90)
                            if found_key:
                                break # Found a suitable unit, exit the loop

                    # Final fallback to the global unit pool if no unit was found in any cultural tier
                    if not found_key:
                        print(f"    -> Low-Confidence Pass (Global Fallback): Class-based search in global pool for '{maa_definition_name}'...")
                        global_pool = all_units
                        if excluded_units_set:
                            global_pool = global_pool - excluded_units_set
                        if levy_keys_in_faction:
                            global_pool = global_pool - levy_keys_in_faction
                        # Use a lenient threshold for the global fallback to ensure we get the best possible match
                        found_key = unit_selector._find_replacement_for_maa(internal_pool, global_pool, unit_to_class_map, tier, unit_variant_map, unit_to_description_map, threshold=0.0)
                # --- END: Gradual Fallback Logic ---

                if found_key:
                    if element is None: # This shouldn't happen for MenAtArm, but as a safeguard
                        print(f"  -> WARNING: MenAtArm element was None for '{maa_definition_name}'. Cannot apply replacement.")
                    else:
                        element.set('key', found_key)
                        replacement_successful = True
            elif tag_name in ['General', 'Knights', 'Levies', 'Garrison']:
                # For these, we don't have a direct 'internal_type' to use _find_replacement_for_maa
                # We'll skip the class-based fallback here and go straight to best-effort role/category.
                # This is because _find_replacement_for_maa is specifically for MAA types.
                pass # Will be handled in Stage 2

            if replacement_successful:
                replacements_made += 1
                print(f"  -> Class-based fallback SUCCESS: Replaced unit for tag '<{tag_name}>' in faction {log_faction_str}.")
            else:
                # Add the dynamically fetched pools to the failure data for the next pass
                failure_data['working_pool'] = working_pool
                failure_data['log_faction_str'] = log_faction_str
                remaining_failures.append(failure_data)

    if not remaining_failures:
        return replacements_made

    # --- Stage 2: Best-effort Role/Category Fallback (broader) ---
    print(f"\nAttempting best-effort role/category fallbacks for {len(remaining_failures)} remaining units...")
    for failure_data in remaining_failures:
        element = failure_data.get('element') # Use .get() for safe access
        faction_element = failure_data.get('faction_element')
        if not faction_element:
            print(f"  -> ERROR: Failure data is missing 'faction_element'. Cannot process. Data: {failure_data}")
            continue # Skip this failure

        faction_name = faction_element.get('name')
        log_faction_str = failure_data.get('log_faction_str') # Use the one from the previous stage
        tag_name = failure_data['tag_name']
        working_pool = failure_data['working_pool'] # Use the one from the previous stage
        tier = failure_data['tier']
        unit_role_description = failure_data.get('unit_role_description')
        rank = failure_data.get('rank')
        level = failure_data.get('level')
        levy_slot = failure_data.get('levy_slot')

        # Combine global and JSON-specific exclusions for the fallback function
        excluded_by_json = failure_data.get('excluded_by_json', set())
        combined_exclusions = set(excluded_units_set) if excluded_units_set else set()
        combined_exclusions.update(excluded_by_json)

        replacement_successful = unit_management.find_and_apply_best_effort_replacement(
            root,
            faction_name,
            element,
            log_faction_str,
            categorized_units,
            working_pool,
            unit_categories,
            tier,
            unit_variant_map,
            ck3_maa_definitions,
            unit_to_description_map,
            tag_name,
            unit_role_description,
            rank=rank,
            level=level,
            unit_stats_map=unit_stats_map,
            levy_slot=levy_slot,
            global_unit_pool=all_units,
            excluded_units_set=combined_exclusions
        )

        if replacement_successful:
                replacements_made += 1
                print(f"  -> Best-effort fallback SUCCESS: Replaced unit for tag '<{tag_name}>' in faction {log_faction_str}.")
        else:
            unit_type_info = f" (type: {failure_data['maa_definition_name']})" if failure_data.get('maa_definition_name') else f" (role: {failure_data.get('unit_role_description')})"
            print(f"  -> ERROR: No replacement found for tag '<{tag_name}>{unit_type_info}' for faction {log_faction_str} after all strategies. This unit will remain missing.")

    return replacements_made

def ensure_levy_structure_and_percentages(root, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool, faction_to_subculture_map=None, subculture_to_factions_map=None, faction_key_to_screen_name_map=None, culture_to_faction_map=None, unit_to_class_map=None,
                                          faction_to_json_map=None, all_units=None, unit_to_training_level=None, tier=None, faction_elite_units=None, excluded_units_set=None, faction_pool_cache=None, faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, destructive_on_failure=True, faction_culture_map=None, is_submod_mode=False, factions_in_main_mod=None):
    """
    Iterates through all factions and ensures their levy structure and percentages are correct
    using the new `manage_faction_levies` function.
    Returns (factions_changed_count, list of failures).
    """
    print("\nEnsuring levy structure and percentages for all factions...")
    if is_submod_mode:
        print("  -> Submod mode: Will only process factions not present in the main mod.")
    factions_changed_count = 0
    all_levy_failures = []

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')

        # In submod mode, skip factions that are already in the main mod.
        if is_submod_mode and factions_in_main_mod and faction_name in factions_in_main_mod:
            continue

        # --- NEW: Faction-specific Excluded Units ---
        # Start with a copy of the global excluded set
        combined_excluded_units = set(excluded_units_set) if excluded_units_set else set()
        json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None

        json_exclusions = faction_json_utils.get_json_general_exclusions_for_faction(
            faction_name, json_group_data, faction_culture_map, all_units, log_context="Levies"
        )
        combined_excluded_units.update(json_exclusions)

        working_pool = set()
        log_faction_str = f"'{faction_name}'"

        if faction_name == "Default":
            working_pool = template_faction_unit_pool
            log_faction_str = "'Default' (template)"
            if not working_pool:
                num_levy_slots_to_fill = random.randint(3, 5)
                for i in range(num_levy_slots_to_fill):
                    all_levy_failures.append({
                        'faction_element': faction,
                        'tag_name': 'Levies',
                        'unit_role_description': f"A low-quality levy unit (slot {i+1}).",
                        'tier': tier,
                        'unit_categories': unit_categories,
                        'unit_to_class_map': unit_to_class_map,
                        'levy_slot': i + 1
                    })
                print(f"  -> WARNING (Levies): Default faction template pool is empty. Cannot manage levies for Default faction. Queued {num_levy_slots_to_fill} requests for LLM.")
                continue
        else:
            # Use the centralized helper function to get the working pool for non-Default factions
            working_pool, log_faction_str, _ = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, combined_excluded_units,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Levies)"
            )

        # NEW: Get JSON group data for this faction
        json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None

        # NEW: Retrieve faction-specific elite units
        faction_specific_elite_units = faction_elite_units.get(faction_name, set()) if faction_elite_units else set()

        changes, failures = unit_management.manage_faction_levies(faction, working_pool, unit_to_training_level, unit_categories, log_faction_str, unit_to_class_map, json_group_data, all_units, tier, faction_elite_units=faction_specific_elite_units, faction_name=faction_name, excluded_units_set=combined_excluded_units, destructive_on_failure=destructive_on_failure, levy_composition_override=None)
        if changes:
            factions_changed_count += 1
        all_levy_failures.extend(failures)

    if factions_changed_count > 0:
        print(f"Adjusted levy structure and percentages for {factions_changed_count} factions.")
    else:
        print("All levy structures and percentages are already correct.")

    return factions_changed_count, all_levy_failures

def ensure_garrison_structure(root, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool, faction_to_subculture_map=None, subculture_to_factions_map=None, faction_key_to_screen_name_map=None, culture_to_faction_map=None, unit_to_class_map=None, general_units=None, unit_to_training_level=None, tier=None, unit_to_tier_map=None, excluded_units_set=None, faction_pool_cache=None, faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, destructive_on_failure=True,
                                 faction_to_json_map=None, all_units=None, faction_culture_map=None, is_submod_mode=False, factions_in_main_mod=None): # Added faction_to_json_map and all_units
    """
    Iterates through all factions and ensures their garrison structure is correct
    using the `manage_faction_garrisons` function.
    Returns (factions_changed_count, list of failures).
    """
    print("\nEnsuring garrison structure and percentages for all factions...")
    if is_submod_mode:
        print("  -> Submod mode: Will only process factions not present in the main mod.")
    factions_changed_count = 0
    all_garrison_failures = []

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')

        # In submod mode, skip factions that are already in the main mod.
        if is_submod_mode and factions_in_main_mod and faction_name in factions_in_main_mod:
            continue

        # --- NEW: Faction-specific Excluded Units ---
        # Start with a copy of the global excluded set
        combined_excluded_units = set(excluded_units_set) if excluded_units_set else set()
        json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None

        json_exclusions = faction_json_utils.get_json_general_exclusions_for_faction(
            faction_name, json_group_data, faction_culture_map, all_units, log_context="Garrisons"
        )
        combined_excluded_units.update(json_exclusions)

        working_pool = set()
        log_faction_str = f"'{faction_name}'"

        if faction_name == "Default":
            working_pool = template_faction_unit_pool
            log_faction_str = "'Default' (template)"
            if not working_pool:
                all_garrison_failures.append({
                    'faction_element': faction,
                    'tag_name': 'Garrison',
                    'unit_role_description': "A low-quality garrison unit for a level 1 fortification.",
                    'tier': tier,
                    'unit_categories': unit_categories,
                    'unit_to_class_map': unit_to_class_map,
                    'level': '1',
                    'garrison_slot': 1
                })
                print(f"  -> WARNING (Garrisons): Default faction template pool is empty. Cannot manage garrisons for Default faction.")
                continue
        else:
            # Use the centralized helper function to get the working pool for non-Default factions
            working_pool, log_faction_str, _ = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, combined_excluded_units,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Garrisons)"
            )

        # NEW: Get JSON group data for this faction
        json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None

        changes, failures = unit_management.manage_faction_garrisons(
            faction, working_pool, unit_to_training_level, unit_categories, log_faction_str,
            general_units, unit_to_class_map, tier, unit_to_tier_map,
            excluded_units_set=combined_excluded_units, destructive_on_failure=destructive_on_failure,
            faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map,
            screen_name_to_faction_key_map=screen_name_to_faction_key_map, faction_key_to_units_map=faction_key_to_units_map,
            json_group_data=json_group_data, all_units=all_units # Pass new args
        )
        if changes:
            factions_changed_count += 1
        all_garrison_failures.extend(failures)

    if factions_changed_count > 0:
        print(f"Adjusted garrison structure and percentages for {factions_changed_count} factions.")
    else:
        print("All garrison structures and percentages are already correct.")

    return factions_changed_count, all_garrison_failures

def run_final_fix_pass(root, validation_failures, categorized_units, all_units, unit_categories, tier, unit_variant_map, ck3_maa_definitions, unit_to_description_map, unit_stats_map, general_units, unit_to_class_map, excluded_units_set, screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, faction_pool_cache, faction_to_json_map=None, faction_culture_map=None, llm_helper=None, unit_to_training_level=None, faction_elite_units=None):
    """
    Attempts a final, best-effort fix for any remaining validation failures.
    Uses the global all_units pool for maximum fallback.
    Returns the number of fixes successfully applied.
    """
    print(f"\nAttempting final best-effort fix for {len(validation_failures)} validation issues...")
    fixes_applied_count = 0

    # Filter all_units once for efficiency using only global exclusions.
    # Faction-specific exclusions will be handled inside the loop.
    filtered_all_units = all_units - excluded_units_set if excluded_units_set else all_units

    # NEW: Prioritize fixes to handle dependencies (e.g., MenAtArm fallback needs Levies to have keys)
    def get_sort_key(failure):
        tag = failure['tag_name']
        is_max_error = failure.get('validation_error') == 'missing_max_attribute'

        # Priority Order:
        # 0: Key/tag errors for core structure (Levies, Garrison, General, Knights)
        # 1: Key/tag errors for MenAtArm (may depend on Levies)
        # 100+: 'max' attribute errors (depend on keys being present)

        priority = 99 # Default for unknown tags
        if tag in ['Levies', 'Garrison', 'General', 'Knights']:
            priority = 0
        elif tag == 'MenAtArm':
            priority = 1
        elif tag == 'LevyComposition': # NEW: Handle LevyComposition failures
            priority = -1 # Highest priority to resolve composition first

        if is_max_error:
            return 100 + priority

        # This assumes if it's not a max error, it's a key/missing tag error
        return priority

    sorted_failures = sorted(validation_failures, key=get_sort_key)

    for failure_data in sorted_failures:
        faction_element = failure_data['faction_element']
        faction_name = faction_element.get('name')

        # Get faction-specific JSON exclusions and combine with global ones
        combined_exclusions_for_faction = set(excluded_units_set) if excluded_units_set else set()
        if faction_to_json_map:
            json_group_data = faction_to_json_map.get(faction_name)
            json_exclusions = faction_json_utils.get_json_general_exclusions_for_faction(
                faction_name, json_group_data, faction_culture_map, all_units, log_context="Final Fix Pass"
            )
            combined_exclusions_for_faction.update(json_exclusions)

        # --- NEW: Exclude levy keys for MenAtArm failures ---
        if failure_data['tag_name'] == 'MenAtArm':
            levy_keys_in_faction = {el.get('key') for el in faction_element.findall('Levies') if el.get('key')}
            if levy_keys_in_faction:
                combined_exclusions_for_faction.update(levy_keys_in_faction)
                print(f"  -> (Final Fix) Excluding {len(levy_keys_in_faction)} levy units for MAA '{failure_data.get('maa_definition_name')}'.")
        # --- END NEW ---

        # Get the proper working pool, now filtered with combined exclusions
        working_pool, log_faction_str, _ = faction_xml_utils.get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, combined_exclusions_for_faction,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Final Fix)"
        )

        tag_name = failure_data['tag_name']
        element_to_fix = failure_data.get('element') # This will be None if it's a missing tag

        # Extract specific attributes for _find_and_apply_best_effort_replacement
        rank = failure_data.get('rank')
        level = failure_data.get('level')
        maa_definition_name = failure_data.get('maa_definition_name')
        unit_role_description = failure_data.get('unit_role_description')

        # --- NEW: Handle LevyComposition failures in final fix pass ---
        if tag_name == 'LevyComposition':
            req_id = f"{faction_name}|LevyComposition|tier_{tier if tier is not None else 'any'}"
            cached_composition = None
            if llm_helper:
                cached_entry = llm_helper.cache.get(req_id)
                if isinstance(cached_entry, dict):
                    cached_composition = cached_entry.get("chosen_composition")

            if cached_composition:
                print(f"  -> Applying LLM-generated levy composition for faction '{faction_name}' during final fix: {cached_composition}")
                changes, _ = unit_management.manage_faction_levies(
                    faction_element, working_pool, unit_to_training_level, unit_categories, log_faction_str,
                    unit_to_class_map, faction_to_json_map.get(faction_name) if faction_to_json_map else {}, all_units, tier,
                    faction_elite_units=faction_elite_units, faction_name=faction_name,
                    excluded_units_set=excluded_units_set, destructive_on_failure=False,
                    levy_composition_override=cached_composition
                )
                if changes:
                    fixes_applied_count += 1
                    print(f"    -> SUCCESS: Applied LLM-generated levy composition for faction '{faction_name}'.")
                else:
                    print(f"    -> WARNING: LLM-generated levy composition for faction '{faction_name}' resulted in no changes.")
            else:
                # LLM failed, apply a hardcoded default
                default_composition = {
                    'spear': 25, 'infantry': 25, 'missile': 25, 'cavalry': 25
                }
                print(f"  -> LLM failed to provide levy composition for faction '{faction_name}' during final fix. Applying default: {default_composition}")
                changes, _ = unit_management.manage_faction_levies(
                    faction_element, working_pool, unit_to_training_level, unit_categories, log_faction_str,
                    unit_to_class_map, faction_to_json_map.get(faction_name) if faction_to_json_map else {}, all_units, tier,
                    faction_elite_units=faction_elite_units, faction_name=faction_name,
                    excluded_units_set=excluded_units_set, destructive_on_failure=False,
                    levy_composition_override=default_composition
                )
                if changes:
                    fixes_applied_count += 1
                    print(f"    -> SUCCESS: Applied default levy composition for faction '{faction_name}'.")
                else:
                    print(f"    -> WARNING: Default levy composition for faction '{faction_name}' resulted in no changes.")
            continue # Move to next failure, this one is handled
        # --- END NEW ---

        # NEW: Handle missing_max_attribute specifically
        if failure_data.get('validation_error') == 'missing_max_attribute' and element_to_fix is not None:
            print(f"  -> Attempting to fix missing 'max' attribute for <{tag_name}> in faction '{faction_name}'...")
            if tag_name in ['Levies', 'Garrison']:
                element_to_fix.set('max', 'LEVY')
                fixes_applied_count += 1
                print(f"    -> SUCCESS: Added max='LEVY' to <{tag_name}> for faction '{faction_name}'.")
                continue # Skip further processing for this failure, it's fixed.
            elif tag_name == 'MenAtArm':
                unit_key = element_to_fix.get('key')
                maa_definition_name = element_to_fix.get('type') # Get type from element
                if not unit_key:
                    print(f"    -> FAILED: Cannot fix missing 'max' for <MenAtArm type='{maa_definition_name}'> because it has no 'key' attribute.")
                    continue

                internal_type = ck3_maa_definitions.get(maa_definition_name)
                is_siege_by_ck3_type = (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) is None) or \
                                       (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type) is None)
                unit_class = unit_to_class_map.get(unit_key)
                is_siege_by_attila_class = (unit_class == 'art_siege')
                unit_category = unit_categories.get(unit_key)
                is_siege_by_attila_category = (unit_category == 'artillery')
                is_siege = is_siege_by_ck3_type or is_siege_by_attila_class or is_siege_by_attila_category

                if is_siege:
                    # This is a siege unit, it should not have a 'max' attribute.
                    if element_to_fix.get('siege') != 'true':
                        element_to_fix.set('siege', 'true')
                        fixes_applied_count += 1
                        print(f"    -> SUCCESS: Identified unit '{unit_key}' as siege weapon and added siege='true'.")
                    if 'max' in element_to_fix.attrib:
                        del element_to_fix.attrib['max']
                        fixes_applied_count += 1
                    continue
                else:
                    # This is NOT a siege unit, it MUST have a 'max' attribute.
                    max_value = mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) or \
                                (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type))

                    if not max_value:
                        # Fallback to unit's specific category if mapping is missing
                        specific_category = unit_categories.get(unit_key)
                        max_value = mappings.ATTILA_CATEGORY_TO_MAX_VALUE.get(specific_category, "INFANTRY")
                        print(f"    -> WARNING: Could not find CK3 mapping for <MenAtArm type='{maa_definition_name}'>. Applying fallback max='{max_value}' based on unit's category '{specific_category}'.")

                    element_to_fix.set('max', max_value)
                    fixes_applied_count += 1
                    print(f"    -> SUCCESS: Added max='{max_value}' to <MenAtArm type='{maa_definition_name}'>.")

                    if 'siege' in element_to_fix.attrib:
                        del element_to_fix.attrib['siege']
                        fixes_applied_count += 1
                    continue

        print(f"  -> Attempting to fix <{tag_name}> for faction {log_faction_str} (Role: '{unit_role_description}')...")

        # Call the existing best-effort replacement function
        # Pass `element_to_fix` (which can be None) to allow creation of missing tags
        replacement_successful = unit_management.find_and_apply_best_effort_replacement(
            root,
            faction_name,
            element_to_fix, # Pass the specific element if it exists, else None
            log_faction_str, # Use the log string from the pool getter
            categorized_units,
            working_pool, # Use the proper, heritage-aware pool
            unit_categories,
            tier, # Pass tier_arg
            unit_variant_map,
            ck3_maa_definitions,
            unit_to_description_map,
            tag_name, # Corrected position for tag_name
            unit_role_description, # Corrected position for unit_role_description
            rank=rank,
            level=level,
            unit_stats_map=unit_stats_map, # Passed as keyword
            levy_slot=failure_data.get('levy_slot'), # Pass levy_slot if present
            global_unit_pool=filtered_all_units, # Pass the globally filtered pool
            excluded_units_set=combined_exclusions_for_faction # Pass the combined exclusions
        )

        if replacement_successful:
            fixes_applied_count += 1
            print(f"    -> SUCCESS: Fixed <{tag_name}> for faction {log_faction_str}.")
        else:
            print(f"    -> FAILED: Could not fix <{tag_name}> for faction {log_faction_str}.")

    return fixes_applied_count
