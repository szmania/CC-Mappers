from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mapper_tools import faction_xml_utils
from mapper_tools import unit_management
from mapper_tools import unit_selector
from mapper_tools import ck3_to_attila_mappings as mappings

def prepare_llm_request(failure_data, screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map, excluded_units_set, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, all_units, unit_to_class_map, unit_to_tier_map, unit_to_description_map, unit_stats_map, faction_pool_cache, max_tier_level=None, min_pool_size=5, override_pool=None):
    """
    Prepares a single, consolidated LLM request object for a given failure.
    It constructs a prioritized list of candidate units based on the specified tier level.
    The final unit_pool for the LLM prompt will be curated later based on batch size.
    Returns None if the constructed pool is smaller than min_pool_size.
    """
    faction_name = failure_data['faction_element'].get('name', 'Unknown')
    tag_name = failure_data['tag_name']

    # --- NEW: Handle LevyComposition requests ---
    if tag_name == 'LevyComposition':
        req_id = f"{faction_name}|LevyComposition|tier_{failure_data['tier'] if failure_data['tier'] is not None else 'any'}"
        llm_request_obj = {
            'id': req_id,
            'faction': faction_name,
            'tag_name': 'LevyComposition',
            'unit_role_description': failure_data['unit_role_description'],
            'available_levy_categories': failure_data['available_levy_categories'],
            'tier': failure_data['tier'],
            'excluded_units_set': list(failure_data['excluded_units_set']) if failure_data['excluded_units_set'] else []
        }
        return req_id, llm_request_obj
    # --- END NEW ---

    # 1. Build the request ID
    req_id_parts = [faction_name, tag_name]
    if failure_data.get('maa_definition_name'):
        req_id_parts.append(failure_data['maa_definition_name'])
    if failure_data.get('rank'):
        req_id_parts.append(f"rank_{failure_data['rank']}")
    if failure_data.get('level'):
        req_id_parts.append(f"level_{failure_data['level']}")
    if failure_data.get('garrison_slot'):
        req_id_parts.append(f"slot_{failure_data['garrison_slot']}")
    if failure_data.get('levy_slot'):
        req_id_parts.append(f"slot_{failure_data['levy_slot']}")
    req_id_parts.append(f"tier_{failure_data['tier'] if failure_data['tier'] is not None else 'any'}")
    req_id = "|".join(req_id_parts)

    # 2. Construct the unit pool
    if override_pool is not None:
        # Use the provided override pool (for global thematic search)
        current_tier_pool = set(override_pool)
    else:
        # Use the tiered cultural pool logic
        # The cache key for faction_pool_cache should include excluded_units_set
        cache_key = faction_name # Changed cache key to only faction_name
        if cache_key not in faction_pool_cache:
            tiered_pools, _ = faction_xml_utils.get_all_tiered_pools(
                faction_name, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, faction_to_heritage_map=faction_to_heritage_map,
                heritage_to_factions_map=heritage_to_factions_map, faction_to_heritages_map=faction_to_heritages_map
            )
            faction_pool_cache[cache_key] = (tiered_pools, _)
        tiered_pools, _ = faction_pool_cache[cache_key]

        # Apply exclusions dynamically after retrieving from cache
        filtered_tiered_pools = []
        if excluded_units_set:
            for pool in tiered_pools:
                filtered_tiered_pools.append(pool - excluded_units_set)
        else:
            filtered_tiered_pools = tiered_pools

        current_tier_pool = set()
        num_tiers_available = len(filtered_tiered_pools) # Use filtered_tiered_pools here
        # If max_tier_level is None, use all available tiers. Otherwise, use up to the specified level.
        effective_max_tier = num_tiers_available if max_tier_level is None else max_tier_level + 1

        for i in range(effective_max_tier):
            if i < num_tiers_available:
                current_tier_pool.update(filtered_tiered_pools[i]) # Use filtered_tiered_pools here

    # --- NEW: Apply JSON exclusions ---
    excluded_by_json = failure_data.get('excluded_by_json')
    if excluded_by_json:
        current_tier_pool = current_tier_pool - excluded_by_json
    # --- END NEW ---

    if len(current_tier_pool) < min_pool_size:
        return None, None # Not enough units to form a valid request for this tier.

    # 3. Build prioritized candidate lists, but only with units from the current tiered pool.
    primary_candidates = []
    secondary_candidates_with_scores = []

    expected_attila_classes = None
    if tag_name == 'MenAtArm':
        maa_def_name = failure_data['maa_definition_name']
        internal_type = failure_data.get('internal_type')
        expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(maa_def_name) or \
                                  (mappings.CK3_TYPE_TO_ATTILA_CLASS.get(internal_type) if internal_type else None)
    elif tag_name in ['General', 'Knights']:
        expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get('heavy_cavalry')
    elif tag_name == 'Levies':
        expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get('light_infantry')
    elif tag_name == 'Garrison':
        expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get('heavy_infantry')

    # Iterate through the tiered pool to create candidates
    for unit_key in sorted(list(current_tier_pool)): # Sort for determinism
        unit_class = unit_to_class_map.get(unit_key)
        if expected_attila_classes and unit_class in expected_attila_classes:
            primary_candidates.append(unit_key)
        else:
            stats = unit_stats_map.get(unit_key)
            score = unit_selector._calculate_quality_score(stats) if stats else 0
            secondary_candidates_with_scores.append((score, unit_key))

    secondary_candidates_with_scores.sort(key=lambda x: x[0], reverse=True)

    if not primary_candidates and not secondary_candidates_with_scores:
        print(f"      -> WARNING: No primary/secondary candidates identified for '{req_id}' in the current tier pool. This should not happen if pool has units.")
        return None, None

    # 4. Build the final request object
    faction_key = screen_name_to_faction_key_map.get(faction_name)
    subculture = faction_to_subculture_map.get(faction_key)

    llm_request_obj = {
        'id': req_id,
        'faction': faction_name,
        'subculture': subculture,
        'validation_pool': sorted(list(current_tier_pool)), # Full pool for this tier for validation
        'primary_candidates': primary_candidates,
        'secondary_candidates': [key for score, key in secondary_candidates_with_scores],
        'tier': failure_data['tier']
    }

    if tag_name == 'MenAtArm':
        llm_request_obj['maa_type'] = failure_data['maa_definition_name']
        llm_request_obj['expected_attila_classes'] = expected_attila_classes
    else:
        llm_request_obj.update({
            'tag_name': tag_name,
            'unit_role_description': failure_data['unit_role_description'],
            'rank': failure_data.get('rank'),
            'level': failure_data.get('level'),
            'garrison_slot': failure_data.get('garrison_slot'),
            'levy_slot': failure_data.get('levy_slot')
        })

    return req_id, llm_request_obj

def run_iterative_llm_pass(llm_helper, all_llm_failures_to_process, time_period_context, llm_threads, llm_batch_size,
                             faction_pool_cache, all_units, excluded_units_set, unit_to_tier_map, unit_to_class_map,
                             unit_to_description_map, unit_stats_map, screen_name_to_faction_key_map, faction_key_to_units_map,
                             faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                             culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, ck3_maa_definitions):
    """
    Runs the LLM pass iteratively, expanding the unit pool with each tier for failed requests.
    This function replaces the main LLM block in process_units_xml.
    """
    if not llm_helper or not all_llm_failures_to_process:
        log_msg = "LLM integration is disabled or no units required LLM intervention."
        print(f"\n{log_msg}")
        return 0, all_llm_failures_to_process

    log_msg = "cache and/or LLM" if llm_helper.network_calls_enabled else "LLM cache"
    print(f"\nAttempting to resolve {len(all_llm_failures_to_process)} missing units using {log_msg} with iterative tiered pools...")

    MAX_LLM_FAILURES_THRESHOLD = 500000
    if len(all_llm_failures_to_process) > MAX_LLM_FAILURES_THRESHOLD:
        print(f"  -> WARNING: Number of LLM requests ({len(all_llm_failures_to_process)}) exceeds threshold of {MAX_LLM_FAILURES_THRESHOLD}.")
        print("  -> This usually indicates a problem with the input data (e.g., TSV files) or configuration, causing the high-confidence pass to fail for most units.")
        print("  -> Skipping LLM pass to prevent hanging. Proceeding directly to low-confidence procedural fallback.")
        return 0, all_llm_failures_to_process

    llm_replacements_made = 0
    remaining_failures = list(all_llm_failures_to_process)

    # Determine max number of tiers from a sample faction. Default to 7 cultural tiers + 1 global thematic tier.
    num_cultural_tiers = 7
    if faction_pool_cache:
        sample_faction_name = next(iter(faction_pool_cache), None)
        if sample_faction_name:
            num_cultural_tiers = len(faction_pool_cache[sample_faction_name][0])

    total_passes = num_cultural_tiers + 1 # Add one for the global thematic pass

    for tier_level in range(total_passes):
        if not remaining_failures:
            break

        is_global_pass = (tier_level == num_cultural_tiers)
        pass_name = f"Global Thematic Search Pass ({tier_level + 1}/{total_passes})" if is_global_pass else f"Cultural Tier Pass {tier_level + 1}/{total_passes}"
        print(f"\n--- Running LLM Pass ({pass_name}) for {len(remaining_failures)} units ---")

        requests_for_this_tier = []
        element_map = {}
        failures_for_next_tier = []

        if is_global_pass:
            # --- NEW: Global Thematic Search Logic ---
            for failure_data in remaining_failures:
                # This global search is primarily for highly thematic MAA types.
                if failure_data['tag_name'] != 'MenAtArm':
                    failures_for_next_tier.append(failure_data)
                    continue

                global_candidates = unit_management.find_global_thematic_candidates(
                    failure_data, all_units, unit_to_class_map, unit_to_description_map, ck3_maa_definitions
                )

                if global_candidates:
                    # --- NEW: Exclude levy keys ---
                    current_excluded_units = set(excluded_units_set) if excluded_units_set else set()
                    if failure_data['tag_name'] == 'MenAtArm':
                        levy_keys_in_faction = {el.get('key') for el in failure_data['faction_element'].findall('Levies') if el.get('key')}
                        current_excluded_units.update(levy_keys_in_faction)
                    # --- END NEW ---
                    # We found potential global matches, create a request for the LLM
                    req_id, llm_request_obj = prepare_llm_request(
                        failure_data, screen_name_to_faction_key_map, faction_key_to_units_map,
                        faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                        culture_to_faction_map, current_excluded_units, faction_to_heritage_map,
                        heritage_to_factions_map, faction_to_heritages_map, all_units, unit_to_class_map, unit_to_tier_map,
                        unit_to_description_map, unit_stats_map, faction_pool_cache,
                        max_tier_level=tier_level, min_pool_size=1, # Use min_pool_size=1 for global pass
                        override_pool=global_candidates # Provide the globally found candidates
                    )
                    if req_id and llm_request_obj:
                        print(f"  -> Found {len(global_candidates)} global thematic candidates for '{failure_data['maa_definition_name']}'. Queuing for LLM.")
                        requests_for_this_tier.append(llm_request_obj)
                        element_map[req_id] = failure_data
                    else:
                        failures_for_next_tier.append(failure_data)
                else:
                    # No global candidates found, this unit will fail completely.
                    failures_for_next_tier.append(failure_data)
        else:
            # --- Existing Cultural Tier Logic ---
            for failure_data in remaining_failures:
                # --- NEW: Exclude levy keys ---
                current_excluded_units = set(excluded_units_set) if excluded_units_set else set()
                if failure_data['tag_name'] == 'MenAtArm':
                    levy_keys_in_faction = {el.get('key') for el in failure_data['faction_element'].findall('Levies') if el.get('key')}
                    current_excluded_units.update(levy_keys_in_faction)
                # --- END NEW ---
                req_id, llm_request_obj = prepare_llm_request(
                    failure_data, screen_name_to_faction_key_map, faction_key_to_units_map,
                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                    culture_to_faction_map, current_excluded_units, faction_to_heritage_map,
                    heritage_to_factions_map, faction_to_heritages_map, all_units, unit_to_class_map, unit_to_tier_map,
                    unit_to_description_map, unit_stats_map, faction_pool_cache,
                    max_tier_level=tier_level, min_pool_size=5
                )
                if req_id and llm_request_obj:
                    requests_for_this_tier.append(llm_request_obj)
                    element_map[req_id] = failure_data
                else:
                    failures_for_next_tier.append(failure_data)

        if not requests_for_this_tier:
            print("No requests with sufficient pool size for this tier. Moving to next tier.")
            remaining_failures = failures_for_next_tier
            continue

        # The rest of this logic is adapted from the original single-pass implementation
        cached_results, uncached_requests, cache_modified = llm_helper.filter_requests_against_cache(requests_for_this_tier)
        if cache_modified:
            llm_helper.save_cache()

        print(f"  -> Found {len(cached_results)} valid cached results. {len(uncached_requests)} requests require LLM call for this tier.")

        network_llm_results = {}
        if uncached_requests and llm_helper.network_calls_enabled:
            # --- NEW: Batching and parallel execution logic ---
            all_batches = [uncached_requests[i:i + llm_batch_size] for i in range(0, len(uncached_requests), llm_batch_size)]
            print(f"  -> Submitting {len(uncached_requests)} requests to LLM in {len(all_batches)} batches using {llm_threads} threads...")

            with ThreadPoolExecutor(max_workers=llm_threads) as executor:
                # Submit all batches to the executor
                future_to_batch = {
                    executor.submit(llm_helper.get_batch_unit_replacements, batch, time_period_context): batch
                    for batch in all_batches
                }

                processed_requests = 0
                total_requests_to_process = len(uncached_requests)

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    processed_requests += len(batch)
                    print(f"  -> LLM progress: {processed_requests}/{total_requests_to_process} requests completed for this tier.")
                    try:
                        batch_results = future.result()
                        if batch_results:
                            network_llm_results.update(batch_results)
                    except Exception as exc:
                        print(f"  -> ERROR: A batch of {len(batch)} requests generated an exception: {exc}")
                        # Optionally, log the failing batch requests for debugging
                        for req in batch:
                            print(f"    - Failed request ID: {req.get('id', 'N/A')}")
            # --- END NEW ---

        final_llm_results = {**cached_results, **network_llm_results}

        if final_llm_results:
            print(f"\nApplying {len(final_llm_results)} suggestions for this tier...")
            newly_failed_requests = []
            processed_req_ids = set()

            for req_id, suggestion in final_llm_results.items():
                processed_req_ids.add(req_id)
                req_data = element_map.get(req_id)
                if not req_data:
                    print(f"  -> WARNING: Could not find original request data for suggestion '{req_id}'.")
                    continue

                chosen_unit = suggestion.get("chosen_unit")
                chosen_composition = suggestion.get("chosen_composition") # NEW: For LevyComposition

                if chosen_unit or chosen_composition: # A non-None result is considered a success
                    # ... (apply the result to the XML element) ...
                    llm_replacements_made += 1
                else: # A None result is a failure for this tier
                    newly_failed_requests.append(req_data)

            # Determine which of the original requests were not processed at all (e.g., network error)
            for req in requests_for_this_tier:
                if req['id'] not in processed_req_ids:
                    newly_failed_requests.append(element_map[req['id']])

            remaining_failures = failures_for_next_tier + newly_failed_requests
        else:
            remaining_failures = failures_for_next_tier + [element_map[req['id']] for req in requests_for_this_tier]

    print(f"\nTotal LLM replacements made across all tiers: {llm_replacements_made}.")
    return llm_replacements_made, remaining_failures

def run_llm_roster_review_pass(root, llm_helper, time_period_context, llm_threads, llm_batch_size,
                                faction_pool_cache, all_units, excluded_units_set,
                                screen_name_to_faction_key_map, faction_key_to_units_map,
                                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map,
                                faction_to_heritages_map, ck3_maa_definitions):
    """
    Runs the LLM Roster Review pass. It gathers the current roster for each faction,
    sends it to the LLM for thematic and cultural review, and applies the suggested corrections.
    """
    if not llm_helper or not llm_helper.network_calls_enabled:
        print("ERROR: LLM Roster Review requires network calls to be enabled.")
        return 0

    print("\n--- Starting LLM Roster Review Pass ---")
    all_review_requests = []
    faction_element_map = {} # Maps faction name to its XML element

    for faction in root.findall('Faction'):
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        faction_element_map[faction_name] = faction

        # 1. Get the local unit pool for this faction
        local_unit_pool, _, _ = faction_xml_utils.get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map, heritage_to_factions_map,
            faction_to_heritages_map, log_prefix="(Roster Review)"
        )

        # 2. Build the structured roster object
        roster = defaultdict(list)
        for child in faction:
            if child.tag in ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']:
                unit_key = child.get('key')
                if not unit_key:
                    continue

                # Create a unique, serializable identifier for this specific tag
                identifier = {k: v for k, v in child.attrib.items() if k != 'key'}

                roster[child.tag].append({
                    "current_unit": unit_key,
                    "identifier": identifier
                })

        if not roster:
            print(f"  -> Skipping review for faction '{faction_name}' as it has no units.")
            continue

        # 3. Prepare the request
        req_id = f"review|{faction_name}"
        faction_key = screen_name_to_faction_key_map.get(faction_name)
        subculture = faction_to_subculture_map.get(faction_key)

        all_review_requests.append({
            'id': req_id,
            'faction': faction_name,
            'subculture': subculture,
            'roster': roster,
            'local_unit_pool': local_unit_pool
        })

    if not all_review_requests:
        print("No factions found to review.")
        return 0

    # 4. Filter against cache
    cached_results, uncached_requests, cache_modified = llm_helper.filter_requests_against_cache(all_review_requests)
    if cache_modified:
        llm_helper.save_cache()

    print(f"  -> Found {len(cached_results)} valid cached roster reviews. {len(uncached_requests)} factions require a new LLM review.")

    # 5. Send uncached requests to LLM
    network_llm_results = {}
    if uncached_requests:
        all_batches = [uncached_requests[i:i + llm_batch_size] for i in range(0, len(uncached_requests), llm_batch_size)]
        print(f"  -> Submitting {len(uncached_requests)} review requests to LLM in {len(all_batches)} batches using {llm_threads} threads...")

        with ThreadPoolExecutor(max_workers=llm_threads) as executor:
            future_to_batch = {
                executor.submit(llm_helper.get_batch_roster_reviews, batch, time_period_context): batch
                for batch in all_batches
            }
            processed_requests = 0
            for future in as_completed(future_to_batch):
                processed_requests += len(future_to_batch[future])
                print(f"  -> LLM review progress: {processed_requests}/{len(uncached_requests)} requests completed.")
                try:
                    batch_results = future.result()
                    if batch_results:
                        network_llm_results.update(batch_results)
                except Exception as exc:
                    print(f"  -> ERROR: A review batch generated an exception: {exc}")

    # 6. Apply corrections
    final_results = {**cached_results, **network_llm_results}
    total_corrections_applied = 0
    for req_id, result_data in final_results.items():
        faction_name = req_id.split('|')[1]
        faction_element = faction_element_map.get(faction_name)
        corrections = result_data.get("corrections", [])

        if not faction_element or not corrections:
            continue

        print(f"  -> Applying {len(corrections)} corrections for faction '{faction_name}'...")
        for correction in corrections:
            tag_to_find = correction['tag']
            identifier = correction['identifier']
            current_unit_from_llm = correction['current_unit']
            suggested_unit = correction['suggested_unit']

            # --- Improved Matching Logic ---
            element_to_modify = None
            log_current_unit = current_unit_from_llm

            # Pass 1: Strict match on identifier AND current_unit key
            for child in faction_element.findall(tag_to_find):
                # Use str(v) to handle potential type mismatches from JSON (e.g., int vs str)
                if all(child.get(k) == str(v) for k, v in identifier.items()) and child.get('key') == current_unit_from_llm:
                    element_to_modify = child
                    break

            # Pass 2: If no strict match, try a lenient match on identifier only, if it's unique
            if not element_to_modify:
                potential_matches = [
                    child for child in faction_element.findall(tag_to_find)
                    # Use str(v) here as well for consistency
                    if all(child.get(k) == str(v) for k, v in identifier.items())
                ]
                if len(potential_matches) == 1:
                    element_to_modify = potential_matches[0]
                    original_key = element_to_modify.get('key', 'N/A')
                    log_current_unit = original_key # Use the actual key for logging
                    # Only print the warning if the key is actually different
                    if original_key != current_unit_from_llm:
                        print(f"    -> WARNING: Found unique element for ID {identifier} in '{faction_name}', but its key has changed (expected '{current_unit_from_llm}', found '{original_key}'). Applying correction anyway.")

            if element_to_modify:
                element_to_modify.set('key', suggested_unit)
                total_corrections_applied += 1
                print(f"    - Changed <{tag_to_find}> (ID: {identifier}): '{log_current_unit}' -> '{suggested_unit}'. Reason: {correction.get('reason', 'N/A')}")
            else:
                print(f"    -> WARNING: Could not find a unique matching element for correction in '{faction_name}': Tag={tag_to_find}, ID={identifier}, current_unit='{current_unit_from_llm}'")

    return total_corrections_applied
