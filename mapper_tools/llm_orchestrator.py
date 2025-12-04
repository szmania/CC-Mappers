from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from mapper_tools import faction_xml_utils
from mapper_tools import unit_management
from mapper_tools import unit_selector
from mapper_tools import ck3_to_attila_mappings as mappings

def _build_llm_request_object(failure_data, unit_pool_for_request, screen_name_to_faction_key_map, faction_to_subculture_map, unit_to_class_map, unit_stats_map):
    """
    Builds a single, consolidated LLM request object for a given failure using a pre-calculated unit pool.
    """
    faction_name = failure_data['faction_element'].get('name', 'Unknown')
    tag_name = failure_data['tag_name']

    # Handle LevyComposition requests separately as they don't use a unit pool in the same way.
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

    # 2. Build prioritized candidate lists from the provided pool.
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

    for unit_key in sorted(list(unit_pool_for_request)): # Sort for determinism
        unit_class = unit_to_class_map.get(unit_key)
        if expected_attila_classes and unit_class in expected_attila_classes:
            primary_candidates.append(unit_key)
        else:
            stats = unit_stats_map.get(unit_key)
            score = unit_selector._calculate_quality_score(stats) if stats else 0
            secondary_candidates_with_scores.append((score, unit_key))

    secondary_candidates_with_scores.sort(key=lambda x: x[0], reverse=True)

    if not primary_candidates and not secondary_candidates_with_scores:
        print(f"      -> WARNING: No primary/secondary candidates identified for '{req_id}' in the provided pool. This should not happen if pool has units.")
        return None, None

    # 3. Build the final request object
    faction_key = screen_name_to_faction_key_map.get(faction_name)
    subculture = faction_to_subculture_map.get(faction_key)

    llm_request_obj = {
        'id': req_id,
        'faction': faction_name,
        'subculture': subculture,
        'validation_pool': sorted(list(unit_pool_for_request)), # Full pool for this tier for validation
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

    # 1. Group initial failures by faction name.
    failures_by_faction = defaultdict(list)
    for failure in all_llm_failures_to_process:
        faction_name = failure['faction_element'].get('name')
        if faction_name:
            failures_by_faction[faction_name].append(failure)

    # 2. Initialize an empty `accumulated_pools` dictionary
    accumulated_pools = defaultdict(set)

    # Determine max number of tiers from a sample faction. Default to 7 cultural tiers + 1 global thematic tier.
    num_cultural_tiers = 7
    if faction_pool_cache:
        sample_faction_name = next(iter(faction_pool_cache), None)
        if sample_faction_name:
            num_cultural_tiers = len(faction_pool_cache[sample_faction_name][0])

    total_passes = num_cultural_tiers + 1 # Add one for the global thematic pass

    # 3. Implement the new main loop that iterates through tiers
    for tier_level in range(total_passes):
        if not failures_by_faction:
            break

        is_global_pass = (tier_level == num_cultural_tiers)
        pass_name = f"Global Thematic Search Pass ({tier_level + 1}/{total_passes})" if is_global_pass else f"Cultural Tier Pass {tier_level + 1}/{total_passes}"

        # Count pending failures
        pending_failure_count = sum(len(v) for v in failures_by_faction.values())
        print(f"\n--- Running LLM Pass ({pass_name}) for {pending_failure_count} units ---")

        requests_for_this_tier = []
        element_map = {} # Maps req_id back to original failure data

        # Factions whose failures will be processed in this tier
        factions_to_process_in_this_tier = list(failures_by_faction.keys())

        # 4. ... then through factions with pending failures.
        for faction_name in factions_to_process_in_this_tier:
            pending_failures_for_faction = failures_by_faction[faction_name]

            if is_global_pass:
                # 5. Correctly integrate the global thematic search pass
                # For global pass, we generate requests for each failure individually with a special pool
                failures_processed_this_pass = []
                for failure_data in pending_failures_for_faction:
                    if failure_data['tag_name'] != 'MenAtArm':
                        continue # Global pass is only for MAA

                    global_candidates = unit_management.find_global_thematic_candidates(
                        failure_data, all_units, unit_to_class_map, unit_to_description_map, ck3_maa_definitions
                    )

                    if global_candidates:
                        # Exclude levy keys for this specific request
                        current_excluded_units = set(excluded_units_set) if excluded_units_set else set()
                        levy_keys_in_faction = {el.get('key') for el in failure_data['faction_element'].findall('Levies') if el.get('key')}
                        current_excluded_units.update(levy_keys_in_faction)

                        # Apply JSON exclusions
                        excluded_by_json = failure_data.get('excluded_by_json')
                        if excluded_by_json:
                            current_excluded_units.update(excluded_by_json)

                        final_global_pool = set(global_candidates) - current_excluded_units

                        if final_global_pool:
                            req_id, llm_request_obj = _build_llm_request_object(
                                failure_data, final_global_pool, screen_name_to_faction_key_map,
                                faction_to_subculture_map, unit_to_class_map, unit_stats_map
                            )
                            if req_id and llm_request_obj:
                                print(f"  -> Found {len(final_global_pool)} global thematic candidates for '{failure_data['maa_definition_name']}' in '{faction_name}'. Queuing for LLM.")
                                requests_for_this_tier.append(llm_request_obj)
                                element_map[req_id] = failure_data
                                failures_processed_this_pass.append(failure_data)

                # Remove processed failures from the faction's pending list
                failures_by_faction[faction_name] = [f for f in pending_failures_for_faction if f not in failures_processed_this_pass]
                if not failures_by_faction[faction_name]:
                    del failures_by_faction[faction_name]

            else: # Cultural pass
                # 6. Inside the loop, add the logic to progressively build the `accumulated_pool`
                # The `tiered_pools` are unfiltered by global exclusions.
                tiered_pools, _ = faction_pool_cache.get(faction_name, ([], []))
                if tier_level < len(tiered_pools):
                    accumulated_pools[faction_name].update(tiered_pools[tier_level])

                # Apply global exclusions to the accumulated pool
                current_accumulated_pool = accumulated_pools[faction_name]
                if excluded_units_set:
                    current_accumulated_pool = current_accumulated_pool - excluded_units_set

                # Check if pool is large enough to proceed
                if len(current_accumulated_pool) >= 5: # min_pool_size
                    # 7. Generate requests for all of a faction's pending failures
                    print(f"  -> Faction '{faction_name}': Pool ready at Tier {tier_level + 1} with {len(current_accumulated_pool)} units. Generating {len(pending_failures_for_faction)} requests.")

                    for failure_data in pending_failures_for_faction:
                        # Apply request-specific exclusions (JSON, levy keys)
                        final_pool_for_request = set(current_accumulated_pool)

                        excluded_by_json = failure_data.get('excluded_by_json')
                        if excluded_by_json:
                            final_pool_for_request.difference_update(excluded_by_json)

                        if failure_data['tag_name'] == 'MenAtArm':
                            levy_keys_in_faction = {el.get('key') for el in failure_data['faction_element'].findall('Levies') if el.get('key')}
                            final_pool_for_request.difference_update(levy_keys_in_faction)

                        if not final_pool_for_request:
                            continue # Skip if exclusions emptied the pool for this specific request

                        req_id, llm_request_obj = _build_llm_request_object(
                            failure_data, final_pool_for_request, screen_name_to_faction_key_map,
                            faction_to_subculture_map, unit_to_class_map, unit_stats_map
                        )
                        if req_id and llm_request_obj:
                            requests_for_this_tier.append(llm_request_obj)
                            element_map[req_id] = failure_data

                    # All failures for this faction have been processed for this tier, so clear them.
                    del failures_by_faction[faction_name]

        if not requests_for_this_tier:
            print("No requests with sufficient pool size for this tier. Moving to next tier.")
            continue

        # 8. Adapt the LLM batching, execution, and result-handling logic
        cached_results, uncached_requests, cache_modified = llm_helper.filter_requests_against_cache(requests_for_this_tier)
        if cache_modified:
            llm_helper.save_cache()

        print(f"  -> Found {len(cached_results)} valid cached results. {len(uncached_requests)} requests require LLM call for this tier.")

        network_llm_results = {}
        if uncached_requests and llm_helper.network_calls_enabled:
            all_batches = [uncached_requests[i:i + llm_batch_size] for i in range(0, len(uncached_requests), llm_batch_size)]
            print(f"  -> Submitting {len(uncached_requests)} requests to LLM in {len(all_batches)} batches using {llm_threads} threads...")

            with ThreadPoolExecutor(max_workers=llm_threads) as executor:
                future_to_batch = {
                    executor.submit(llm_helper.get_batch_unit_replacements, batch, time_period_context): batch
                    for batch in all_batches
                }
                processed_requests = 0
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    processed_requests += len(batch)
                    print(f"  -> LLM progress: {processed_requests}/{len(uncached_requests)} requests completed for this tier.")
                    try:
                        batch_results = future.result()
                        if batch_results:
                            network_llm_results.update(batch_results)
                    except Exception as exc:
                        print(f"  -> ERROR: A batch of {len(batch)} requests generated an exception: {exc}")

        final_llm_results = {**cached_results, **network_llm_results}

        if final_llm_results:
            print(f"\nApplying {len(final_llm_results)} suggestions for this tier...")
            processed_req_ids = set()

            for req_id, suggestion in final_llm_results.items():
                processed_req_ids.add(req_id)
                req_data = element_map.get(req_id)
                if not req_data:
                    print(f"  -> WARNING: Could not find original request data for suggestion '{req_id}'.")
                    continue

                chosen_unit = suggestion.get("chosen_unit")
                chosen_composition = suggestion.get("chosen_composition")

                if chosen_unit or chosen_composition:
                    # Apply the result
                    if chosen_unit:
                        req_data['element'].set('key', chosen_unit)
                        print(f"    -> SUCCESS: Replaced unit for '{req_id}' with '{chosen_unit}'.")
                    elif chosen_composition:
                        # For LevyComposition, we don't set a key. The result is used in the low-confidence pass.
                        print(f"    -> SUCCESS: Received levy composition for '{req_id}'. It will be applied in the next pass.")
                    llm_replacements_made += 1
                else: # A None result is a failure for this tier, add it back to be retried
                    faction_name = req_data['faction_element'].get('name')
                    failures_by_faction[faction_name].append(req_data)

            # Add back any requests that failed due to network errors etc.
            for req in requests_for_this_tier:
                if req['id'] not in processed_req_ids:
                    req_data = element_map[req['id']]
                    faction_name = req_data['faction_element'].get('name')
                    failures_by_faction[faction_name].append(req_data)

    # After all tiers, any remaining failures in failures_by_faction are the final failures.
    final_failures = []
    for faction_failures in failures_by_faction.values():
        final_failures.extend(faction_failures)

    print(f"\nTotal LLM replacements made across all tiers: {llm_replacements_made}.")
    return llm_replacements_made, final_failures

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

    # 1. Inject Temporary Unique IDs
    temp_id_counter = 0
    tags_to_review = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']
    for faction in root.findall('Faction'):
        for child in faction:
            if child.tag in tags_to_review:
                child.set('__review_id__', str(temp_id_counter))
                temp_id_counter += 1

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
            if child.tag in tags_to_review:
                unit_key = child.get('key')
                review_id = child.get('__review_id__') # Get the temporary ID
                if not unit_key or not review_id:
                    continue

                # Simplify the LLM Request Identifier
                identifier = {'__review_id__': review_id}

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
            # Validate correction object structure
            tag_to_find = correction.get('tag')
            identifier = correction.get('identifier')
            current_unit_from_llm = correction.get('current_unit')
            suggested_unit = correction.get('suggested_unit')

            if not (tag_to_find and current_unit_from_llm and suggested_unit and isinstance(identifier, dict)):
                print(f"    -> WARNING: Invalid correction object received from LLM. Skipping. Data: {correction}")
                continue

            # Replace the Element Matching Logic
            review_id = identifier.get('__review_id__')
            if not review_id:
                print(f"    -> WARNING: Correction object missing '__review_id__'. Skipping. Data: {correction}")
                continue

            element_to_modify = faction_element.find(f".//{tag_to_find}[@__review_id__='{review_id}']")

            if element_to_modify is not None:
                original_key = element_to_modify.get('key')
                if original_key != current_unit_from_llm:
                    print(f"    -> WARNING: Element for ID '{review_id}' in '{faction_name}' has changed key (expected '{current_unit_from_llm}', found '{original_key}'). Applying correction anyway.")
                
                element_to_modify.set('key', suggested_unit)
                total_corrections_applied += 1
                print(f"    - Changed <{tag_to_find}> (ID: {review_id}): '{original_key}' -> '{suggested_unit}'. Reason: {correction.get('reason', 'N/A')}")
            else:
                print(f"    -> WARNING: Could not find matching element for correction in '{faction_name}': Tag={tag_to_find}, ID={review_id}, current_unit='{current_unit_from_llm}'")

    # 4. Clean Up Temporary IDs
    print("\nCleaning up temporary '__review_id__' attributes...")
    for elem in root.findall(".//*[@__review_id__]"):
        del elem.attrib['__review_id__']
    print("Temporary '__review_id__' attributes removed.")

    return total_corrections_applied
