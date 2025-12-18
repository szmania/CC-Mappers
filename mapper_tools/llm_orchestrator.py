from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET # Added import for ElementTree

# --- NEW: Constants for LLM processing ---
MAX_LLM_FAILURES_THRESHOLD = 500000 # Threshold for early exit

from mapper_tools import faction_xml_utils
from mapper_tools import unit_management
from mapper_tools import unit_selector
from mapper_tools import ck3_to_attila_mappings as mappings
from mapper_tools import shared_utils
import re

def _build_llm_request_object(failure_data, tiered_pools, tiered_log_strings, screen_name_to_faction_key_map, faction_to_subculture_map, unit_to_class_map, unit_stats_map):
    """
    Builds a single, consolidated LLM request object for a given failure using pre-calculated tiered unit pools.
    """
    faction_name = failure_data['faction_element'].get('name', 'Unknown')
    tag_name = failure_data['tag_name']

    # Handle LevyComposition requests separately as they don't use a unit pool in the same way.
    if tag_name == 'LevyComposition':
        req_id = f"LevyComposition|{faction_name}|tier_{failure_data['tier'] if failure_data['tier'] is not None else 'any'}"
        llm_request_obj = {
            'id': req_id,
            'faction': faction_name,
            'subculture': faction_to_subculture_map.get(screen_name_to_faction_key_map.get(faction_name)),
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

    # 2. Build prioritized candidate lists from the provided tiered pools.
    validation_pool = set()
    for pool in tiered_pools:
        validation_pool.update(pool)

    prioritized_candidates = {}
    # Use a set to track units already added to avoid duplicates in the prompt
    # while preserving tier information.
    added_units = set()
    for i, pool in enumerate(tiered_pools):
        # Extract a clean tier name like "Tier 1: Faction-specific"
        tier_name_full = tiered_log_strings[i]
        tier_name_match = re.search(r'\((.*?)\)', tier_name_full)
        tier_name = tier_name_match.group(1) if tier_name_match else f"Tier {i+1}"

        # Get units in this tier that haven't been seen in a higher-priority tier
        unique_units_in_tier = sorted(list(pool - added_units))

        if unique_units_in_tier:
            prioritized_candidates[tier_name] = unique_units_in_tier
            added_units.update(unique_units_in_tier)

    if not prioritized_candidates:
        print(f"      -> WARNING: No candidate units identified for '{req_id}' from tiered pools. This should not happen if pools have units.")
        return None, None

    # 3. Build the final request object
    faction_key = screen_name_to_faction_key_map.get(faction_name)
    subculture = faction_to_subculture_map.get(faction_key)

    llm_request_obj = {
        'id': req_id,
        'faction': faction_name,
        'subculture': subculture,
        'validation_pool': sorted(list(validation_pool)),
        'prioritized_candidates': prioritized_candidates, # NEW
        'tier': failure_data['tier'],
        'faction_element': failure_data['faction_element'], # Keep reference to parent faction element
        'tag_name': tag_name, # Keep tag_name for element creation
    }

    if tag_name == 'MenAtArm':
        llm_request_obj['maa_type'] = failure_data['maa_definition_name']
        expected_attila_classes = mappings.CK3_TYPE_TO_ATTILA_CLASS.get(failure_data['maa_definition_name']) or \
                                  (mappings.CK3_TYPE_TO_ATTILA_CLASS.get(failure_data.get('internal_type')) if failure_data.get('internal_type') else None)
        llm_request_obj['expected_attila_classes'] = expected_attila_classes
        if 'element' in failure_data: # Only add if it exists
            llm_request_obj['element'] = failure_data['element']
    else:
        llm_request_obj.update({
            'unit_role_description': failure_data['unit_role_description'],
            'rank': failure_data.get('rank'),
            'level': failure_data.get('level'),
            'garrison_slot': failure_data.get('garrison_slot'),
            'levy_slot': failure_data.get('levy_slot')
        })
        if 'element' in failure_data: # Only add if it exists
            llm_request_obj['element'] = failure_data['element']

    return req_id, llm_request_obj

def run_llm_unit_assignment_pass(llm_helper, all_llm_failures_to_process, time_period_context, llm_threads, llm_batch_size,
                             faction_pool_cache, all_units, excluded_units_set, unit_to_tier_map, unit_to_class_map,
                             unit_to_description_map, unit_stats_map, screen_name_to_faction_key_map, faction_key_to_units_map,
                             faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                             culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, ck3_maa_definitions):
    """
    Runs the LLM pass in a single, consolidated batch.
    It generates the full tiered unit pool for each failure upfront and sends one request per failure.
    """
    if not llm_helper or not all_llm_failures_to_process:
        log_msg = "LLM integration is disabled or no units required LLM intervention."
        print(f"\n{log_msg}")
        return 0, all_llm_failures_to_process

    # --- NEW: Early exit if too many failures ---
    if len(all_llm_failures_to_process) > MAX_LLM_FAILURES_THRESHOLD:
        print(f"\n--- WARNING: Skipping LLM Unit Assignment Pass ---")
        print(f"Number of LLM requests ({len(all_llm_failures_to_process)}) exceeds threshold of {MAX_LLM_FAILURES_THRESHOLD}.")
        print("This usually indicates a problem with the input data (e.g., TSV files) or configuration,")
        print("causing the high-confidence pass to fail for most units.")
        print("Skipping LLM pass to prevent hanging. Proceeding directly to low-confidence procedural fallback.")

        # Log detailed information about the failures to help diagnose
        failure_types = defaultdict(int)
        for failure in all_llm_failures_to_process:
            tag_name = failure.get('tag_name', 'unknown')
            failure_types[tag_name] += 1

        print(f"Failure breakdown by type: {dict(failure_types)}")

        return 0, all_llm_failures_to_process

    log_msg = "cache and/or LLM" if llm_helper.network_calls_enabled else "LLM cache"
    print(f"\n--- Running Single-Pass LLM Unit Assignment for {len(all_llm_failures_to_process)} units using {log_msg} ---")

    llm_replacements_made = 0
    final_failures = []
    unit_requests = []
    levy_requests = []
    element_map = {} # Maps req_id back to original failure data

    # 1. Group failures by faction to get tiered pools efficiently.
    failures_by_faction = defaultdict(list)
    for failure in all_llm_failures_to_process:
        faction_name = failure['faction_element'].get('name')
        if faction_name:
            failures_by_faction[faction_name].append(failure)

    # 2. Generate requests for each failure.
    for faction_name, failures in failures_by_faction.items():
        # Get the complete, unfiltered tiered pools and log strings for this faction ONCE.
        # The cache ensures this is a cheap operation after the first call.
        unfiltered_tiered_pools, log_strings = faction_pool_cache.get(faction_name, ([], []))

        for failure_data in failures:
            # Apply global and request-specific exclusions to the tiered pools
            current_excluded_units = set(excluded_units_set) if excluded_units_set else set()
            excluded_by_json = failure_data.get('excluded_by_json')
            if excluded_by_json:
                current_excluded_units.update(excluded_by_json)
            if failure_data['tag_name'] == 'MenAtArm':
                levy_keys_in_faction = {el.get('key') for el in failure_data['faction_element'].findall('Levies') if el.get('key')}
                current_excluded_units.update(levy_keys_in_faction)

            filtered_tiered_pools = [pool - current_excluded_units for pool in unfiltered_tiered_pools]

            # For LevyComposition, we need to recalculate available categories based on the full pool
            if failure_data['tag_name'] == 'LevyComposition':
                full_pool_for_faction = set()
                for pool in filtered_tiered_pools:
                    full_pool_for_faction.update(pool)

                LEVY_CATEGORY_TO_CLASSES = {
                    'spear': ['inf_spear'], 'infantry': ['inf_melee', 'inf_heavy'],
                    'missile': ['inf_bow', 'inf_sling', 'inf_javelin'], 'cavalry': ['cav_melee', 'cav_heavy', 'cav_shock'],
                    'missile_cavalry': ['cav_missile']
                }
                available_levy_categories = set()
                for unit_key in full_pool_for_faction:
                    unit_class = unit_to_class_map.get(unit_key)
                    for category, classes in LEVY_CATEGORY_TO_CLASSES.items():
                        if unit_class in classes:
                            available_levy_categories.add(category)
                failure_data['available_levy_categories'] = sorted(list(available_levy_categories))

            # Build the request object
            req_id, llm_request_obj = _build_llm_request_object(
                failure_data, filtered_tiered_pools, log_strings, screen_name_to_faction_key_map,
                faction_to_subculture_map, unit_to_class_map, unit_stats_map
            )

            if req_id and llm_request_obj:
                if llm_request_obj['tag_name'] == 'LevyComposition':
                    levy_requests.append(llm_request_obj)
                else:
                    unit_requests.append(llm_request_obj)
                element_map[req_id] = failure_data

    # 3. Process all generated requests
    # --- Unit Request Pipeline ---
    cached_unit_results, uncached_unit_requests, unit_cache_modified = llm_helper.filter_requests_against_cache(unit_requests)
    if unit_cache_modified:
        llm_helper.save_cache()
    print(f"  -> Unit Requests: Found {len(cached_unit_results)} valid cached results. {len(uncached_unit_requests)} requests require LLM call.")

    network_unit_results = {}
    if uncached_unit_requests and llm_helper.network_calls_enabled:
        # Enhanced deduplication: Create a more comprehensive unique key
        unique_requests = {}
        for req in uncached_unit_requests:
            # Include more fields in the deduplication key to catch more duplicates
            req_key = (
                req.get('faction'),
                req.get('tag_name'),
                req.get('maa_type'),
                req.get('rank'),
                req.get('level'),
                req.get('garrison_slot'),
                req.get('levy_slot'),
                tuple(sorted(req.get('validation_pool', []))),
                tuple(sorted(req.get('prioritized_candidates', {}).keys())) if req.get('prioritized_candidates') else ()
            )
            if req_key not in unique_requests:
                unique_requests[req_key] = req

        deduplicated_requests = list(unique_requests.values())
        if len(deduplicated_requests) < len(uncached_unit_requests):
            print(f"  -> Deduplication reduced unit requests from {len(uncached_unit_requests)} to {len(deduplicated_requests)}")

        # Smart batching: Group by multiple criteria for better context
        # Group by faction culture first, then by request type
        requests_by_culture_and_type = defaultdict(list)
        for req in deduplicated_requests:
            faction_name = req.get('faction', '')
            faction_key = screen_name_to_faction_key_map.get(faction_name)
            subculture = faction_to_subculture_map.get(faction_key) if faction_key else None
            request_type = req.get('tag_name', 'unknown')

            # Create a grouping key that combines culture and type
            group_key = f"{subculture or 'no_culture'}|{request_type}"
            requests_by_culture_and_type[group_key].append(req)

        # Process batches with optimal grouping
        network_unit_results = {}
        for group_key, group_requests in requests_by_culture_and_type.items():
            subculture, request_type = group_key.split('|', 1)
            # Use larger batch size (up to 200) for better efficiency
            batch_size = llm_batch_size
            group_batches = [group_requests[i:i + batch_size] for i in range(0, len(group_requests), batch_size)]
            print(f"  -> Submitting {len(group_requests)} '{request_type}' requests for subculture '{subculture}' to LLM in {len(group_batches)} batches (batch size: {batch_size})...")

            with ThreadPoolExecutor(max_workers=llm_threads) as executor:
                future_to_batch = {
                    executor.submit(llm_helper.get_batch_unit_replacements, batch, time_period_context): batch
                    for batch in group_batches
                }
                processed_requests = 0
                for future in as_completed(future_to_batch):
                    processed_requests += len(future_to_batch[future])
                    print(f"  -> LLM '{request_type}' progress for subculture '{subculture}': {processed_requests}/{len(group_requests)} requests completed.")
                    try:
                        batch_results = future.result()
                        if batch_results:
                            network_unit_results.update(batch_results)
                    except Exception as exc:
                        print(f"  -> ERROR: A '{request_type}' batch for subculture '{subculture}' generated an exception: {exc}")
    final_unit_results = {**cached_unit_results, **network_unit_results}

    # --- Levy Request Pipeline ---
    cached_levy_results, uncached_levy_requests, levy_cache_modified = llm_helper.filter_requests_against_cache(levy_requests)
    if levy_cache_modified:
        llm_helper.save_cache()
    print(f"  -> Levy Requests: Found {len(cached_levy_results)} valid cached results. {len(uncached_levy_requests)} requests require LLM call.")

    network_levy_results = {}
    if uncached_levy_requests and llm_helper.network_calls_enabled:
        # Remove duplicate levy requests
        unique_levy_requests = {}
        for req in uncached_levy_requests:
            # Create a unique key based on the request content to deduplicate
            req_key = (req.get('faction'), req.get('tier'), tuple(req.get('available_levy_categories', [])))
            if req_key not in unique_levy_requests:
                unique_levy_requests[req_key] = req

        deduplicated_levy_requests = list(unique_levy_requests.values())
        if len(deduplicated_levy_requests) < len(uncached_levy_requests):
            print(f"  -> Deduplication reduced levy requests from {len(uncached_levy_requests)} to {len(deduplicated_levy_requests)}")

        # Group levy requests by faction for more efficient processing
        levy_requests_by_faction = defaultdict(list)
        for req in deduplicated_levy_requests:
            levy_requests_by_faction[req['faction']].append(req)

        # Process batches by faction for better LLM efficiency
        network_levy_results = {}
        for faction_name, faction_requests in levy_requests_by_faction.items():
            # Use larger batch size for better efficiency
            faction_batches = [faction_requests[i:i + llm_batch_size] for i in range(0, len(faction_requests), llm_batch_size)]
            print(f"  -> Submitting {len(faction_requests)} levy requests for faction '{faction_name}' to LLM in {len(faction_batches)} batches using {llm_threads} threads...")
            with ThreadPoolExecutor(max_workers=llm_threads) as executor:
                future_to_batch = {executor.submit(llm_helper.get_batch_levy_compositions, batch, time_period_context): batch for batch in faction_batches}
                processed_requests = 0
                for future in as_completed(future_to_batch):
                    processed_requests += len(future_to_batch[future])
                    print(f"  -> LLM levy progress for '{faction_name}': {processed_requests}/{len(faction_requests)} requests completed.")
                    try:
                        network_levy_results.update(future.result())
                    except Exception as exc:
                        print(f"  -> ERROR: A levy batch for '{faction_name}' generated an exception: {exc}")
    final_levy_results = {**cached_levy_results, **network_levy_results}

    # 4. Apply results and collect final failures
    all_results = {**final_unit_results, **final_levy_results}
    all_requests = unit_requests + levy_requests
    processed_req_ids = set(all_results.keys())

    for req_id, suggestion in all_results.items():
        req_data = element_map.get(req_id)
        if not req_data:
            print(f"  -> WARNING: Could not find original request data for suggestion '{req_id}'.")
            continue

        chosen_unit = suggestion.get("chosen_unit")
        chosen_composition = suggestion.get("chosen_composition")

        # Filter LLM results against excluded units
        if chosen_unit and excluded_units_set and chosen_unit in excluded_units_set:
            print(f"    -> WARNING: LLM suggested excluded unit '{chosen_unit}'. Skipping and treating as failure.")
            final_failures.append(req_data)  # Add back to failures
            continue

        if chosen_unit:
            element_to_modify = req_data.get('element')
            if element_to_modify is not None:
                element_to_modify.set('key', chosen_unit)
                print(f"    -> SUCCESS: Replaced unit for '{req_id}' with '{chosen_unit}'.")
            else:
                faction_element = req_data['faction_element']
                tag_name = req_data['tag_name']
                attrs = {'key': chosen_unit}
                if req_data.get('rank'): attrs['rank'] = str(req_data['rank'])

                if tag_name == 'Garrison':
                    if 'level' in req_data:
                        attrs['level'] = str(req_data['level'])
                    else:
                        attrs['level'] = '1' # Schema requires level, default to 1 if missing
                    attrs['percentage'] = '0' # Default percentage
                    attrs['max'] = 'LEVY' # Schema requires max
                elif tag_name == 'Levies':
                    attrs['percentage'] = '100' # Default percentage, will be normalized later
                    attrs['max'] = 'LEVY' # Schema requires max

                if tag_name == 'MenAtArm' and req_data.get('maa_type'):
                    attrs['type'] = req_data['maa_type']

                ET.SubElement(faction_element, tag_name, attrs)
                print(f"    -> SUCCESS: Created missing <{tag_name}> tag for '{req_id}' and set key to '{chosen_unit}'.")
            llm_replacements_made += 1
        elif chosen_composition:
            # Attach the composition to the original failure object for the low-confidence pass to use.
            req_data['levy_composition_override'] = chosen_composition
            final_failures.append(req_data) # Add back to failures so it can be processed by the next stage
            print(f"    -> SUCCESS: Received levy composition for '{req_id}'. It will be applied in the next pass.")
            llm_replacements_made += 1
        else:
            # LLM returned null, this is a final failure.
            final_failures.append(req_data)

    # Add back any requests that failed due to network errors etc.
    for req in all_requests:
        if req['id'] not in processed_req_ids:
            final_failures.append(element_map[req['id']])

    print(f"\nTotal LLM replacements made: {llm_replacements_made}.")
    return llm_replacements_made, final_failures

def run_llm_roster_review_pass(root, llm_helper, time_period_context, llm_threads, llm_batch_size,
                                faction_pool_cache, all_units, excluded_units_set,
                                screen_name_to_faction_key_map, faction_key_to_units_map,
                                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                                culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map,
                                faction_to_heritages_map, ck3_maa_definitions, all_faction_elements=None): # Added all_faction_elements
    """
    Runs the LLM Roster Review pass. It gathers the current roster for each faction,
    sends it to the LLM for thematic and cultural review, and applies the suggested corrections.
    Includes a retry mechanism for failed requests.
    """
    if not llm_helper or not llm_helper.network_calls_enabled:
        print("ERROR: LLM Roster Review requires network calls to be enabled.")
        return 0

    print("\n--- Starting LLM Roster Review Pass ---")

    MAX_REVIEW_RETRIES = 3
    total_corrections_applied = 0

    # Determine which list of factions to iterate over
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    # 1. Inject Temporary Unique IDs
    temp_id_counter = 0
    tags_to_review = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']
    for faction in factions_to_iterate:
        for child in faction:
            if child.tag in tags_to_review:
                child.set('__review_id__', str(temp_id_counter))
                temp_id_counter += 1

    initial_review_requests = []
    faction_element_map = {} # Maps faction name to its XML element

    for faction in factions_to_iterate:
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        faction_element_map[faction_name] = faction

        # 1. Get the local unit pool for this faction
        local_unit_pool, _, unfiltered_tiered_pools = faction_xml_utils.get_cached_faction_working_pool(
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
                if not review_id:
                    continue

                # Simplify the LLM Request Identifier
                identifier = {'__review_id__': review_id}

                roster[child.tag].append({
                    "current_unit": unit_key, # This will be None (json: null) if key was removed
                    "identifier": identifier
                })

        if not roster:
            print(f"  -> Skipping review for faction '{faction_name}' as it has no units.")
            continue

        # 3. Prepare the request
        req_id = f"review|{faction_name}"
        faction_key = screen_name_to_faction_key_map.get(faction_name)
        subculture = faction_to_subculture_map.get(faction_key)

        initial_review_requests.append({
            'id': req_id,
            'faction': faction_name,
            'subculture': subculture,
            'roster': roster,
            'local_unit_pool': local_unit_pool,
            'tiered_pools': unfiltered_tiered_pools
        })

    if not initial_review_requests:
        print("No factions found to review.")
        return 0

    # Check threshold for roster review as well
    if len(initial_review_requests) > MAX_LLM_FAILURES_THRESHOLD:
        print(f"\n--- WARNING: Skipping LLM Roster Review Pass ---")
        print(f"Number of review requests ({len(initial_review_requests)}) exceeds threshold of {MAX_LLM_FAILURES_THRESHOLD}.")
        print("This indicates a potential configuration issue or extremely large mod.")
        return 0

    requests_to_process = list(initial_review_requests)

    for attempt in range(MAX_REVIEW_RETRIES):
        if not requests_to_process:
            print(f"\nAll roster review requests processed successfully after {attempt} attempts.")
            break

        print(f"\n--- Running LLM Roster Review Attempt {attempt + 1}/{MAX_REVIEW_RETRIES} for {len(requests_to_process)} requests ---")

        # 4. Filter against cache
        cached_results, uncached_requests, cache_modified = llm_helper.filter_requests_against_cache(requests_to_process)
        if cache_modified:
            llm_helper.save_cache()

        print(f"  -> Found {len(cached_results)} valid cached results. {len(uncached_requests)} requests require LLM call.")

        network_llm_results = {}
        if uncached_requests:
            # Create batches from all uncached requests for true parallel processing
            # Dynamically adjust batch size to better utilize threads, while respecting the max batch size.
            num_requests = len(uncached_requests)
            # We want at least as many batches as threads, if possible, without making batches too small.
            # A batch size of 1 is the minimum.
            # Calculate a batch size that would create roughly `llm_threads` batches.
            ideal_batch_size = (num_requests + llm_threads - 1) // llm_threads if llm_threads > 0 else num_requests

            # Use the smaller of the user's configured max batch size and our ideal size.
            # But don't go below 1.
            effective_batch_size = max(1, min(llm_batch_size, ideal_batch_size))

            all_batches = [uncached_requests[i:i + effective_batch_size] for i in range(0, len(uncached_requests), effective_batch_size)]
            print(f"  -> Submitting {len(uncached_requests)} total review requests to LLM in {len(all_batches)} batches using {llm_threads} threads (effective batch size: {effective_batch_size})...")

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
                        # It's hard to know which faction failed here, but we can log the exception
                        print(f"  -> ERROR: A review batch generated an exception: {exc}")

        # 5. Apply corrections and collect failures for next attempt
        final_results = {**cached_results, **network_llm_results}
        requests_for_next_attempt = []
        processed_req_ids_this_attempt = set()

        for req_id, result_data in final_results.items():
            processed_req_ids_this_attempt.add(req_id)
            faction_name = req_id.split('|')[1]
            faction_element = faction_element_map.get(faction_name)
            corrections = result_data.get("corrections", [])

            if not faction_element:
                print(f"  -> WARNING: Could not find faction element for '{faction_name}'. Skipping result for '{req_id}'.")
                continue

            if result_data['status'] == 'success':
                if corrections:
                    print(f"  -> Applying {len(corrections)} corrections for faction '{faction_name}' (Attempt {attempt + 1})...")
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
                        # Sanitize review_id to handle potential LLM malformations (e.g., adding quotes).
                        # It should be a string of digits.
                        original_review_id = str(review_id) # Ensure it's a string for processing
                        sanitized_review_id = re.sub(r'\D', '', original_review_id)
                        if sanitized_review_id != original_review_id:
                            print(f"    -> WARNING: Sanitized malformed '__review_id__' from LLM. Original: '{original_review_id}', Used: '{sanitized_review_id}'.")

                        if not sanitized_review_id:
                            print(f"    -> WARNING: Sanitized '__review_id__' is empty. Skipping. Original: '{original_review_id}'")
                            continue

                        review_id = sanitized_review_id

                        # Sanitize and validate the tag from the LLM to prevent XPath errors.
                        valid_tags = ['General', 'Knights', 'Levies', 'Garrison', 'MenAtArm']
                        sanitized_tag = shared_utils.find_best_fuzzy_match(tag_to_find, valid_tags, threshold=0.7)

                        if not sanitized_tag:
                            print(f"    -> WARNING: LLM returned an invalid or unrecognizable tag '{tag_to_find}'. Skipping correction.")
                            continue

                        if sanitized_tag != tag_to_find:
                                print(f"    -> WARNING: LLM returned a potentially malformed tag '{tag_to_find}'. Sanitized to '{sanitized_tag}' via fuzzy matching.")

                        # --- NEW: Uniqueness Validation ---
                        all_current_keys_in_faction = {
                            el.get('key') for el in faction_element if el.get('key')
                        }
                        # We must temporarily exclude the key of the very unit we are about to change
                        # to allow for valid self-replacement or swapping with another unit that will also be changed.
                        # First, find the element that *would* be changed.
                        element_being_changed = faction_element.find(f".//{sanitized_tag}[@__review_id__='{review_id}']")
                        if element_being_changed is not None and element_being_changed.get('key') in all_current_keys_in_faction:
                            all_current_keys_in_faction.remove(element_being_changed.get('key'))

                        if suggested_unit in all_current_keys_in_faction:
                            print(f"    -> WARNING: LLM suggested unit '{suggested_unit}' for faction '{faction_name}', but this unit is already in use by another tag in this faction. Skipping correction to prevent duplicates.")
                            continue # Skip this correction
                        # --- END NEW ---

                        element_to_modify = faction_element.find(f".//{sanitized_tag}[@__review_id__='{review_id}']")

                        if element_to_modify is not None:
                            original_key = element_to_modify.get('key')
                            if original_key != current_unit_from_llm:
                                print(f"    -> WARNING: Element for ID '{review_id}' in '{faction_name}' has changed key (expected '{current_unit_from_llm}', found '{original_key}'). Applying correction anyway.")

                            element_to_modify.set('key', suggested_unit)
                            total_corrections_applied += 1
                            print(f"    - Changed <{tag_to_find}> (ID: {review_id}): '{original_key}' -> '{suggested_unit}'. Reason: {correction.get('reason', 'N/A')}")
                        else:
                            print(f"    -> WARNING: Could not find matching element for correction in '{faction_name}': Tag={tag_to_find}, ID={review_id}, current_unit='{current_unit_from_llm}'")
                else:
                    print(f"  -> Faction '{faction_name}' reviewed successfully, no corrections suggested (Attempt {attempt + 1}).")
            else: # status == 'failure'
                print(f"  -> Faction '{faction_name}' review FAILED (Attempt {attempt + 1}): {result_data.get('reason', 'No reason provided')}. Retrying...")
                # Find the original request object to add to requests_for_next_attempt
                original_request = next((req for req in requests_to_process if req['id'] == req_id), None)
                if original_request:
                    # NEW: Attach detailed invalid suggestions for the next retry attempt
                    if result_data.get('invalid_suggestions'):
                        original_request['retry_context'] = {'previous_errors': result_data['invalid_suggestions']}
                    requests_for_next_attempt.append(original_request)
                else:
                    print(f"    -> WARNING: Original request for failed ID '{req_id}' not found in current batch. Cannot retry.")

        # Identify requests that were in requests_to_process but did not appear in final_results (e.g., network errors)
        for req in requests_to_process:
            if req['id'] not in processed_req_ids_this_attempt:
                print(f"  -> Faction '{req['faction']}' review FAILED at network level (Attempt {attempt + 1}). Retrying...")
                requests_for_next_attempt.append(req)

        requests_to_process = requests_for_next_attempt

        if requests_to_process and attempt == MAX_REVIEW_RETRIES - 1:
            print(f"\n--- WARNING: {len(requests_to_process)} roster review requests still failed after {MAX_REVIEW_RETRIES} attempts. ---")
            for req in requests_to_process:
                print(f"  - Faction '{req['faction']}' (ID: {req['id']})")

    # 6. Clean Up Temporary IDs
    print("\nCleaning up temporary '__review_id__' attributes...")
    for elem in root.findall(".//*[@__review_id__]"):
        del elem.attrib['__review_id__']
    print("Temporary '__review_id__' attributes removed.")

    return total_corrections_applied

# NEW: LLM Subculture Assignment Pass
def run_llm_subculture_pass(root, llm_helper, time_period_context, llm_threads, llm_batch_size,
                            faction_to_subculture_map, subculture_to_factions_map, screen_name_to_faction_key_map,
                            all_faction_elements=None): # Added all_faction_elements
    """
    Identifies factions missing subcultures, queries the LLM for assignments, and applies them.
    """
    if not llm_helper or not llm_helper.network_calls_enabled:
        print("\n--- Skipping LLM Subculture Assignment Pass ---")
        print("LLM Subculture Assignment requires network calls to be enabled.")
        return 0

    print("\n--- Starting LLM Subculture Assignment Pass ---")

    subcultures_assigned_count = 0
    subculture_requests = []
    faction_elements_to_process = {} # Map faction_name to XML element

    # Get all available subcultures from the existing map
    all_available_subcultures = sorted(list(subculture_to_factions_map.keys()))
    if not all_available_subcultures:
        print("  -> WARNING: No available subcultures found in the Attila DB. LLM cannot assign subcultures.")
        return 0

    # Determine which list of factions to iterate over
    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    # 1. Identify factions missing subcultures
    for faction_element in factions_to_iterate:
        faction_name = faction_element.get('name')
        if not faction_name or faction_name == "Default":
            continue

        faction_key = screen_name_to_faction_key_map.get(faction_name)
        if not faction_key:
            # This faction name might be a fuzzy match, or not in the DB at all.
            # For now, we only process factions that have a known key.
            continue

        # A subculture request is needed if the faction does not have a subculture defined in the Attila DB.
        # We intentionally ignore any 'subculture' attribute that may already exist in the XML,
        # as the DB and LLM cache are the sources of truth.
        if faction_key not in faction_to_subculture_map:
            req_id = f"subculture|{faction_name}"
            subculture_requests.append({
                'id': req_id,
                'faction': faction_name,
                'available_subcultures': all_available_subcultures,
                'validation_pool': all_available_subcultures # For cache validation
            })
            faction_elements_to_process[req_id] = faction_element

    if not subculture_requests:
        print("  -> No factions found missing subcultures. Skipping LLM assignment.")
        return 0

    print(f"  -> Found {len(subculture_requests)} factions missing subcultures. Attempting LLM assignment.")

    # 2. Filter requests against cache
    cached_results, uncached_requests, cache_modified = llm_helper.filter_requests_against_cache(subculture_requests)
    if cache_modified:
        llm_helper.save_cache()

    print(f"  -> Found {len(cached_results)} valid cached results. {len(uncached_requests)} requests require LLM call.")

    network_llm_results = {}
    if uncached_requests:
        all_batches = [uncached_requests[i:i + llm_batch_size] for i in range(0, len(uncached_requests), llm_batch_size)]
        print(f"  -> Submitting {len(uncached_requests)} subculture requests to LLM in {len(all_batches)} batches using {llm_threads} threads...")

        with ThreadPoolExecutor(max_workers=llm_threads) as executor:
            future_to_batch = {
                executor.submit(llm_helper.get_batch_subculture_assignments, batch, time_period_context): batch
                for batch in all_batches
            }
            processed_requests = 0
            for future in as_completed(future_to_batch):
                processed_requests += len(future_to_batch[future])
                print(f"  -> LLM subculture progress: {processed_requests}/{len(uncached_requests)} requests completed.")
                try:
                    batch_results = future.result()
                    if batch_results:
                        network_llm_results.update(batch_results)
                except Exception as exc:
                    print(f"  -> ERROR: A subculture batch generated an exception: {exc}")

    # 3. Apply results
    final_results = {**cached_results, **network_llm_results}

    for req_id, result_data in final_results.items():
        chosen_subculture = result_data.get("chosen_subculture")
        faction_element = faction_elements_to_process.get(req_id)

        if faction_element and chosen_subculture:
            faction_element.set('subculture', chosen_subculture)
            subcultures_assigned_count += 1
            print(f"    -> Assigned subculture '{chosen_subculture}' to faction '{faction_element.get('name')}' via LLM.")
        elif faction_element:
            print(f"    -> WARNING: LLM failed to assign a valid subculture for faction '{faction_element.get('name')}' (Request ID: {req_id}).")

    print(f"\nLLM Subculture Assignment Pass complete. Assigned {subcultures_assigned_count} subcultures.")
    return subcultures_assigned_count
