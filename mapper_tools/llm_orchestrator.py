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

        # Group by culture and type for logging purposes
        requests_by_culture_and_type = defaultdict(list)
        for req in deduplicated_requests:
            faction_name = req.get('faction', '')
            faction_key = screen_name_to_faction_key_map.get(faction_name)
            subculture = faction_to_subculture_map.get(faction_key) if faction_key else None
            request_type = req.get('tag_name', 'unknown')
            group_key = f"{subculture or 'no_culture'}|{request_type}"
            requests_by_culture_and_type[group_key].append(req)

        # --- PARALLEL UNIT REQUEST PROCESSING ---
        all_requests_to_process = deduplicated_requests
        print(f"  -> Submitting {len(all_requests_to_process)} unit requests to LLM using {llm_threads} threads...")

        network_unit_results_list = []
        group_progress = defaultdict(int)
        
        req_id_to_group_info = {}
        for group_key, group_requests in requests_by_culture_and_type.items():
            subculture, request_type = group_key.split('|', 1)
            for req in group_requests:
                req_id_to_group_info[req['id']] = (subculture, request_type, len(