import os
import json
import threading
import hashlib
import datetime
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import random # Added for random.sample in prompt builders
from typing import Optional
from collections import defaultdict

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import g4f
    G4F_AVAILABLE = True
except ImportError:
    G4F_AVAILABLE = False

from mapper_tools import shared_utils, llm_cache_schema
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMHelper:
    """
    A helper class to manage interactions with a Large Language Model (LLM),
    including prompt generation, caching, and API calls.
    """
    def __init__(self, cache_dir, llm_config=None, cache_tag=None,
                 force_refresh=False, units_to_recache=None, excluded_units_set=None,
                 network_calls_enabled=True, unit_to_screen_name_map=None, use_g4f=False,
                 clear_null_cache=False, faction_culture_map=None,
                 all_units=None, unit_to_class_map=None,
                 main_units_tables_required=False, main_units_keys=None,
                 force_refresh_thematic=False, unit_to_tier_map=None, unit_to_description_map=None):
        """
        Initializes the LLMHelper.

        Args:
            cache_dir (str): The directory to store cache files.
            llm_config (dict, optional): A dictionary containing 'llm_configs' and 'global_settings'.
            cache_tag (str, optional): A unique tag for the cache file. Defaults to None.
            force_refresh (bool, optional): If True, ignore existing cache. Defaults to False.
            units_to_recache (set, optional): A set of unit keys to force recache for. Defaults to None.
            excluded_units_set (set, optional): A set of globally excluded unit keys. Defaults to None.
            network_calls_enabled (bool, optional): If False, runs in cache-only mode. Defaults to True.
            unit_to_screen_name_map (dict, optional): Map of unit keys to screen names. Defaults to None.
            use_g4f (bool, optional): If True, use g4f instead of litellm. Defaults to False.
            clear_null_cache (bool, optional): If True, clear null entries from cache on load. Defaults to False.
        """
        llm_config = llm_config or {}
        self.llm_configs = llm_config.get('llm_configs', [])
        self.review_llm_configs = llm_config.get('review_llm_configs')
        global_llm_settings = llm_config.get('global_settings', {})
        self.loop_on_failure = global_llm_settings.get('loop_on_failure', False)
        self.g4f_fallback_config = global_llm_settings.get('g4f_fallback_config', {})
        self.llm_cooldowns = {}
        self.tier_consecutive_failures = defaultdict(int)
        self.randomize_tiers = global_llm_settings.get('randomize_tiers', False)
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        self.force_refresh = force_refresh
        self.units_to_recache = units_to_recache or set()
        self.excluded_units_set = excluded_units_set or set()
        self.network_calls_enabled = network_calls_enabled
        self.unit_to_screen_name_map = unit_to_screen_name_map or {}
        self.use_g4f = use_g4f
        self.faction_culture_map = faction_culture_map or {}
        self.all_units = all_units or set()
        self.unit_to_class_map = unit_to_class_map or {}
        self.main_units_tables_required = main_units_tables_required
        self.main_units_keys = main_units_keys or set()
        self.unit_to_description_map = unit_to_description_map or {}

        self.unit_to_tier_map = unit_to_tier_map or {}

        if self.use_g4f and not G4F_AVAILABLE:
            raise ImportError("The 'g4f' library is required. Please install it using 'pip install g4f'.")
        if not self.use_g4f:
            if not LITELLM_AVAILABLE:
                raise ImportError("The 'litellm' library is required. Please install it using 'pip install litellm'.")
            if not G4F_AVAILABLE:
                print("Warning: 'g4f' library not found. Fallback functionality will be disabled.")

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = self._get_cache_path()
        self.cache_lock = threading.Lock()
        self.cache = self.load_cache(clear_null_cache)

        if force_refresh_thematic:
            self.clear_thematic_cache()



    def _get_single_preset_assignment(self, request, time_period_context, prompt_builder_func):
        """
        Processes a single preset assignment request.
        This includes prompt building, LLM call with retries, response parsing, and validation.
        Returns a tuple (request_id, chosen_preset).
        chosen_preset is the validated preset key string or None if it fails.
        """
        MAX_RETRIES = 3
        chosen_preset = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = prompt_builder_func(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_chosen_preset = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    temp_chosen_preset = response_data.get("chosen_preset")
                    # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                    if "chosen_preset" in response_data:
                        # Valid JSON with explicit key - accept the value (even if None)
                        chosen_preset = temp_chosen_preset
                        break
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            if temp_chosen_preset is not None:
                available_presets = request.get('preset_pool', [])
                if available_presets and temp_chosen_preset in available_presets:
                    chosen_preset = temp_chosen_preset
                    break  # Success, exit retry loop
                else:
                    failure_reason = f"LLM suggested invalid preset '{temp_chosen_preset}' (not in available list)"
            elif not failure_reason and response_content: # If temp_chosen_preset is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for preset assignment"

            # Log failure and decide whether to retry
            if attempt < MAX_RETRIES - 1:
                # print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...") # Optional verbose log
                current_delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(current_delay)  # Exponential backoff with jitter
            # else:
                # print(f"  -> ERROR: Failed to get a valid preset for request {request_id} after {MAX_RETRIES} attempts. Reason: {failure_reason}") # Optional verbose log

        # If chosen_preset is still None here, it means all retries failed.
        # The calling function will handle the None case and update cache accordingly.
        return request_id, chosen_preset

    def clear_thematic_cache(self):
        """
        Removes all cache entries related to thematic MAA classification and global thematic assignments.
        """
        if not self.cache:
            return

        cache_modified = False
        with self.cache_lock:
            keys_to_remove = [
                req_id for req_id in self.cache.keys()
                if req_id.startswith("maa_classification|") or req_id.startswith("global_thematic_assignment|")
            ]
            for req_id in keys_to_remove:
                del self.cache[req_id]
                cache_modified = True

        if cache_modified:
            print(f"LLM Cache: Cleared {len(keys_to_remove)} thematic classification and assignment entries.")
            self.save_cache()


    def clear_unit_from_cache(self, unit_key):
        """
        Removes all cache entries that reference a specific unit key.
        This is used to clear excluded units from the cache.
        """
        if not self.cache:
            return

        cache_modified = False
        with self.cache_lock:
            # Get list of keys to remove to avoid modifying dict during iteration
            keys_to_remove = []
            for req_id, cached_entry in self.cache.items():
                if isinstance(cached_entry, dict):
                    # Check if this cache entry references the unit_key
                    chosen_unit = cached_entry.get("chosen_unit")
                    if chosen_unit == unit_key:
                        keys_to_remove.append(req_id)

            # Remove the identified keys
            for req_id in keys_to_remove:
                del self.cache[req_id]
                cache_modified = True

        if cache_modified:
            self.save_cache()

    def is_unit_in_cache(self, unit_key):
        """
        Checks if a unit key exists in any cache entry.
        Returns True if the unit is found in any cache entry, False otherwise.
        """
        if not self.cache:
            return False

        with self.cache_lock:
            for cached_entry in self.cache.values():
                if isinstance(cached_entry, dict):
                    chosen_unit = cached_entry.get("chosen_unit")
                    if chosen_unit == unit_key:
                        return True
        return False

    def _get_cache_path(self):
        """Constructs the path to the cache file based on the cache_tag."""
        if not self.cache_tag:
            return None
        # Sanitize tag for filename
        safe_tag = "".join(c for c in self.cache_tag if c.isalnum() or c in ('_', '-')).rstrip()
        return os.path.join(self.cache_dir, f"llm_cache_{safe_tag}.json")

    def load_cache(self, clear_null_cache=False):
        """Loads the cache from a JSON file."""
        if not self.cache_path:
            return {}

        def _apply_clear_null_cache(cache_data):
            """Apply clear_null_cache logic to cache data."""
            if clear_null_cache:
                original_size = len(cache_data)
                # Filter out entries where the value is None or an empty dict
                cache_data = {k: v for k, v in cache_data.items() if v}
                if len(cache_data) < original_size:
                    print(f"LLM Cache: Cleared {original_size - len(cache_data)} null/empty entries.")
            return cache_data

        with self.cache_lock:
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Validate keys and filter out malformed ones
                original_size = len(cache_data)
                validated_cache = {
                    k: v for k, v in cache_data.items() if llm_cache_schema.validate_key_format(k)
                }
                if len(validated_cache) < original_size:
                    print(f"LLM Cache: Discarded {original_size - len(validated_cache)} entries with malformed keys.")

                validated_cache = _apply_clear_null_cache(validated_cache)
                return validated_cache
            except FileNotFoundError:
                print(f"Info: LLM cache file not found at '{self.cache_path}'. A new one will be created on save.")
                return {}
            except (json.JSONDecodeError, IOError, OSError) as e:
                print(f"Warning: Could not load or parse LLM cache file at '{self.cache_path}': {e}")
                # Try to load from backup file
                backup_path = self.cache_path + '.bak'
                if os.path.exists(backup_path):
                    print(f"Attempting to load cache from backup file: {backup_path}")
                    try:
                        with open(backup_path, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        print("Successfully loaded cache from backup file")

                        # Validate keys and filter out malformed ones from backup
                        original_size = len(cache_data)
                        validated_cache = {
                            k: v for k, v in cache_data.items() if llm_cache_schema.validate_key_format(k)
                        }
                        if len(validated_cache) < original_size:
                            print(f"LLM Cache (Backup): Discarded {original_size - len(validated_cache)} entries with malformed keys.")

                        # Apply clear_null_cache to backup data
                        cache_data = _apply_clear_null_cache(validated_cache)

                        # Restore primary cache file from backup
                        try:
                            with open(self.cache_path, 'w', encoding='utf-8') as f:
                                json.dump(cache_data, f, indent=2)
                            print(f"Restored primary cache file from backup: {self.cache_path}")
                        except (IOError, OSError) as restore_error:
                            print(f"Warning: Could not restore primary cache file: {restore_error}")

                        return cache_data
                    except (json.JSONDecodeError, IOError, OSError) as backup_error:
                        print(f"Fatal Error: Backup cache file is also corrupt or unavailable: {backup_error}")
                        return {}  # Return empty cache to allow application to continue
                else:
                    print(f"Info: Primary cache file is corrupt or unreadable and no backup exists. A new cache will be created.")
                    return {}  # Return empty cache to allow application to continue

    def save_cache(self):
        """Saves the current cache to a JSON file."""
        if not self.cache_path:
            return
        with self.cache_lock:
            try:
                # Create backup of existing cache file before overwriting
                if os.path.exists(self.cache_path):
                    backup_path = self.cache_path + '.bak'
                    os.replace(self.cache_path, backup_path)

                # Write new cache file
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
            except (IOError, OSError) as e:
                print(f"Error: Could not save LLM cache to '{self.cache_path}': {e}")
                # Don't raise exception - let application continue with backup intact

    def is_cache_entry_valid(self, req_id, validation_pool):
        """
        Checks if a cache entry is valid.
        An entry is invalid if it's missing, its unit is excluded, marked for recache,
        or no longer in the faction's potential unit pool (stale).
        """
        # First, validate the key format itself.
        if not llm_cache_schema.validate_key_format(req_id):
            print(f"Warning: Attempted to check cache with invalid key format: '{req_id}'")
            return False

        if req_id not in self.cache:
            return False

        cached_entry = self.cache.get(req_id)
        if not isinstance(cached_entry, dict):
            return False # Invalid entry format

        # --- Helper to check if a null entry is stale by comparing context hashes ---
        def is_null_entry_stale(entry, current_pool):
            cached_hash = entry.get('pool_hash')
            if not cached_hash:
                return True # Legacy null entry without a hash is considered stale.

            # Calculate the hash of the current context to see if it has changed.
            current_pool_list = sorted(list(current_pool or []))
            current_hash = hashlib.sha1(','.join(current_pool_list).encode()).hexdigest()
            return cached_hash != current_hash

        # --- Generic Handlers ---
        # Handler for any request that returns a single chosen item (unit, faction, key, etc.)
        def handle_single_choice(key_name):
            if key_name not in cached_entry:
                return False

            chosen_item = cached_entry.get(key_name)

            if chosen_item is not None:
                # It's a non-null entry, validate it.
                if validation_pool and chosen_item not in validation_pool:
                    return False # Stale: The chosen item is no longer in the valid pool.

                # For units, check additional exclusion criteria.
                if key_name in ["chosen_unit", "chosen_naval_key"]:
                    if chosen_item in self.excluded_units_set: return False
                    if chosen_item in self.units_to_recache: return False
                    if self.main_units_tables_required and chosen_item not in self.main_units_keys: return False

                return True
            else:
                # It's a null entry, check if the context has changed.
                return not is_null_entry_stale(cached_entry, validation_pool)

        # Handler for any request that returns a list of items.
        def handle_list_choice(key_name):
            if key_name not in cached_entry:
                return False

            chosen_list = cached_entry.get(key_name)

            if chosen_list is not None:
                if not isinstance(chosen_list, list): return False
                # It's a non-empty list, validate its contents.
                if validation_pool:
                    for item in chosen_list:
                        # Handle different list item formats
                        item_key = None
                        if isinstance(item, dict):
                            item_key = item.get('key') or item.get('naval_key')
                        elif isinstance(item, str):
                            item_key = item

                        if item_key and item_key not in validation_pool:
                            return False # Stale: An item in the list is no longer valid.
                return True
            else:
                # It's a null entry, check context hash.
                return not is_null_entry_stale(cached_entry, validation_pool)

        # --- Routing based on key prefix ---
        if req_id.startswith("review|"):
            return cached_entry.get("status") == "success"
        elif req_id.startswith("maa_classification|"):
            return "classification" in cached_entry
        elif req_id.startswith("unit_replacement|"):
            return handle_single_choice("chosen_unit")
        elif req_id.startswith("global_thematic_assignment|"):
            return handle_single_choice("chosen_unit")
        elif req_id.startswith("faction_match|"):
            return handle_single_choice("best_match")
        elif req_id.startswith("json_group_match|"):
            return handle_single_choice("chosen_group_key")
        elif req_id.startswith("subculture|"):
            return handle_single_choice("chosen_subculture")
        elif req_id.startswith("NavalKey|"):
            return handle_single_choice("chosen_naval_key")
        elif req_id.startswith("Heritage|") or req_id.startswith("Culture|"):
            return handle_single_choice("chosen_faction")
        elif req_id.startswith("settlement|") or req_id.startswith("bridge_") or req_id.startswith("coastal_") or req_id.startswith("building_") or req_id.startswith("terrain_"):
            return handle_single_choice("chosen_preset")
        elif req_id.startswith("semantic_match|"):
            return handle_list_choice("similar_factions")
        elif req_id.startswith("LevyComposition|"):
            return handle_list_choice("chosen_composition")
        elif req_id.startswith("NavalRoster|"):
            return handle_list_choice("chosen_roster")

        # Fallback for original unit assignment keys
        return handle_single_choice("chosen_unit")



    def filter_requests_against_cache(self, requests):
        """
        Filters a list of requests, separating them into cached and uncached lists.
        """
        cached_results = {}
        uncached_requests = []
        cache_modified = False

        for req in requests:
            req_id = req['id']
            validation_pool = req.get('validation_pool') # LevyComposition requests might not have this

            if not self.force_refresh and self.is_cache_entry_valid(req_id, validation_pool):
                cached_results[req_id] = self.cache[req_id]
            else:
                # Evict invalid/stale entry if it exists
                if req_id in self.cache:
                    with self.cache_lock:
                        del self.cache[req_id]
                        cache_modified = True
                uncached_requests.append(req)

        return cached_results, uncached_requests, cache_modified

    def _truncate_prompt(self, prompt, max_length):
        """
        Intelligently truncates a prompt string to be within max_length,
        focusing on lists of candidates which are the most likely cause of overflow.
        """
        # This is a heuristic, but it's better than failing.
        # It assumes the largest part of the prompt is a candidate list.
        candidate_markers = [
            "**Candidate Units (Prioritized by Cultural Relevance):**",
            "**Available Naval Transport Units:**",
            "**Available Warships (Categorized):**",
            "**List of Valid Canonical Faction Names:**",
            "**Available Subcultures:**",
            "**Available Settlement Presets (Architectural Styles):",
            "**Available Land Bridge Presets:**",
            "**Available Coastal Battle Presets:**",
            "**Available Battle Presets:**",
            "**Strictly Cultural Unit Pool (for Levies and Garrisons):**",
            "**Local Unit Pool (for General, Knights, and generic MenAtArm):**",
            "**Global Unit Pool (for Thematic Consistency):**",
            "Available Units (Global Pool):"
        ]
        instruction_marker = "\n**Instructions:**"

        start_index = -1
        for marker in candidate_markers:
            start_index = prompt.find(marker)
            if start_index != -1:
                break

        # If no specific candidate marker is found, look for a generic JSON block
        if start_index == -1:
            start_index = prompt.find("```json\n")

        end_index = prompt.find(instruction_marker)

        if start_index != -1 and end_index != -1 and start_index < end_index:
            header = prompt[:start_index]
            footer = prompt[end_index:]
            middle = prompt[start_index:end_index]

            available_space_for_middle = max_length - len(header) - len(footer)

            if available_space_for_middle < 200:  # Need some space for at least a few candidates
                # Cannot truncate meaningfully, just truncate the whole prompt at the end
                return prompt[:max_length]

            truncated_middle = middle[:available_space_for_middle]
            # Ensure we don't cut in the middle of a line
            last_newline = truncated_middle.rfind('\n')
            if last_newline != -1:
                truncated_middle = truncated_middle[:last_newline]

            return header + truncated_middle + "\n... (list truncated)\n" + footer

        # If markers not found, just truncate from the end (less ideal)
        return prompt[:max_length]

    def _call_llm_with_retry(self, prompt, base_delay=10, use_review_config=False):
        """
        Wrapper for LLM calls with tiered fallback, retry, cooldown, and looping logic.
        Handles exponential backoff for both retries and cooldowns based on tier configuration.
        """
        while True:
            # --- NEW: Select config list ---
            configs_to_use = self.llm_configs
            if use_review_config and self.review_llm_configs:
                print("LLM Helper: Using 'review_llm_configs' for this request.")
                configs_to_use = self.review_llm_configs
            # --- END NEW ---

            # Sort configurations by priority, with a default of 99 for items without a priority
            configs_to_use = sorted(configs_to_use, key=lambda c: c.get('priority', 99))

            all_tiers_on_cooldown = True
            for tier_idx, config in enumerate(configs_to_use):
                model = config.get('model')
                api_base = config.get('api_base')
                api_key = config.get('api_key') or "not-needed"
                max_retries = config.get('max_retries', 3)
                timeout = config.get('timeout', 200)
                tier_name = config.get('name', f"Tier {tier_idx + 1}")
                max_prompt_length = config.get('max_prompt_length')
                context_size = config.get('context_size', 90000)

                # Dynamic backoff and cooldown parameters from config
                # initial_retry_delay: Seconds to wait for the first retry.
                # retry_backoff_factor: Multiplier for each subsequent retry wait (e.g., 2 for doubling).
                # max_retry_delay: The maximum wait time between retries.
                initial_retry_delay = config.get('initial_retry_delay', base_delay)
                retry_backoff_factor = config.get('retry_backoff_factor', 2)
                max_retry_delay = config.get('max_retry_delay', 120)

                # initial_cooldown: Seconds for the first cooldown period after exhausting retries.
                # cooldown_backoff_factor: Multiplier for each subsequent cooldown period.
                # max_cooldown: The maximum duration for a cooldown period.
                initial_cooldown = config.get('initial_cooldown', 60)
                cooldown_backoff_factor = config.get('cooldown_backoff_factor', 2)
                max_cooldown = config.get('max_cooldown', 3600)

                if tier_name in self.llm_cooldowns and time.time() < self.llm_cooldowns[tier_name]:
                    continue

                all_tiers_on_cooldown = False
                print(f"LLM Helper: Attempting {tier_name} (Model: {model})")

                current_prompt = prompt
                effective_max_length = max_prompt_length or context_size
                if effective_max_length and len(current_prompt) > effective_max_length:
                    print(f"  -> Prompt length ({len(current_prompt)}) exceeds limit ({effective_max_length}). Truncating...")
                    current_prompt = self._truncate_prompt(current_prompt, effective_max_length)
                    print(f"  -> New prompt length: {len(current_prompt)}")

                for attempt in range(max_retries):
                    try:
                        if self.use_g4f:
                            client = g4f.client.Client(api_key=api_key, verify=False)
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": current_prompt}],
                            )
                            self.tier_consecutive_failures[tier_name] = 0
                            return response.choices[0].message.content
                        else:
                            litellm.api_base = api_base
                            litellm.api_key = api_key
                            response = litellm.completion(
                                model=model,
                                messages=[{"role": "user", "content": current_prompt}],
                                temperature=0.1,
                                request_timeout=timeout
                            )
                            self.tier_consecutive_failures[tier_name] = 0
                            return response.choices[0].message.content
                    except Exception as e:
                        # NEW: Handle Context Window Exceeded Error
                        is_context_window_error = False
                        if LITELLM_AVAILABLE and hasattr(litellm, 'ContextWindowExceededError') and isinstance(e, litellm.ContextWindowExceededError):
                            is_context_window_error = True
                        elif "ContextWindowExceededError" in str(e): # Fallback for nested errors
                            is_context_window_error = True

                        if is_context_window_error:
                            # Reduce prompt size and prepare for the next attempt
                            new_length = int(len(current_prompt) * 0.8) # Reduce by 20%
                            print(f"  -> {tier_name} failed due to ContextWindowExceededError. Truncating prompt from {len(current_prompt)} to {new_length} for the next attempt.")
                            current_prompt = self._truncate_prompt(current_prompt, new_length)

                        print(f"  -> {tier_name} failed on attempt {attempt + 1}/{max_retries}: {e}")
                        if attempt < max_retries - 1:
                            delay = initial_retry_delay * (retry_backoff_factor ** attempt)
                            current_delay = min(delay, max_retry_delay) + random.uniform(0, 1)
                            time.sleep(current_delay)

                print(f"  -> {tier_name} exhausted all retries. Setting cooldown and moving to next fallback...")
                self.tier_consecutive_failures[tier_name] += 1
                consecutive_failures = self.tier_consecutive_failures[tier_name]

                cooldown_duration = initial_cooldown * (cooldown_backoff_factor ** (consecutive_failures - 1))
                final_cooldown = min(cooldown_duration, max_cooldown)

                self.llm_cooldowns[tier_name] = time.time() + final_cooldown
                print(f"  -> {tier_name} has failed {consecutive_failures} consecutive time(s). Cooldown set for {final_cooldown:.1f} seconds.")

            # Final Fallback: g4f as last resort if enabled
            g4f_tier_name = "g4f_fallback"
            if G4F_AVAILABLE and (g4f_tier_name not in self.llm_cooldowns or time.time() >= self.llm_cooldowns[g4f_tier_name]):
                all_tiers_on_cooldown = False
                print(f"\n--- CRITICAL: All configured LLM tiers failed. Attempting g4f last-resort fallback ---")
                try:
                    client = g4f.client.Client(verify=False)
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                    )
                    self.tier_consecutive_failures[g4f_tier_name] = 0
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"  -> g4f last-resort fallback failed: {e}")

                    # Use configurable backoff for g4f fallback
                    g4f_config = self.g4f_fallback_config
                    initial_cooldown = g4f_config.get('initial_cooldown', 60)
                    cooldown_backoff_factor = g4f_config.get('cooldown_backoff_factor', 2)
                    max_cooldown = g4f_config.get('max_cooldown', 3600)

                    self.tier_consecutive_failures[g4f_tier_name] += 1
                    consecutive_failures = self.tier_consecutive_failures[tier_name]
                    cooldown_duration = initial_cooldown * (cooldown_backoff_factor ** (consecutive_failures - 1))
                    final_cooldown = min(cooldown_duration, max_cooldown)

                    self.llm_cooldowns[g4f_tier_name] = time.time() + final_cooldown
                    print(f"  -> {g4f_tier_name} has failed {consecutive_failures} consecutive time(s). Cooldown set for {final_cooldown:.1f} seconds.")

            if not self.loop_on_failure:
                print("  -> ERROR: LLM call failed after all tiers and fallbacks. Looping is disabled.")
                return None

            if all_tiers_on_cooldown:
                active_cooldowns = {k: v for k, v in self.llm_cooldowns.items() if v > time.time()}
                if not active_cooldowns:
                    print("  -> All cooldowns expired. Retrying LLM call sequence immediately...")
                    continue

                earliest_expiry = min(active_cooldowns.values())
                wait_time = earliest_expiry - time.time()
                if wait_time > 0:
                    print(f"All LLM tiers are on cooldown. Waiting for {wait_time:.1f} seconds until the next tier is available...")
                    time.sleep(wait_time)

            print("  -> Retrying LLM call sequence...")

    def get_unit_replacement(self, request, time_period_context):
        """
        Gets a unit replacement for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        chosen_unit = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_unit_replacement_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_chosen_unit = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_chosen_unit = response_data.get("chosen_unit")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_unit" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_unit = temp_chosen_unit
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            if temp_chosen_unit is not None:
                if temp_chosen_unit in request.get('validation_pool', []):
                    chosen_unit = temp_chosen_unit
                    break
                else:
                    failure_reason = f"LLM suggested unit '{temp_chosen_unit}' which is not in the validation pool"
            elif not failure_reason and response_content: # If temp_chosen_unit is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for unit replacement"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(current_delay)

        result_data = {"chosen_unit": chosen_unit}
        if chosen_unit is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data


    def get_faction_to_json_group_match(self, request, time_period_context):
        """
        Gets the best JSON cultural group key for a faction name from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        chosen_group_key = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_faction_to_json_group_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_group_key = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_group_key = response_data.get("chosen_group_key")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_group_key" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_group_key = temp_group_key
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            valid_keys = request.get('validation_pool', [])
            if temp_group_key is not None:
                if temp_group_key in valid_keys:
                    chosen_group_key = temp_group_key
                    break
                else:
                    failure_reason = f"LLM returned invalid group key '{temp_group_key}' (not in validation pool)"
            elif not failure_reason and response_content: # If temp_group_key is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for JSON group match"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(current_delay)

        result_data = {"chosen_group_key": chosen_group_key}
        if chosen_group_key is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data


    def get_naval_key_assignment(self, request, time_period_context):
        """
        Gets a naval key assignment for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        chosen_naval_key = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_naval_key_assignment_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_naval_key = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_naval_key = response_data.get("chosen_naval_key")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_naval_key" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_naval_key = temp_naval_key
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            if temp_naval_key is not None:
                if temp_naval_key in request.get('validation_pool', []):
                    chosen_naval_key = temp_naval_key
                    break
                else:
                    failure_reason = f"LLM suggested naval key '{temp_naval_key}' which is not in the validation pool"
            elif not failure_reason and response_content: # If temp_naval_key is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for naval key assignment"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        result_data = {
            "chosen_naval_key": chosen_naval_key,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }

        return request_id, result_data


    def get_faction_assignment(self, request, time_period_context):
        """
        Gets a faction assignment for a single heritage/culture from the LLM.
        Returns a tuple of (request_id, chosen_faction).
        """
        MAX_RETRIES = 3
        chosen_faction = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_faction_assignment_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_chosen_faction = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_chosen_faction = response_data.get("chosen_faction")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_faction" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_faction = temp_chosen_faction
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            valid_factions = request.get('faction_pool', [])
            if temp_chosen_faction is not None:
                if temp_chosen_faction in valid_factions:
                    chosen_faction = temp_chosen_faction
                    break
                else:
                    failure_reason = f"LLM returned invalid faction name '{temp_chosen_faction}' (not in faction pool)"
            elif not failure_reason and response_content: # If temp_chosen_faction is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for faction assignment"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        return request_id, chosen_faction

    def get_batch_faction_assignments(self, batch, time_period_context):
        """
        Gets faction assignments for a batch of heritages/cultures from the LLM.
        This function orchestrates the parallel execution of individual requests.
        """
        if not self.network_calls_enabled:
            print("LLM network calls are disabled. Cannot process faction assignments.")
            return {}

        results = {}
        # It's processing a single batch from the outer loop in culture_fixer.
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_req_id = {
                executor.submit(self.get_faction_assignment, req, time_period_context): req['id']
                for req in batch
            }
            for future in as_completed(future_to_req_id):
                req_id = future_to_req_id[future]
                try:
                    _, chosen_faction = future.result()
                    results[req_id] = {
                        "chosen_faction": chosen_faction,
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                except Exception as exc:
                    print(f"  -> ERROR: A faction assignment request for '{req_id}' generated an exception: {exc}")
                    results[req_id] = {"chosen_faction": None}
                    raise exc

        # Update cache with all new results from this batch
        if results:
            with self.cache_lock:
                self.cache.update(results)
            self.save_cache()

        return results

    def _build_faction_assignment_prompt(self, request, time_period_context):
        """Builds a detailed prompt for a faction assignment request."""
        heritage = request.get('heritage', 'N/A')
        cultures = request.get('cultures', [])
        prioritized_candidates = request.get('prioritized_candidates') # Check for this first
        faction_pool = request.get('faction_pool', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to assign the most culturally and thematically appropriate Attila faction to a Crusader Kings 3 (CK3) heritage.\n\n"
        )
        prompt += f"**CK3 Heritage to Assign Faction:** {heritage}\n"
        prompt += f"**Associated CK3 Cultures in this Heritage:** {', '.join(cultures)}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        if prioritized_candidates:
            prompt += "**Candidate Factions (Prioritized by Semantic Relevance):**\n"
            prompt += "Your choice should come from this list of thematically similar factions.\n\n"
            for faction in prioritized_candidates:
                prompt += f"- {faction}\n"
        else:
            prompt += "**List of Available Attila Factions:**\n"
            if faction_pool:
                MAX_NAMES_TO_SEND = 200
                if len(faction_pool) > MAX_NAMES_TO_SEND:
                    sampled_names = random.sample(faction_pool, MAX_NAMES_TO_SEND)
                    prompt += f"(Showing a random sample of {MAX_NAMES_TO_SEND} out of {len(faction_pool)} total factions)\n"
                    for name in sorted(sampled_names):
                        prompt += f"- {name}\n"
                else:
                    for name in sorted(faction_pool):
                        prompt += f"- {name}\n"
            else:
                prompt += "None\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the CK3 heritage and its associated cultures.\n"
            "2.  Select the single best faction from the provided list that is the closest thematic and cultural match.\n"
            "3.  Consider the historical context and the likely Attila equivalent of the CK3 cultures.\n"
            "4.  You MUST choose a faction from the provided list. Do not invent a new one.\n"
            "5.  Provide your answer in a JSON block with the key 'chosen_faction'. Example: {\"chosen_faction\": \"att_fact_franks\"}\n"
            "6.  IMPORTANT: Your answer must come from the above list of factions.\n"
            "7.  If no faction is a good match, provide null. Example: {\"chosen_faction\": null}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt


    def _build_faction_to_json_group_prompt(self, request, time_period_context):
        """Builds a prompt for matching a faction to a JSON cultural group key."""
        faction_name = request.get('faction_name', 'N/A')
        available_keys = request.get('available_group_keys', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to match a faction to its most appropriate cultural/thematic group from a predefined list of group keys.\n\n"
        )
        prompt += f"**Faction Name to Match:** {faction_name}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Available Cultural Group Keys:**\n"
        if available_keys:
            for key in sorted(available_keys):
                prompt += f"- {key}\n"
        else:
            prompt += "None\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the faction name and its likely culture (e.g., 'Norse' is North Germanic, 'Castilian' is Iberian).\n"
            "2.  Select the single best group key from the 'Available Cultural Group Keys' list that represents this faction's culture.\n"
            "3.  The key '*' is a global fallback and should only be chosen if no other key is remotely appropriate.\n"
            "4.  You MUST choose a key from the provided list. Do not invent a new one.\n"
            "5.  Provide your answer in a JSON block with the key 'chosen_group_key'. Example: {\"chosen_group_key\": \"North Germanic\"}\n"
            "6.  If no key is a good match, provide null. Example: {\"chosen_group_key\": null}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def get_levy_composition(self, request, time_period_context):
        """
        Gets a levy composition (specific units and percentages) for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        chosen_composition = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_levy_composition_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_composition = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_composition = response_data.get("chosen_composition")
                        if "chosen_composition" in response_data:
                            chosen_composition = temp_composition
                            if chosen_composition is None:
                                break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation
            if temp_composition is not None and isinstance(temp_composition, list):
                valid_composition = True
                total_percentage = 0
                validation_pool = set(request.get('validation_pool', []))

                for item in temp_composition:
                    unit_key = item.get('key')
                    percentage = item.get('percentage')
                    if not unit_key or unit_key not in validation_pool:
                        failure_reason = f"LLM suggested invalid unit key '{unit_key}'"
                        valid_composition = False
                        break
                    if not isinstance(percentage, int) or percentage < 0:
                        failure_reason = f"LLM suggested invalid percentage '{percentage}' for unit '{unit_key}'"
                        valid_composition = False
                        break
                    total_percentage += percentage

                if valid_composition and total_percentage != 100:
                    failure_reason = f"LLM suggested levy percentages sum to {total_percentage}, expected 100"
                    valid_composition = False

                if valid_composition:
                    chosen_composition = temp_composition
                    break
            elif not failure_reason and response_content:
                failure_reason = "LLM explicitly returned null for levy composition"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        result_data = {"chosen_composition": chosen_composition}
        if chosen_composition is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data


    def get_naval_roster(self, request, time_period_context):
        """
        Gets a naval roster for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        chosen_roster = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_naval_roster_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_roster = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_roster = response_data.get("chosen_roster")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_roster" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_roster = temp_roster
                            if chosen_roster is None:
                                break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            if temp_roster is not None and isinstance(temp_roster, list):
                valid_roster = True
                validation_pool = request.get('validation_pool', [])

                for ship in temp_roster:
                    if not isinstance(ship, dict) or 'naval_key' not in ship:
                        failure_reason = "Malformed ship object in roster"
                        valid_roster = False
                        break
                    if ship['naval_key'] not in validation_pool:
                        failure_reason = f"LLM suggested invalid naval unit '{ship['naval_key']}'"
                        valid_roster = False
                        break

                if valid_roster:
                    chosen_roster = temp_roster
                    break
            elif not failure_reason and response_content: # If temp_roster is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for naval roster"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        result_data = {"chosen_roster": chosen_roster}
        if chosen_roster is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data

    def get_batch_mapping_updates(self, batch):
        """
        Gets mapping updates for a batch of new CK3 MAA types.
        """
        # This is a placeholder for the actual implementation.
        # In a real scenario, you would batch these requests.
        print("LLM mapping update feature is a placeholder.")
        return {}


    def get_batch_unit_assignment(self, request, time_period_context):
        """
        Gets unit assignments for a batch of slots from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        req_id = request['id']
        result_data = {}

        for attempt in range(MAX_RETRIES):
            result_data = {
                "assignments": {},
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "failure",
                "reason": "Unknown error during processing."
            }

            prompt = self._build_batch_unit_assignment_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            if not response_content:
                result_data["reason"] = "LLM call failed or returned no content."
            else:
                response_data = shared_utils.parse_llm_json_response(response_content, req_id)
                if isinstance(response_data, dict) and "assignments" in response_data:
                    assignments = response_data.get("assignments", {})
                    if isinstance(assignments, dict):
                        # Basic validation
                        valid_assignments = {}
                        all_valid = True
                        for slot_id, unit_key in assignments.items():
                            if unit_key is None or (unit_key in request['validation_pool']):
                                valid_assignments[slot_id] = unit_key
                            else:
                                all_valid = False
                                result_data["reason"] = f"LLM suggested invalid unit '{unit_key}' for slot '{slot_id}'."
                                break
                        
                        if all_valid:
                            result_data["status"] = "success"
                            result_data["reason"] = "Successfully processed."
                            result_data["assignments"] = valid_assignments
                            break
                    else:
                        result_data["reason"] = "LLM 'assignments' key is not a dictionary."
                else:
                    result_data["reason"] = "Could not find or parse JSON block with 'assignments' key in LLM response."

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: Batch assignment failed for {req_id} on attempt {attempt + 1}: {result_data['reason']}. Retrying...")
                time.sleep(2 * (2 ** attempt) + random.uniform(0, 1))

        return req_id, result_data

    def _build_batch_unit_assignment_prompt(self, request, time_period_context):
        """Builds a detailed prompt for a batch unit assignment request."""
        prompt = f"You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. Your task is to select the best possible unit replacements for multiple empty slots in a faction's roster.\n\n"
        prompt += f"**Faction:** {request.get('faction', 'N/A')}\n"
        prompt += f"**Subculture:** {request.get('subculture', 'N/A')}\n"
        if request.get('prefix'):
            prompt += f"**Cultural Unit Prefix:** {request.get('prefix')}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Slots to Fill:**\n"
        prompt += "You must provide a unit assignment for each of the following `slot_id`s.\n"
        slots_to_fill = {s['slot_id']: s['description'] for s in request.get('slots', [])}
        prompt += f"```json\n{json.dumps(slots_to_fill, indent=2)}\n```\n\n"

        prompt += "**Candidate Unit Pools:**\n"
        prompt += "Use the correctly labeled pool for each slot. For MenAtArm, use the pool with the matching `slot_id`. For other roles, use the 'shared' pool.\n"
        
        pools = request.get('candidate_pools', {})
        for pool_name, unit_list in pools.items():
            prompt += f"\n--- Pool: '{pool_name}' ---\n"
            if unit_list:
                # Limit display to avoid excessive length, but the LLM knows all are available
                displayed_units = unit_list[:100]
                for unit in displayed_units:
                    tier = self.unit_to_tier_map.get(unit, 'N/A')
                    name = self.unit_to_screen_name_map.get(unit, 'N/A')
                    desc = self.unit_to_description_map.get(unit, '')
                    desc_text = f" | Desc: {desc}" if desc else ""
                    prompt += f"- {unit} (Tier: {tier}, Name: {name}{desc_text})\n"
                if len(unit_list) > 100:
                    prompt += f"... and {len(unit_list) - 100} more units in this pool.\n"
            else:
                prompt += "None\n"

        prompt += "\n**Instructions:**\n"
        prompt += "1. For each `slot_id` in the 'Slots to Fill' list, select the best unit from the appropriate candidate pool.\n"
        prompt += "2. **Prefix Guidance**: If a 'Cultural Unit Prefix' is provided, you MUST strongly prefer units that start with that prefix.\n"
        prompt += "3. **Quality Guidance**: Use the numerical `Tier` provided for each unit. Low numbers (e.g., Tier 1) are for levies/garrisons, while high numbers (e.g., Tier 3) are for generals/knights.\n"
        prompt += "4. Provide your answer in a single JSON block with the key 'assignments'. The value should be a dictionary mapping each `slot_id` to your chosen `unit_key`.\n"
        prompt += "5. IMPORTANT: Every `slot_id` from the request must be a key in your response.\n"
        prompt += "6. If no unit is suitable for a specific slot, use `null` as the value for that `slot_id`.\n"
        prompt += "Example Response:\n"
        prompt += "```json\n{\n  \"assignments\": {\n    \"General|1|\": \"att_rom_praeventores\",\n    \"MenAtArm|pikemen|\": \"att_rom_lanciarii_seniores\",\n    \"Levies|Melee Infantry|\": null\n  }\n}\n```\n"
        prompt += "Your response must contain nothing but the JSON block.\n"

        return prompt

    def get_roster_review(self, request, time_period_context, unit_to_tier_map=None):
        """
        Gets a roster review for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        req_id = request['id']
        result_data = {}

        for attempt in range(MAX_RETRIES):
            # Initialize result_data with a default failure state for each attempt
            result_data = {
                "corrections": [],
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "failure",
                "reason": "Unknown error during processing.",
                "invalid_suggestions": []
            }

            prompt = self._build_roster_review_prompt(request, time_period_context, unit_to_tier_map)
            response_content = self._call_llm_with_retry(prompt, use_review_config=True)

            if not response_content:
                result_data["reason"] = "LLM call failed or returned no content."
            else:
                response_data = shared_utils.parse_llm_json_response(response_content, req_id)
                if isinstance(response_data, dict):
                    corrections = response_data.get("corrections", [])

                    # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if empty list), set success
                    if "corrections" in response_data:
                        # Valid JSON with explicit key - accept the value and set success status
                        result_data["status"] = "success"
                        result_data["reason"] = "Successfully processed."
                        result_data["corrections"] = corrections if isinstance(corrections, list) else []
                        break

                    # Basic validation of corrections
                    valid_corrections = []
                    invalid_suggestions_list = []
                    has_invalid_suggestion = False

                    if isinstance(corrections, list):
                        for item in corrections:
                            if isinstance(item, dict) and 'tag' in item and 'identifier' in item and 'current_unit' in item and 'suggested_unit' in item:
                                suggested_unit = item['suggested_unit']
                                tag = item['tag']

                                # NEW: Validate 'type' for MenAtArm
                                if tag == 'MenAtArm' and 'type' not in item:
                                    # Try to recover the type from the original request
                                    original_maa_item = None
                                    req_identifier = item.get('identifier')
                                    if req_identifier and 'MenAtArm' in request.get('roster', {}):
                                        for maa_item in request['roster']['MenAtArm']:
                                            if maa_item.get('identifier') == req_identifier:
                                                original_maa_item = maa_item
                                                break

                                    if original_maa_item and 'type' in original_maa_item:
                                        item['type'] = original_maa_item['type']
                                        print(f"  -> INFO: Recovered missing 'type' ('{item['type']}') for MenAtArm correction on review_id {req_identifier.get('__review_id__')}.")
                                    else:
                                        has_invalid_suggestion = True
                                        invalid_suggestions_list.append({
                                            'tag': tag, 'identifier': item['identifier'], 'current_unit': item['current_unit'],
                                            'suggested_unit': suggested_unit, 'error': "Correction for MenAtArm is missing the required 'type' field and it could not be recovered."
                                        })
                                        continue

                                if suggested_unit not in self.all_units:
                                    has_invalid_suggestion = True
                                    invalid_suggestions_list.append({
                                        'tag': tag, 'identifier': item['identifier'], 'current_unit': item['current_unit'],
                                        'suggested_unit': suggested_unit, 'error': 'Unit does not exist in the global pool.'
                                    })
                                    continue

                                if suggested_unit in self.excluded_units_set:
                                    has_invalid_suggestion = True
                                    invalid_suggestions_list.append({
                                        'tag': tag, 'identifier': item['identifier'], 'current_unit': item['current_unit'],
                                        'suggested_unit': suggested_unit, 'error': 'Unit is globally excluded.'
                                    })
                                    continue

                                is_generic_tag = tag in ['General', 'Knights', 'Levies', 'Garrison']
                                local_pool = request.get('local_unit_pool', set())
                                if is_generic_tag and suggested_unit not in local_pool:
                                    has_invalid_suggestion = True
                                    invalid_suggestions_list.append({
                                        'tag': tag, 'identifier': item['identifier'], 'current_unit': item['current_unit'],
                                        'suggested_unit': suggested_unit, 'error': 'Unit is not in the faction\'s local unit pool for a generic role.'
                                    })
                                    continue

                                valid_corrections.append(item)
                            else:
                                has_invalid_suggestion = True
                                invalid_suggestions_list.append({
                                    'tag': item.get('tag', 'N/A'), 'identifier': item.get('identifier', {}),
                                    'current_unit': item.get('current_unit', 'N/A'), 'suggested_unit': item.get('suggested_unit', 'N/A'),
                                    'error': f'Malformed correction item: {item}'
                                })

                        if has_invalid_suggestion:
                            result_data["reason"] = "LLM suggested one or more invalid units or provided malformed correction items."
                            result_data["invalid_suggestions"] = invalid_suggestions_list
                        else:
                            result_data["status"] = "success"
                            result_data["reason"] = "Successfully processed."
                            result_data["corrections"] = valid_corrections
                            break # Success, exit retry loop
                    else:
                        result_data["reason"] = "LLM response 'corrections' key is not a list."
                elif response_data is not None:
                    result_data["reason"] = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    result_data["reason"] = "Could not find or parse JSON block in LLM review response."

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: Roster review failed for {req_id} on attempt {attempt + 1}: {result_data['reason']}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        return req_id, result_data

    def get_culture_review(self, request, time_period_context):
        """
        Gets a culture-to-faction assignment review for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        req_id = request['id']
        result_data = {}

        for attempt in range(MAX_RETRIES):
            result_data = {
                "corrections": [],
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "failure",
                "reason": "Unknown error during processing."
            }

            prompt = self._build_culture_review_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt, use_review_config=True)

            if not response_content:
                result_data["reason"] = "LLM call failed or returned no content."
            else:
                response_data = shared_utils.parse_llm_json_response(response_content, req_id)
                if isinstance(response_data, dict) and "corrections" in response_data:
                    corrections = response_data.get("corrections", [])
                    if isinstance(corrections, list):
                        # Basic validation of corrections
                        valid_corrections = []
                        has_invalid_suggestion = False
                        for item in corrections:
                            if isinstance(item, dict) and 'heritage_name' in item and 'current_faction' in item and 'suggested_faction' in item:
                                valid_corrections.append(item)
                            else:
                                has_invalid_suggestion = True
                                result_data["reason"] = f"LLM provided malformed correction item: {item}"
                                break
                        
                        if not has_invalid_suggestion:
                            result_data["status"] = "success"
                            result_data["reason"] = "Successfully processed."
                            result_data["corrections"] = valid_corrections
                            break # Success
                    else:
                        result_data["reason"] = "LLM response 'corrections' key is not a list."
                else:
                    result_data["reason"] = "Could not find or parse JSON block with 'corrections' key in LLM review response."

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: Culture review failed for {req_id} on attempt {attempt + 1}: {result_data['reason']}. Retrying...")
                time.sleep(2 * (2 ** attempt) + random.uniform(0, 1))

        return req_id, result_data

    def _build_unit_replacement_prompt(self, request, time_period_context):
        """Builds a detailed prompt for a single unit replacement request."""
        prompt = f"You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. Your task is to select the best possible unit replacement.\n\n"
        prompt += f"**Faction:** {request.get('faction', 'N/A')}\n"
        prompt += f"**Subculture:** {request.get('subculture', 'N/A')}\n"
        if request.get('prefix'):
            prompt += f"**Cultural Unit Prefix:** {request.get('prefix')}\n"
        if request.get('graph_context'):
            prompt += f"**Relational Context:** {request.get('graph_context')}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        if request.get('maa_type'):
            prompt += f"**Unit Role to Fill:** Men-at-Arms, specifically '{request.get('maa_type', 'N/A')}'\n"
            prompt += f"**Expected Attila Unit Classes:** {request.get('expected_attila_classes', 'N/A')}\n\n"
        else:
            prompt += f"**Unit Role to Fill:** {request.get('tag_name', 'N/A')} (Rank: {request.get('rank', 'N/A')}, Level: {request.get('level', 'N/A')})\n"
            prompt += f"**Role Description:** {request.get('unit_role_description', 'N/A')}\n\n"

        prompt += "**Candidate Units (Prioritized by Cultural Relevance):**\n"
        prompt += "Your choice should come from the highest-priority tier (e.g., 'Tier 1') that contains a suitable unit. The tiers represent cultural closeness, from the faction's own units to those of its broader culture.\n\n"

        MAX_CANDIDATES_PER_TIER = 30
        prioritized_candidates = request.get('prioritized_candidates', {})
        if prioritized_candidates:
            for tier_name, unit_list in prioritized_candidates.items():
                prompt += f"--- {tier_name} ---\n"
                if unit_list:
                    displayed_units = unit_list[:MAX_CANDIDATES_PER_TIER]
                    for unit in displayed_units:
                        tier = self.unit_to_tier_map.get(unit, 'N/A')
                        name = self.unit_to_screen_name_map.get(unit, 'N/A')
                        desc = self.unit_to_description_map.get(unit, '')
                        desc_text = f" | Desc: {desc}" if desc else ""
                        prompt += f"- {unit} (Tier: {tier}, Name: {name}{desc_text})\n"
                    if len(unit_list) > MAX_CANDIDATES_PER_TIER:
                        prompt += f"... and {len(unit_list) - MAX_CANDIDATES_PER_TIER} more units in this tier.\n"
                else:
                    prompt += "None\n"
                prompt += "\n"
        else:
            prompt += "No candidate units available.\n"

        prompt += "\n**Instructions:**\n"
        prompt += "1. Analyze the faction, role, and prioritized candidate units.\n"
        prompt += "2. Select the single best unit from the provided lists. Prioritize higher cultural tiers (e.g., 'Tier 0' or 'Tier 1').\n"
        prompt += "3. **Prefix Guidance**: If a 'Cultural Unit Prefix' is provided, you MUST strongly prefer units that start with that prefix, as they are the most culturally appropriate.\n"
        prompt += "4. **Semantic Guidance**: If 'Tier 0: Semantic Search Results' is present, these are high-confidence suggestions from a vector database. You should strongly prefer a unit from this tier if it fits the role.\n"
        prompt += "5. **Quality Guidance**: For quality, use the numerical `Tier` provided in parentheses for each unit: low numbers (e.g., Tier 1) are for levies/garrisons, while high numbers (e.g., Tier 3) are for generals/knights.\n"
        prompt += "6. Provide your answer in a JSON block with the key 'chosen_unit'. Example: {\"chosen_unit\": \"att_inf_frankish_axemen\"}\n"
        prompt += "7. IMPORTANT: Your answer must come from the above list of units.\n"
        prompt += "8. If no unit is suitable at all, provide null. Example: {\"chosen_unit\": null}\n"
        prompt += "Your response must contain nothing but the JSON block.\n"

        return prompt

    def _build_levy_composition_prompt(self, request, time_period_context):
        """Builds a detailed prompt for generating a levy composition with specific unit keys."""
        faction_name = request.get('faction', 'N/A')
        subculture = request.get('subculture', 'N/A')

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to design a culturally and thematically appropriate levy roster for a faction by selecting specific unit keys.\n\n"
        )
        prompt += f"**Faction:** {faction_name}\n"
        prompt += f"**Subculture:** {subculture}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Candidate Units (Prioritized by Cultural Relevance):**\n"
        prompt += "Levies represent peasant conscripts. You MUST select LOW-QUALITY, poorly-equipped units (e.g., basic spearmen, light infantry, peasants). Elite units, knights, or professional soldiers are forbidden.\n\n"

        prioritized_candidates = request.get('prioritized_candidates', {})
        if prioritized_candidates:
            for tier_name, unit_list in prioritized_candidates.items():
                prompt += f"--- {tier_name} ---\n"
                if unit_list:
                    for unit in unit_list:
                        name = self.unit_to_screen_name_map.get(unit, 'N/A')
                        desc = self.unit_to_description_map.get(unit, '')
                        desc_text = f" | Desc: {desc}" if desc else ""
                        prompt += f"- {unit} ({name}{desc_text})\n"
                else:
                    prompt += "None\n"
                prompt += "\n"
        else:
            prompt += "No candidate units available.\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the faction's cultural context and the candidate units.\n"
            "2.  Select 3 to 5 unique unit keys from the provided lists to form a balanced levy roster (e.g., some spearmen, some archers, some light melee).\n"
            "3.  Assign an integer percentage to each selected unit. The percentages MUST sum to exactly 100.\n"
            "4.  Prioritize units from higher cultural tiers (e.g., 'Tier 1') and with a low numerical tier (e.g., Tier 1) that fit the 'low-quality conscript' role.\n"
            "5.  Provide your answer in a JSON block with the key 'chosen_composition' as a list of objects.\n"
            "Example: {\"chosen_composition\": [{\"key\": \"att_west_levy_spearmen\", \"percentage\": 40}, {\"key\": \"att_west_peasants\", \"percentage\": 60}]}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_roster_review_prompt(self, request, time_period_context, unit_to_tier_map=None):
        """Builds a detailed prompt for a single roster review request."""
        faction_name = request['faction']
        roster = request['roster']
        local_pool = request['local_unit_pool']
        tiered_pools = request.get('tiered_pools', [])
        prefix = request.get('prefix')

        # Build the strictly cultural pool from the first two tiers (faction-specific and subculture-specific)
        strictly_cultural_pool = set()
        if len(tiered_pools) > 0:
            strictly_cultural_pool.update(tiered_pools[0])
        if len(tiered_pools) > 1:
            strictly_cultural_pool.update(tiered_pools[1])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to review a faction's military roster for cultural and thematic consistency and suggest improvements. "
            "You must follow the rules provided precisely.\n\n"
        )

        # NEW: Add feedback section for retry attempts
        if 'retry_context' in request and request['retry_context'].get('previous_errors'):
            prompt += "**Feedback on Your Previous Attempt:**\n"
            prompt += "The following issues were found in your last response. Please correct them in this attempt:\n"
            for error in request['retry_context']['previous_errors']:
                error_reason = error.get('error', 'No specific reason provided.')
                # NEW: Handle generic vs specific errors
                if 'tag' in error and 'current_unit' in error:
                    tag = error.get('tag', 'N/A')
                    current_unit = error.get('current_unit', 'N/A')
                    suggested_unit = error.get('suggested_unit', 'N/A')
                    prompt += f"- You suggested '{suggested_unit}' for the unit '{current_unit}' (tag: '{tag}'). This was incorrect. Reason: {error_reason}\n"
                else:
                    # Generic error
                    prompt += f"- {error_reason}\n"
            prompt += "Please adhere strictly to the provided unit pools and rules in this new attempt. Do not suggest units that are not in the specified pools or do not exist.\n\n"
        # END NEW

        prompt += f"**Faction to Review:** {faction_name}\n"
        prompt += f"**Cultural/Heritage Context:** {request.get('subculture', 'N/A')}\n"
        if prefix:
            prompt += f"**Cultural Unit Prefix:** {prefix}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        # Add bad example to guide the LLM
        prompt += """**Example of a Bad Roster and Desired Correction:**
Consider a Frankish faction with these incorrect `MenAtArm` and `Levies` assignments:
```json
{
  "MenAtArm": [
    {
      "current_unit": "att_fact_visigoths_royal_lancers",
      "identifier": {"__review_id__": "201"},
      "type": "heavy_infantry"
    }
  ],
  "Levies": [
    {"current_unit": "att_cantaber_arqueros", "identifier": {"__review_id__": "101"}}
  ]
}
```
This is incorrect. The `MenAtArm` unit is cavalry but the role is `heavy_infantry`. The `Levies` unit is not culturally appropriate.
The correct correction JSON should be:
```json
{
  "corrections": [
    {
      "tag": "MenAtArm",
      "identifier": {"__review_id__": "201"},
      "type": "heavy_infantry",
      "current_unit": "att_fact_visigoths_royal_lancers",
      "suggested_unit": "att_fact_franks_royal_anstrutiones",
      "reason": "Role mismatch. Replaced cavalry with a culturally appropriate heavy infantry unit."
    },
    {
      "tag": "Levies",
      "identifier": {"__review_id__": "101"},
      "current_unit": "att_cantaber_arqueros",
      "suggested_unit": "att_west_levy_spearmen",
      "reason": "Cultural mismatch. Replace Iberian unit with culturally appropriate Frankish levy."
    }
  ]
}
```
Always follow this pattern for corrections, ensuring cultural appropriateness.\n\n"""

        prompt += "**Faction Roster:**\n"
        prompt += "The roster is provided as a JSON object. Each key is a unit tag, and the value is a list of assigned units. Each unit object includes its key, an `identifier` for its specific slot, and for `MenAtArm`, a `type`.\n"
        prompt += f"```json\n{json.dumps(roster, indent=2)}\n```\n\n"

        prompt += (
            """**Review Rules & Unit Role Definitions:**
1.  **Analyze each unit based on its tag and role:**
    - `General`: This is the commander's elite bodyguard. It MUST be one of the BEST heavy cavalry or elite infantry units available in the 'Local Unit Pool'.
    - `Knights`: These represent the faction's premier, noble heavy cavalry. They MUST be the BEST available shock cavalry in the 'Local Unit Pool'.
    - `Levies`: These are peasant conscripts. They MUST be LOW-QUALITY, poorly-equipped units like basic spearmen or light infantry. They MUST be selected from the 'Strictly Cultural Unit Pool'. **Elite units, knights, or professional soldiers are absolutely forbidden for this role. IT IS A FAILURE IF YOU DO NOT CORRECT THIS. Siege engines (catapults, onagers, etc.) and artillery are also strictly forbidden.**
    - `Garrison`: These are defensive troops. They MUST be mid-to-low quality spearmen or infantry. They MUST be selected from the 'Strictly Cultural Unit Pool'. There MUST be a clear progression in quality, with level 4 garrisons having better units than level 3, level 3 better than level 2, and so on. Elite units are forbidden.
    - `MenAtArm` (Generic types like `heavy_infantry`, `archers`): These are professional soldiers. They MUST be culturally appropriate units from the 'Local Unit Pool' that fit the role.
    - `MenAtArm` (Thematic/Exotic types like `war_elephant`, `camel_cavalry`): These are an EXCEPTION. They MUST be the BEST POSSIBLE THEMATIC MATCH from the ENTIRE `Global Unit Pool`, even if that unit is not in the faction's `Local Unit Pool`.

2.  **Cultural Consistency:** For all tags EXCEPT thematic/exotic Men-At-Arms, the suggested unit MUST exist in the 'Local Unit Pool'. For `Levies` and `Garrison`, they MUST exist in the more restrictive 'Strictly Cultural Unit Pool'. If a unit is not in the correct pool, it is culturally inappropriate and MUST be replaced. FAILURE TO ENFORCE THIS RULE IS A CRITICAL ERROR.

3.  **Do Not Change Correct Units:** If a unit assignment is already correct according to these rules (both role and cultural fit), do not suggest a change for it.

4.  **Unit Diversity:** For each faction, ensure units are unique where appropriate:
    - `General` and `Knights`: Each rank MUST have a different unit, representing a progression of quality.
    - `Levies`: All levy units MUST be different from each other.
    - `Garrison`: All garrison units for the same fortification level MUST be different.
    - Replace duplicates with other suitable units from the appropriate pool.

5.  **Prefix Enforcement:** If a 'Cultural Unit Prefix' is provided, all units (except for thematic/exotic MenAtArms) SHOULD start with this prefix. If a unit does not match the prefix and is not an obvious thematic outlier (like a unique hero unit), it MUST be replaced with a unit from the pools that DOES match the prefix.
"""
        )

        prompt += "**Strictly Cultural Unit Pool (for Levies and Garrisons):**\n"
        prompt += "For `Levies` and `Garrison` roles, you MUST select units from this list, which contains only the most culturally relevant units for the faction. Each unit is listed with its numerical quality Tier (1 is lowest, 3 is highest). FAILURE TO USE UNITS FROM THIS POOL FOR LEVIES/GARRISONS IS A CRITICAL ERROR.\n"
        # Format with tier information
        pool_with_details = []
        for unit in sorted(list(strictly_cultural_pool)):
            details = []
            if unit_to_tier_map:
                details.append(f"Tier: {unit_to_tier_map.get(unit, 'N/A')}")
            if self.unit_to_description_map.get(unit):
                details.append(f"Desc: {self.unit_to_description_map.get(unit)}")
            pool_with_details.append(f"{unit} ({' | '.join(details)})")
        pool_with_tiers = pool_with_details
        prompt += f"```\n{json.dumps(pool_with_tiers, indent=2)}\n```\n\n"

        prompt += "**Local Unit Pool (for General, Knights, and generic MenAtArm):**\n"
        prompt += "Each unit is listed with its numerical quality Tier (1 is lowest, 3 is highest).\n"
        # Format with tier information
        local_pool_with_details = []
        for unit in sorted(list(local_pool)):
            details = []
            if unit_to_tier_map:
                details.append(f"Tier: {unit_to_tier_map.get(unit, 'N/A')}")
            if self.unit_to_description_map.get(unit):
                details.append(f"Desc: {self.unit_to_description_map.get(unit)}")
            local_pool_with_details.append(f"{unit} ({' | '.join(details)})")
        local_pool_with_tiers = local_pool_with_details
        prompt += f"```\n{json.dumps(local_pool_with_tiers, indent=2)}\n```\n\n"

        prompt += "**Global Unit Pool (for Thematic Consistency):**\n"
        prompt += f"A complete list of all {len(self.all_units)} available units in the game is available for your reference.\n\n"

        prompt += (
            "**Output Format:**\n"
            "Provide your answer ONLY as a single JSON object inside a ```json code block. "
            "The object should contain a single key, `corrections`, which is a list of objects. "
            "Each object in the list represents one unit that needs to be changed and must have the following keys:\n"
            "- `tag`: The XML tag of the unit (e.g., 'MenAtArm', 'General').\n"
            "- `identifier`: The exact `identifier` object provided for that unit in the input roster. This is crucial for mapping.\n"
            "- `current_unit`: The unit key currently assigned.\n"
            "- `suggested_unit`: The new unit key you are suggesting as a replacement.\n"
            "- `reason`: A brief explanation for the change (e.g., 'Cultural mismatch.', 'Better thematic fit available.').\n"
            "- **For `MenAtArm` corrections, you MUST also include the `type` key from the input roster.**\n"
            "If no corrections are needed, provide an empty list: `{\"corrections\": []}`.\n"
        )
        return prompt

    def _build_culture_review_prompt(self, request, time_period_context):
        """Builds a detailed prompt for a single culture-faction review request."""
        assignments = request['assignments']
        faction_pool = request['faction_pool']

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to review a list of Crusader Kings 3 (CK3) culture-to-Attila faction assignments for thematic and cultural consistency.\n\n"
        )
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Assignments to Review:**\n"
        prompt += "The assignments are provided as a JSON object. Each key is a CK3 heritage, which contains a list of its member cultures and their assigned Attila factions.\n"
        prompt += f"```json\n{json.dumps(assignments, indent=2)}\n```\n\n"

        prompt += "**List of All Available Attila Factions:**\n"
        prompt += f"```json\n{json.dumps(faction_pool, indent=2)}\n```\n\n"

        prompt += (
            """**Review Rules:**
1.  **Analyze each heritage and culture.** The assigned Attila faction should be the closest possible cultural and thematic match.
2.  **Heritage Faction:** The `heritage_faction` should represent the most common or dominant culture within that heritage.
3.  **Culture Faction:** The `faction` for each culture should be a good match for that specific culture. It can be the same as the heritage faction or a more specific one if available.
4.  **Consistency:** Factions should be consistent. For example, all Norse cultures should generally be assigned to Viking/Norse factions, not Greek factions.
5.  **Do Not Change Correct Assignments:** If an assignment is already correct, do not suggest a change for it.

**Output Format:**
Provide your answer ONLY as a single JSON object inside a ```json code block.
The object should contain a single key, `corrections`, which is a list of objects.
Each object in the list represents one assignment that needs to be changed and must have the following keys:
- `heritage_name`: The name of the heritage containing the item to change.
- `culture_name`: (Optional) The name of the specific culture to change. If omitted, you are changing the `heritage_faction`.
- `current_faction`: The faction currently assigned.
- `suggested_faction`: The new, correct faction name from the 'Available Attila Factions' list.
- `reason`: A brief explanation for the change.

Example for changing a heritage faction:
`{"heritage_name": "byzantine_heritage", "current_faction": "att_fact_franks", "suggested_faction": "att_fact_ere", "reason": "Franks are not culturally appropriate for Byzantine heritage."}`

Example for changing a specific culture's faction:
`{"heritage_name": "west_germanic_heritage", "culture_name": "saxon", "current_faction": "att_fact_goths", "suggested_faction": "att_fact_saxons", "reason": "A more specific Saxon faction is available."}`

If no corrections are needed, provide an empty list: `{\"corrections\": []}`.
"""
        )
        return prompt


    def _build_naval_key_assignment_prompt(self, request, time_period_context):
        """Builds a detailed prompt for a single naval key assignment request."""
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to select the best possible naval transport unit for a given land unit.\n\n"
        )
        prompt += f"**Faction:** {request.get('faction', 'N/A')}\n"
        prompt += f"**Subculture:** {request.get('subculture', 'N/A')}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"
        prompt += f"**Land Unit to Find Transport For:** {request.get('land_unit_key', 'N/A')}\n"
        prompt += f"**Role of Land Unit:** {request.get('unit_role_description', 'N/A')}\n\n"

        prompt += "**Available Naval Transport Units:**\n"
        available_units = request.get('available_naval_units', [])
        if available_units:
            for unit in sorted(available_units):
                tier = self.unit_to_tier_map.get(unit, 'N/A')
                name = self.unit_to_screen_name_map.get(unit, 'N/A')
                desc = self.unit_to_description_map.get(unit, '')
                desc_text = f" | Desc: {desc}" if desc else ""
                prompt += f"- {unit} (Tier: {tier}, Name: {name}{desc_text})\n"
        else:
            prompt += "None\n"

        prompt += (
            "\n**Instructions:**\n"
            "1. Analyze the land unit and its role.\n"
            "2. Select the single best naval transport unit from the available list that is the most thematically appropriate match (e.g., a transport with spearmen for a land spearman unit).\n"
            "3. Provide your answer in a JSON block with the key 'chosen_naval_key'. Example: {\"chosen_naval_key\": \"att_shp_transport_spearmen_a\"}\n"
            "4. IMPORTANT: Your answer must come from the above list of naval transport units.\n"
            "5. If no unit is suitable, provide null. Example: {\"chosen_naval_key\": null}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_naval_roster_prompt(self, request, time_period_context):
        """Builds a detailed prompt for generating a naval roster."""
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to create a culturally and thematically appropriate naval warship roster for a faction.\n\n"
        )
        prompt += f"**Faction:** {request.get('faction', 'N/A')}\n"
        prompt += f"**Subculture:** {request.get('subculture', 'N/A')}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Available Warships (Categorized):**\n"
        categorized_pool = request.get('categorized_naval_pool', {})
        if categorized_pool:
            for category, units in categorized_pool.items():
                prompt += f"--- {category.replace('_', ' ').title()} ---\n"
                for unit in sorted(units):
                    tier = self.unit_to_tier_map.get(unit, 'N/A')
                    name = self.unit_to_screen_name_map.get(unit, 'N/A')
                    desc = self.unit_to_description_map.get(unit, '')
                    desc_text = f" | Desc: {desc}" if desc else ""
                    prompt += f"- {unit} (Tier: {tier}, Name: {name}{desc_text})\n"
                prompt += "\n"
        else:
            prompt += "No warships available.\n"

        prompt += (
            "\n**Instructions:**\n"
            "1. Analyze the faction's culture and the available warships.\n"
            "2. Create a balanced roster with a variety of ships. The roster MUST include at least one 'light_ship', one 'medium_ship', and one 'heavy_ship' if available in the pools.\n"
            "3. Provide your answer as a JSON object with a single key, `chosen_roster`, which is a list of ship objects.\n"
            "4. Each ship object in the list must have the keys: `type` (always 'warship'), `category`, `class`, and `naval_key`.\n"
            "5. The `category` and `class` values must be taken from the ship's data. You can infer this from the unit names (e.g., 'att_shp_hvy_ram_a' is heavy, ramming ship).\n"
            "Example: {\"chosen_roster\": [{\"type\": \"warship\", \"category\": \"light_ship\", \"class\": \"assault\", \"naval_key\": \"att_shp_lt_assault_a\"}, {\"type\": \"warship\", \"category\": \"medium_ship\", \"class\": \"missile\", \"naval_key\": \"att_shp_med_missile_a\"}]}\n"
            "6. If no suitable roster can be created, provide null. Example: {\"chosen_roster\": null}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt


    def _build_faction_name_match_prompt(self, request, time_period_context):
        """Builds a detailed prompt for matching an ambiguous faction name to a valid one."""
        ambiguous_name = request.get('ambiguous_name', 'N/A')
        valid_names = request.get('valid_names', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to match an ambiguous or potentially misspelled faction name to the correct canonical name from a provided list. "
            "You must follow the rules provided precisely.\n\n"
        )
        prompt += f"**Ambiguous Faction Name to Match:** {ambiguous_name}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**List of Valid Canonical Faction Names:**\n"
        if valid_names:
            # Limit the number of names sent to the LLM to prevent excessively long prompts
            MAX_NAMES_TO_SEND = 200
            if len(valid_names) > MAX_NAMES_TO_SEND:
                # If there are too many, send a random sample
                sampled_names = random.sample(valid_names, MAX_NAMES_TO_SEND)
                prompt += f"(Showing a random sample of {MAX_NAMES_TO_SEND} out of {len(valid_names)} total names)\n"
                for name in sorted(sampled_names):
                    prompt += f"- {name}\n"
            else:
                for name in sorted(valid_names):
                    prompt += f"- {name}\n"
        else:
            prompt += "None\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the ambiguous name and the list of valid names.\n"
            "2.  Find the single best match in the 'Valid Canonical Faction Names' list that corresponds to the 'Ambiguous Faction Name'.\n"
            "3.  Consider variations in spelling, word order, and abbreviations.\n"
            "4.  Note: Multiple different ambiguous names can be mapped to the same canonical faction name if they represent the same entity.\n"
            "5.  If you are confident in a match, provide it exactly as it appears in the list.\n"
            "6.  If there is no good match, or you are uncertain, respond with null.\n"
            "7.  Provide your answer in a JSON block with the key 'best_match'. Example: {\"best_match\": \"Frankish\"}\n"
            "8.  Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    # NEW: Subculture assignment prompt
    def _build_subculture_assignment_prompt(self, request, time_period_context):
        """Builds a detailed prompt for assigning a subculture to a faction."""
        faction_name = request.get('faction', 'N/A')
        available_subcultures = request.get('available_subcultures', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to assign the most culturally and thematically appropriate subculture to a faction. "
            "You must follow the rules provided precisely.\n\n"
        )
        prompt += f"**Faction to Assign Subculture:** {faction_name}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Available Subcultures:**\n"
        if available_subcultures:
            for subculture in sorted(available_subcultures):
                prompt += f"- {subculture}\n"
        else:
            prompt += "None (This indicates a problem, please choose a generic subculture like 'roman' or 'barbarian').\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the faction's name and the time period context.\n"
            "2.  Select the single best subculture from the 'Available Subcultures' list that is the closest thematic and cultural match for the faction.\n"
            "3.  You MUST choose a subculture from the provided list. Do not invent a new one. If no match is perfect, choose the most plausible option.\n"
            "4.  Provide your answer in a JSON block with the key 'chosen_subculture'. Example: {\"chosen_subculture\": \"roman\"}\n"
            "5.  You MUST NOT return null. Always select the best possible fit from the list.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    # NEW: Settlement assignment prompt
    def _build_settlement_assignment_prompt(self, request, time_period_context):
        """Builds a detailed prompt for assigning a settlement preset to a faction."""
        faction_name = request.get('faction_name', 'N/A')
        subculture = request.get('subculture', 'N/A')
        preset_pool = request.get('preset_pool', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'. "
            "Your task is to assign the most culturally and thematically appropriate settlement architectural style to a faction. "
            "You must follow the rules provided precisely.\n\n"
        )
        prompt += f"**Faction to Assign Settlement Style:** {faction_name}\n"
        prompt += f"**Cultural Context (Subculture):** {subculture}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Available Settlement Presets (Architectural Styles):\n"
        if preset_pool:
            # Show a limited, representative sample to the LLM to avoid huge prompts
            sample_size = 100
            if len(preset_pool) > sample_size:
                prompt += f"(Showing a sample of {sample_size} out of {len(preset_pool)} total presets)\n"
                sample_pool = preset_pool[:sample_size]
            else:
                sample_pool = preset_pool

            for preset in sorted(sample_pool):
                prompt += f"- {preset}\n"
        else:
            prompt += "None\n"

        prompt += (
            "\n**Instructions:**\n"
            "1.  Analyze the faction's name, cultural context, and the available presets.\n"
            "2.  The presets represent different architectural styles (e.g., 'att_western_city_...', 'att_nordic_town_...', 'att_desert_village_...').\n"
            "3.  Select the single best preset from the 'Available Settlement Presets' list that is the closest architectural and cultural match for the faction.\n"
            "4.  You MUST choose a preset from the provided list. Do not invent a new one. If no match is perfect, choose the most plausible option.\n"
            "5.  Provide your answer in a JSON block with the key 'chosen_preset'. Example: {\"chosen_preset\": \"att_western_city_a_minor\"}\n"
            "6.  You MUST NOT return null. Always select the best possible fit from the list.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_land_bridge_assignment_prompt(self, request, time_period_context):
        """Builds a prompt for assigning a land bridge preset to a CK3 adjacency."""
        crossing_name = request.get('crossing_name', 'N/A')
        crossing_type = request.get('crossing_type', 'N/A')
        preset_pool = request.get('preset_pool', [])
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to select the most thematically appropriate battle map preset for a land crossing.\n\n"
            f"**Crossing Name:** {crossing_name}\n"
            f"**Crossing Type:** {crossing_type} (e.g., strait, river_large)\n"
            f"**Time Period Context:** {time_period_context}\n\n"
            "**Available Land Bridge Presets:**\n"
            f"{'- ' + '\\n- '.join(sorted(preset_pool))}\n\n"
            "**Instructions:**\n"
            "1. Analyze the crossing's name and type.\n"
            "2. Select the single best preset from the 'Available Land Bridge Presets' list that is the closest thematic match.\n"
            "3. Provide your answer in a JSON block with the key 'chosen_preset'. Example: {\"chosen_preset\": \"land_bridge_strait_a\"}\n"
            "4. You MUST choose a preset from the provided list. Do not invent a new one.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_coastal_battle_assignment_prompt(self, request, time_period_context):
        """Builds a prompt for assigning a coastal battle preset to a CK3 sea adjacency."""
        crossing_name = request.get('crossing_name', 'N/A')
        preset_pool = request.get('preset_pool', [])
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to select the most thematically appropriate battle map preset for a sea crossing or coastal region.\n\n"
            f"**Crossing/Region Name:** {crossing_name}\n"
            f"**Time Period Context:** {time_period_context}\n\n"
            "**Available Coastal Battle Presets:**\n"
            f"{'- ' + '\\n- '.join(sorted(preset_pool))}\n\n"
            "**Instructions:**\n"
            "1. Analyze the region's name.\n"
            "2. Select the single best preset from the 'Available Coastal Battle Presets' list that is the closest thematic match.\n"
            "3. Provide your answer in a JSON block with the key 'chosen_preset'. Example: {\"chosen_preset\": \"coastal_battle_mediterranean_a\"}\n"
            "4. You MUST choose a preset from the provided list. Do not invent a new one.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_building_assignment_prompt(self, request, time_period_context):
        """Builds a prompt for assigning a battle preset to a CK3 building."""
        building_key = request.get('id', 'N/A')
        preset_pool = request.get('preset_pool', [])
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to select the most thematically appropriate battle map preset for a unique CK3 building.\n\n"
            f"**CK3 Building Key:** {building_key}\n"
            f"**Time Period Context:** {time_period_context}\n\n"
            "**Available Battle Presets:**\n"
            f"(Showing a sample of 100 out of {len(preset_pool)} total presets)\n"
            f"{'- ' + '\\n- '.join(sorted(random.sample(preset_pool, min(100, len(preset_pool)))))}\n\n"
            "**Instructions:**\n"
            "1. Analyze the building's key (e.g., 'wonder_pyramids_giza', 'building_cathedral_canterbury').\n"
            "2. Select the single best preset from the 'Available Battle Presets' list that is the closest thematic match.\n"
            "3. Provide your answer in a JSON block with the key 'chosen_preset'. Example: {\"chosen_preset\": \"preset_wonder_pyramids\"}\n"
            "4. You MUST choose a preset from the provided list. Do not invent a new one.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def _build_terrain_assignment_prompt(self, request, time_period_context):
        """Builds a prompt for assigning a battle preset to a CK3 terrain type."""
        terrain_key = request.get('id', 'N/A')
        preset_pool = request.get('preset_pool', [])
        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to select the most thematically appropriate battle map preset for a CK3 terrain type.\n\n"
            f"**CK3 Terrain Type:** {terrain_key}\n"
            f"**Time Period Context:** {time_period_context}\n\n"
            "**Available Battle Presets:**\n"
            f"(Showing a sample of 100 out of {len(preset_pool)} total presets)\n"
            f"{'- ' + '\\n- '.join(sorted(random.sample(preset_pool, min(100, len(preset_pool)))))}\n\n"
            "**Instructions:**\n"
            "1. Analyze the terrain's type (e.g., 'plains', 'forest', 'desert_mountains').\n"
            "2. Select the single best preset from the 'Available Battle Presets' list that is the closest thematic match.\n"
            "3. Provide your answer in a JSON block with the key 'chosen_preset'. Example: {\"chosen_preset\": \"preset_grassland_1\"}\n"
            "4. You MUST choose a preset from the provided list. Do not invent a new one.\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt

    def get_faction_name_match(self, request, time_period_context):
        """
        Gets a faction name match for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        best_match = None
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_faction_name_match_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_match = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_match = response_data.get("best_match")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "best_match" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            best_match = temp_match
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            valid_names = request.get('valid_names', [])
            if temp_match is not None:
                if temp_match in valid_names:
                    best_match = temp_match
                    break
                else:
                    failure_reason = f"LLM returned invalid faction name '{temp_match}' (not in valid names list)"
            elif not failure_reason and response_content: # If temp_match is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for faction name match"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        result_data = {"best_match": best_match}
        if best_match is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data

    def get_subculture_assignment(self, request, time_period_context):
        """
        Gets a subculture assignment for a single request from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_SUBCULTURE_RETRIES = 3
        chosen_subculture = None
        for attempt in range(MAX_SUBCULTURE_RETRIES):
            prompt = self._build_subculture_assignment_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_chosen_subculture = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request['id'])
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_chosen_subculture = response_data.get("chosen_subculture")
                        # NEW LOGIC: If we successfully parsed JSON and it contains the key (even if None), break
                        if "chosen_subculture" in response_data:
                            # Valid JSON with explicit key - accept the value (even if None)
                            chosen_subculture = temp_chosen_subculture
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            # Validation (only if we have a non-null value)
            if temp_chosen_subculture is not None:
                available_subcultures = request.get('validation_pool', [])
                if available_subcultures and temp_chosen_subculture in available_subcultures:
                    chosen_subculture = temp_chosen_subculture
                    print(f"  -> SUCCESS: LLM assigned subculture '{chosen_subculture}' for request {request['id']} on attempt {attempt + 1}.")
                    break  # Success, exit retry loop
                else:
                    failure_reason = f"LLM suggested invalid subculture '{temp_chosen_subculture}' (not in available list)"
            elif not failure_reason and response_content: # If temp_chosen_subculture is None but we had valid JSON
                failure_reason = "LLM explicitly returned null for subculture assignment"

            # Log failure and decide whether to retry
            if attempt < MAX_SUBCULTURE_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request['id']} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)  # Exponential backoff with jitter
            else:
                print(f"  -> ERROR: Failed to get a valid subculture for request {request['id']} after {MAX_SUBCULTURE_RETRIES} attempts. Reason: {failure_reason}")

        result_data = {"chosen_subculture": chosen_subculture}
        if chosen_subculture is None:
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request['id'], result_data

    def get_batch_preset_assignments(self, batch, time_period_context, prompt_builder_func, llm_threads=1):
        """
        A generic, parallel function to get preset assignments for a batch of requests.
        It processes requests concurrently using ThreadPoolExecutor.
        """
        if not self.network_calls_enabled:
            print("LLM network calls are disabled. Cannot process preset assignments.")
            return {}

        # Filter requests against cache
        cached_results_full, uncached_requests, cache_modified = self.filter_requests_against_cache(batch)
        if cache_modified:
            self.save_cache()

        # Initialize results with cached entries, keeping the full dictionary structure.
        results = cached_results_full

        # If all requests were cached, return immediately
        if not uncached_requests:
            return results

        print(f"  -> Found {len(results)} valid cached results. {len(uncached_requests)} requests require LLM call.")

        # Process uncached requests in parallel
        network_results = {}
        if uncached_requests:
            print(f"  -> Submitting {len(uncached_requests)} preset assignment requests to LLM using {llm_threads} threads...")
            network_results_list = []
            with ThreadPoolExecutor(max_workers=llm_threads) as executor:
                # Submit all requests to the executor
                future_to_req = {
                    executor.submit(self._get_single_preset_assignment, req, time_period_context, prompt_builder_func): req
                    for req in uncached_requests
                }

                # Process completed futures as they finish
                processed_requests = 0
                total_requests = len(uncached_requests)
                for future in as_completed(future_to_req):
                    processed_requests += 1
                    req = future_to_req[future]
                    print(f"  -> LLM preset assignment progress: {processed_requests}/{total_requests} requests completed.")
                    try:
                        req_id, chosen_preset = future.result()
                        # Wrap the result in a dictionary to match the cache format
                        result_dict = {
                            "chosen_preset": chosen_preset,
                            "timestamp": datetime.datetime.utcnow().isoformat()
                        }
                        network_results_list.append((req_id, result_dict))
                    except Exception as exc:
                        req_id = req['id']
                        print(f"  -> ERROR: A preset assignment request for '{req_id}' generated an exception: {exc}")
                        network_results_list.append((req_id, {"chosen_preset": None})) # Return a dict on error
                        raise exc

            # Convert list of tuples to dictionary
            network_results = dict(network_results_list)

            # Update cache with all new results
            if network_results:
                with self.cache_lock:
                    self.cache.update(network_results)
                self.save_cache()

        # Merge cached and network results
        results.update(network_results)
        return results

    def get_batch_settlement_assignments(self, batch, time_period_context, llm_threads=1):
        """
        Gets settlement assignments for a batch of requests from the LLM.
        """
        llm_results = self.get_batch_preset_assignments(batch, time_period_context, self._build_settlement_assignment_prompt, llm_threads)

        final_results = {}
        for req_id, result_data in llm_results.items():
            chosen_preset = result_data.get("chosen_preset")
            final_results[req_id] = {
                "chosen_preset": chosen_preset,
                "confidence_score": 5 if chosen_preset else 0
            }

        return final_results

    def get_batch_land_bridge_assignments(self, batch, time_period_context, llm_threads=1):
        """Gets land bridge assignments for a batch of requests from the LLM."""
        return self.get_batch_preset_assignments(batch, time_period_context, self._build_land_bridge_assignment_prompt, llm_threads)

    def get_batch_coastal_battle_assignments(self, batch, time_period_context, llm_threads=1):
        """Gets coastal battle assignments for a batch of requests from the LLM."""
        return self.get_batch_preset_assignments(batch, time_period_context, self._build_coastal_battle_assignment_prompt, llm_threads)

    def get_batch_building_assignments(self, batch, time_period_context, llm_threads=1):
        """Gets historic building assignments for a batch of requests from the LLM."""
        return self.get_batch_preset_assignments(batch, time_period_context, self._build_building_assignment_prompt, llm_threads)

    def get_batch_terrain_assignments(self, batch, time_period_context, llm_threads=1):
        """Gets normal terrain assignments for a batch of requests from the LLM."""
        return self.get_batch_preset_assignments(batch, time_period_context, self._build_terrain_assignment_prompt, llm_threads)



    def get_semantic_faction_matches(self, request, time_period_context):
        """
        Gets up to 5 most similar populated factions for an empty faction from the LLM.
        Returns a tuple of (request_id, result_data).
        """
        MAX_RETRIES = 3
        similar_factions = []
        request_id = request['id']

        for attempt in range(MAX_RETRIES):
            prompt = self._build_semantic_faction_match_prompt(request, time_period_context)
            response_content = self._call_llm_with_retry(prompt)

            temp_matches = None
            failure_reason = ""

            if response_content:
                response_data = shared_utils.parse_llm_json_response(response_content, request_id)
                if response_data is not None:
                    if isinstance(response_data, dict):
                        temp_matches = response_data.get("similar_factions")
                        if "similar_factions" in response_data:
                            similar_factions = temp_matches if isinstance(temp_matches, list) else []
                            break
                    else:
                        failure_reason = f"LLM returned unexpected data type: {type(response_data)}"
                else:
                    failure_reason = "Could not find or parse JSON block in LLM response"
            else:
                failure_reason = "LLM call returned no content"

            if attempt < MAX_RETRIES - 1:
                print(f"  -> WARNING: {failure_reason} for request {request_id} on attempt {attempt + 1}. Retrying...")
                current_delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(current_delay)

        result_data = {"similar_factions": similar_factions}
        if not similar_factions: # Handles both null and empty list
            validation_pool = request.get('validation_pool', [])
            pool_hash = hashlib.sha1(','.join(sorted(list(validation_pool))).encode()).hexdigest()
            result_data['pool_hash'] = pool_hash

        result_data["timestamp"] = datetime.datetime.utcnow().isoformat()

        return request_id, result_data

    def _build_semantic_faction_match_prompt(self, request, time_period_context):
        """Builds a prompt for finding similar factions based on culture and heritage."""
        faction_name = request.get('faction_name', 'N/A')
        subculture = request.get('subculture', 'N/A')
        heritage = request.get('heritage', 'N/A')
        populated_factions = request.get('populated_factions', [])

        prompt = (
            "You are an expert assistant for the Total War: Attila modding tool 'Crusader Conflicts'.\n"
            "Your task is to find up to 5 culturally and thematically similar factions to a target faction.\n\n"
        )
        prompt += f"**Target Faction:** {faction_name}\n"
        prompt += f"**Subculture:** {subculture}\n"
        prompt += f"**Heritage:** {heritage}\n"
        prompt += f"**Time Period Context:** {time_period_context}\n\n"

        prompt += "**Available Populated Factions:**\n"
        for f in populated_factions:
            prompt += f"- {f['name']} (Subculture: {f.get('subculture', 'N/A')}, Heritage: {f.get('heritage', 'N/A')})\n"

        prompt += (
            "\n**Instructions:**\n"
            "1. Analyze the target faction's name, subculture, and heritage.\n"
            "2. Select at least 1 and up to 5 best matches from the 'Available Populated Factions' list that are the closest thematic and cultural matches.\n"
            "3. Prioritize matches with the same heritage, then the same subculture.\n"
            "4. Provide your answer in a JSON block with the key 'similar_factions' as a list of names.\n"
            "Example: {\"similar_factions\": [\"Frankish\", \"Saxon\", \"Lombard\", \"Gothic\", \"Alamanni\"]}\n"
            "Your response must contain nothing but the JSON block.\n"
        )
        return prompt
