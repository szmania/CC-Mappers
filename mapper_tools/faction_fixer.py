import os
import argparse
import xml.etree.ElementTree as ET
import random
from collections import defaultdict, Counter
import re
import Levenshtein
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import sys
import datetime
import hashlib

try:
    from lxml import etree as lxml_etree
except ImportError:
    print("Warning: 'lxml' library not found. XML schema validation will be skipped.")
    print("Please install it using 'pip install lxml' for full functionality.")
    lxml_etree = None

from mapper_tools import shared_utils
from mapper_tools import ck3_to_attila_mappings as mappings
from mapper_tools import faction_xml_utils
from mapper_tools import unit_selector
from mapper_tools import unit_management
from mapper_tools import processing_passes
from mapper_tools import faction_json_utils
from mapper_tools import llm_orchestrator

# --- NEW: Constants for LLM processing ---
LLM_MAX_UNITS_PER_BATCH = 200 # Increased from 60 to reduce network calls
LLM_POOL_MIN_SIZE = 15
MAX_LLM_FAILURES_THRESHOLD = 500000 # Threshold for early exit


# --- DELETED: Logging Setup (Moved to shared_utils) ---

def _load_data_in_parallel(tasks_to_run):
    """Helper to run data loading functions in parallel and return their results."""
    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_name = {executor.submit(func, *args): name for name, (func, args) in tasks_to_run.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
                print(f"  -> Loaded {name}")
            except Exception as exc:
                print(f"  -> ERROR loading {name}: {exc}")
                # Re-raise the exception to halt execution if a critical file fails to load
                raise
    return results


def update_subcultures_only(factions_xml_path, llm_helper, time_period_context, llm_threads, llm_batch_size,
                            faction_to_subculture_map, subculture_to_factions_map, screen_name_to_faction_key_map,
                            no_subculture, most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map,
                            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map):
    """
    Runs only the subculture assignment process, using LLM and procedural fallbacks.
    """
    print(f"\n--- Starting Subculture Update Pass for '{factions_xml_path}' ---")
    total_changes = 0

    try:
        tree = ET.parse(factions_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {factions_xml_path}: {e}. Aborting subculture update.")
        raise
    except FileNotFoundError:
        print(f"Error: Factions XML file not found at '{factions_xml_path}'. Aborting subculture update.")
        raise

    # Cache faction elements for single-pass processing
    all_faction_elements = list(root.findall('Faction'))
    faction_by_name_cache = {f.get('name'): f for f in all_faction_elements if f.get('name')}

    llm_subcultures_assigned_count = 0
    if llm_helper and not no_subculture:
        llm_subcultures_assigned_count = llm_orchestrator.run_llm_subculture_pass(
            root, llm_helper, time_period_context, llm_threads, llm_batch_size,
            faction_to_subculture_map, subculture_to_factions_map, screen_name_to_faction_key_map,
            all_faction_elements=all_faction_elements # Pass cached elements
        )
        if llm_subcultures_assigned_count > 0:
            total_changes += llm_subcultures_assigned_count
            print(f"LLM assigned {llm_subcultures_assigned_count} subcultures.")

    # Add subculture attributes (this will now act as a fallback for LLM failures)
    subculture_attr_count = faction_xml_utils.ensure_subculture_attributes(
        root, screen_name_to_faction_key_map, faction_to_subculture_map, no_subculture=no_subculture,
        most_common_faction_key=most_common_faction_key, faction_key_to_screen_name_map=faction_key_to_screen_name_map,
        culture_to_faction_map=culture_to_faction_map,
        faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map,
        faction_to_heritages_map=faction_to_heritages_map,
        all_faction_elements=all_faction_elements # Pass cached elements
    )
    if subculture_attr_count > 0:
        total_changes += subculture_attr_count
        print(f"Procedural fallback assigned/updated {subculture_attr_count} subculture attributes.")

    if total_changes > 0:
        print(f"\nSubculture update pass complete. Applied {total_changes} changes. Saving file...")
        shared_utils.indent_xml(root)
        tree.write(factions_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Successfully updated '{factions_xml_path}'.")
    else:
        print("\nSubculture update pass complete. No changes were made.")

    return total_changes


def _run_initial_xml_cleaning_pass(root, excluded_units_set, all_units):
    """
    Consolidated pass to perform initial cleaning of the Factions XML.
    Combines multiple cleaning steps into a single loop for performance.
    """
    invalid_maa_removed_count = 0
    duplicate_maa_removed_count = 0
    maa_levy_conflicts_fixed = 0
    excluded_keys_removed_count = 0
    stale_keys_removed_count = 0
    porcentage_rename_count = 0

    # Single pass for attribute renaming and key removal on all elements
    for element in root.iter():
        # Rename 'porcentage' to 'percentage'
        if 'porcentage' in element.attrib:
            element.set('percentage', element.attrib['porcentage'])
            del element.attrib['porcentage']
            porcentage_rename_count += 1

        # Remove keys for excluded or stale units
        if 'key' in element.attrib:
            key = element.get('key')
            if excluded_units_set and key in excluded_units_set:
                del element.attrib['key']
                excluded_keys_removed_count += 1
            elif all_units and key not in all_units:
                del element.attrib['key']
                stale_keys_removed_count += 1

    # Single pass over factions for structural cleaning
    for faction in root.findall('Faction'):
        # Remove invalid MenAtArm tags (missing 'type')
        all_maa_tags = faction.findall('MenAtArm')
        valid_maa_tags = []
        for maa in all_maa_tags:
            if 'type' not in maa.attrib or not maa.get('type'):
                faction.remove(maa)
                invalid_maa_removed_count += 1
            else:
                valid_maa_tags.append(maa)

        # Remove duplicate MenAtArm tags (based on 'type')
        seen_maa_types = set()
        # Iterate over a copy as we are modifying the list
        for maa in list(valid_maa_tags):
            maa_type = maa.get('type')
            if maa_type in seen_maa_types:
                faction.remove(maa)
                duplicate_maa_removed_count += 1
            else:
                seen_maa_types.add(maa_type)

        # Fix conflicts where a MenAtArm key is also a Levy key
        levy_keys = {levy.get('key') for levy in faction.findall('Levies') if levy.get('key')}
        if levy_keys:
            for maa in faction.findall('MenAtArm'):
                if 'key' in maa.attrib and maa.get('key') in levy_keys:
                    del maa.attrib['key']
                    maa_levy_conflicts_fixed += 1

    return (invalid_maa_removed_count, duplicate_maa_removed_count, maa_levy_conflicts_fixed,
            excluded_keys_removed_count, stale_keys_removed_count, porcentage_rename_count)


def _run_attribute_management_pass(root, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege):
    """
    Consolidated pass to manage all unit attributes in a single loop.
    Handles 'max', 'siege', 'siege_engine_per_unit', and 'num_guns'.
    """
    siege_attr_count = 0
    siege_engine_per_unit_attr_count = 0
    num_guns_attr_count = 0
    max_attr_count = 0

    for faction in root.findall('Faction'):
        s, se, ng, m = _run_attribute_management_pass_for_faction(
            faction, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege
        )
        siege_attr_count += s
        siege_engine_per_unit_attr_count += se
        num_guns_attr_count += ng
        max_attr_count += m

    return siege_attr_count, siege_engine_per_unit_attr_count, num_guns_attr_count, max_attr_count

def _run_attribute_management_pass_for_faction(faction_element, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege):
    """
    Manages all unit attributes for a single faction element.
    Handles 'max', 'siege', 'siege_engine_per_unit', and 'num_guns'.
    Returns counts of changes made.
    """
    siege_attr_count = 0
    siege_engine_per_unit_attr_count = 0
    num_guns_attr_count = 0
    max_attr_count = 0

    for element in faction_element:
        # --- Manage 'num_guns' for all unit types ---
        if 'key' in element.attrib:
            unit_key = element.get('key')
            if unit_to_num_guns_map and unit_key in unit_to_num_guns_map:
                num_guns = unit_to_num_guns_map[unit_key]
                if element.get('num_guns') != str(num_guns):
                    element.set('num_guns', str(num_guns))
                    num_guns_attr_count += 1
            elif 'num_guns' in element.attrib:
                del element.attrib['num_guns']
                num_guns_attr_count += 1

        # --- Manage attributes for specific tags ---
        if element.tag in ['Levies', 'Garrison']:
            if element.get('max') != 'LEVY':
                element.set('max', 'LEVY')
                max_attr_count += 1

        elif element.tag == 'MenAtArm':
            unit_key = element.get('key')
            maa_definition_name = element.get('type')

            if not unit_key or not maa_definition_name:
                continue

            # --- Determine if it's a siege unit ---
            internal_type = ck3_maa_definitions.get(maa_definition_name)
            is_siege_by_ck3_type = (mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) is None) or \
                                   (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type) is None)
            unit_class = unit_to_class_map.get(unit_key)
            is_siege_by_attila_class = (unit_class == 'art_siege')
            unit_category = unit_categories.get(unit_key)
            is_siege_by_attila_category = (unit_category == 'artillery')
            is_siege = is_siege_by_ck3_type or is_siege_by_attila_class or is_siege_by_attila_category

            # --- Manage 'siege' attribute ---
            if not no_siege and is_siege:
                if element.get('siege') != 'true':
                    element.set('siege', 'true')
                    siege_attr_count += 1
            elif 'siege' in element.attrib:
                del element.attrib['siege']
                siege_attr_count += 1

            # --- Manage 'max' attribute ---
            if is_siege:
                if 'max' in element.attrib:
                    del element.attrib['max']
                    max_attr_count += 1
            else:  # Not a siege unit, must have 'max'
                max_value = mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(maa_definition_name) or \
                            (internal_type and mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY.get(internal_type))
                if not max_value:
                    specific_category = unit_categories.get(unit_key)
                    max_value = mappings.ATTILA_CATEGORY_TO_MAX_VALUE.get(specific_category, "INFANTRY")

                if element.get('max') != max_value:
                    element.set('max', max_value)
                    max_attr_count += 1

            # --- Manage 'siege_engine_per_unit' attribute ---
            if unit_class == 'art_siege':
                if element.get('siege_engine_per_unit') != '1':
                    element.set('siege_engine_per_unit', '1')
                    siege_engine_per_unit_attr_count += 1
            elif 'siege_engine_per_unit' in element.attrib:
                del element.attrib['siege_engine_per_unit']
                siege_engine_per_unit_attr_count += 1

    return siege_attr_count, siege_engine_per_unit_attr_count, num_guns_attr_count, max_attr_count


def process_units_xml(units_xml_path, categorized_units, all_units, general_units, unit_categories,
                      faction_key_to_screen_name_map, unit_to_faction_key_map,
                      template_faction_unit_pool, culture_factions, tier=None, unit_variant_map=None,
                      unit_to_tier_map=None, variant_to_base_map=None, unit_to_training_level=None,
                      ck3_maa_definitions=None, screen_name_to_faction_key_map=None, faction_key_to_units_map=None, submod_tag=None,
                      excluded_factions=None, unit_to_class_map=None, faction_to_subculture_map=None, subculture_to_factions_map=None,
                      culture_to_faction_map=None, unit_to_description_map=None, unit_stats_map=None,
                      faction_culture_map=None, llm_helper=None, excluded_units_set=None, unit_to_num_guns_map=None, llm_batch_size=50, no_siege=False, no_subculture=False, no_garrison=False, most_common_faction_key=None, main_mod_faction_maa_map=None, llm_threads=1,
                      faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, first_pass_threshold=0.90, is_submod_mode=False, submod_addon_tag=None, faction_to_json_map=None, time_period_context="", force_procedural_recache=False): # Added heritage maps and first_pass_threshold, is_submod_mode
    """
    Processes a single Attila Factions XML file to fix and update unit entries.
    """
    total_changes = 0 # Initialize total_changes here
    faction_pool_cache = {} # Initialize the cache for this run
    print(f"\nProcessing file: {units_xml_path}")

    # Define factions_in_main_mod early
    factions_in_main_mod = set()
    if is_submod_mode:
        factions_in_main_mod = set(main_mod_faction_maa_map.keys()) if main_mod_faction_maa_map else set()

    # The file is guaranteed to exist by prompt_to_create_xml in main.
    # The try-except block below will handle parsing errors.
    try:
        tree = ET.parse(units_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {units_xml_path}: {e}. Skipping.")
        raise

    # Cache faction elements for single-pass processing
    all_faction_elements = list(root.findall('Faction'))
    faction_by_name_cache = {f.get('name'): f for f in all_faction_elements if f.get('name')}

    def _recache_factions():
        nonlocal all_faction_elements, faction_by_name_cache
        all_faction_elements = list(root.findall('Faction'))
        faction_by_name_cache = {f.get('name'): f for f in all_faction_elements if f.get('name')}
        print("  -> Faction cache re-populated.")

    # NEW: Add submod_tag if provided
    submod_tag_added = False
    if submod_tag:
        if 'submod_tag' not in root.attrib:
            root.set('submod_tag', submod_tag)
            print(f"\nAdded submod_tag='{submod_tag}' to the root <Factions> element.")
            submod_tag_added = True
        else:
            print(f"\nRoot <Factions> element already has submod_tag='{root.get('submod_tag')}'. No change made.")

    # NEW: Add submod_addon_tag if provided
    submod_addon_for_added = False
    if submod_addon_tag:
        if 'submod_addon_tag' not in root.attrib or root.get('submod_addon_tag') != submod_addon_tag:
            root.set('submod_addon_tag', submod_addon_tag)
            print(f"\nAdded/Updated submod_addon_tag='{submod_addon_tag}' to the root <Factions> element.")
            submod_addon_for_added = True
        else:
            print(f"\nRoot <Factions> element already has submod_addon_tag='{root.get('submod_addon_tag')}'. No change made.")

    # --- NEW: Consolidated Initial XML Cleaning Pass ---
    print("\nRunning initial XML cleaning and validation pass...")
    (invalid_maa_removed_count, duplicate_maa_removed_count, maa_levy_conflicts_fixed,
     excluded_keys_removed_count, stale_keys_removed_count, porcentage_rename_count) = _run_initial_xml_cleaning_pass(root, excluded_units_set, all_units)

    # NEW: Conditionally remove keys from procedurally-assigned units to force re-evaluation
    procedural_keys_removed_count = 0
    if force_procedural_recache:
        procedural_keys_removed_count = faction_xml_utils.conditionally_remove_procedural_keys(
            root, llm_helper, tier, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map, heritage_to_factions_map,
            faction_to_heritages_map, unit_to_class_map
        )

    # First, validate and fix faction names to correct any fuzzy mismatches.
    # This prevents correct factions from being removed due to minor name differences.
    print("\nValidating and fixing faction names...")
    factions_fixed, faction_name_map = faction_xml_utils.validate_and_fix_faction_names(root, faction_key_to_screen_name_map, unit_to_faction_key_map, culture_factions)
    if factions_fixed > 0:
        print(f"Corrected {factions_fixed} faction names via fuzzy matching.")
        _recache_factions() # Re-cache after potential removals/renames
    else:
        print("All faction names are already valid.")

    # NEW: Remove explicitly excluded factions
    removed_excluded_factions_count = 0
    if excluded_factions:
        removed_excluded_factions_count = faction_xml_utils.remove_excluded_factions(root, excluded_factions, screen_name_to_faction_key_map)
        if removed_excluded_factions_count > 0:
            _recache_factions() # Re-cache after removals

    # NEW: Prune factions not in Cultures.xml (NOW runs after name correction)
    factions_removed_count = faction_xml_utils.remove_factions_not_in_cultures(root, culture_factions, screen_name_to_faction_key_map)
    if factions_removed_count > 0:
        _recache_factions() # Re-cache after removals

    # NEW: Prune factions present in main mod that have no new MenAtArm types
    factions_removed_from_main_mod, removed_faction_names_for_sync = 0, set()
    if is_submod_mode:
        factions_removed_from_main_mod, removed_faction_names_for_sync = faction_xml_utils.remove_factions_in_main_mod(root, main_mod_faction_maa_map)
        if factions_removed_from_main_mod > 0:
            _recache_factions() # Re-cache after removals

    default_created = faction_xml_utils.create_default_faction_if_missing(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, unit_to_class_map=unit_to_class_map, unit_to_description_map=unit_to_description_map, unit_stats_map=unit_stats_map, excluded_units_set=excluded_units_set, is_submod_mode=is_submod_mode)
    if default_created > 0:
        _recache_factions() # Re-cache after addition

    factions_added = faction_xml_utils.sync_factions_from_cultures(root, culture_factions, explicitly_removed_factions=removed_faction_names_for_sync)
    if factions_added > 0:
        _recache_factions() # Re-cache after additions

    # NEW: Sync all MenAtArm unit tags from Default to other factions
    # This must happen BEFORE the unit assignment pipeline to ensure all MAA tags exist.
    faction_sync_count = faction_xml_utils.sync_faction_structure_from_default(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, screen_name_to_faction_key_map, faction_key_to_units_map, unit_to_class_map, faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map=culture_to_faction_map, unit_to_description_map=unit_to_description_map, unit_stats_map=unit_stats_map, main_mod_faction_maa_map=main_mod_faction_maa_map, excluded_units_set=excluded_units_set, faction_pool_cache=faction_pool_cache, faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map, faction_to_heritages_map=faction_to_heritages_map)

    # NEW: In submod mode, remove MenAtArm tags that are already defined in the main mod.
    maa_tags_removed_from_submod = 0
    if is_submod_mode:
        maa_tags_removed_from_submod = faction_xml_utils.remove_maa_tags_present_in_main_mod(root, main_mod_faction_maa_map)

    # --- Unit Assignment Pipeline ---
    # Stage 1: High-Confidence Procedural Pass (JSON Overrides + Class-based)
    high_confidence_replacements, high_confidence_failures = processing_passes.run_high_confidence_unit_pass(
        root, tier, unit_variant_map, ck3_maa_definitions, unit_to_class_map, unit_to_description_map,
        screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map,
        subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map,
        faction_culture_map, categorized_units, unit_categories, unit_stats_map, all_units,
        excluded_units_set=excluded_units_set,
        faction_pool_cache=faction_pool_cache,
        faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map, faction_to_heritages_map=faction_to_heritages_map, # Passed heritage maps
        first_pass_threshold=first_pass_threshold, # Pass the new threshold
        llm_helper=llm_helper,
        faction_to_json_map=faction_to_json_map
    )

    # Start collecting all failures for the LLM pass
    all_llm_failures_to_process = list(high_confidence_failures)

    # --- Handle --no-garrison flag up-front ---
    garrison_fix_count = 0
    if no_garrison:
        print("\n--no-garrison flag is set. Removing all <Garrison> tags...")
        tags_removed = 0
        for faction_element in root.findall('Faction'):
            for garrison_element in list(faction_element.findall('Garrison')):
                faction_element.remove(garrison_element)
                tags_removed += 1
        if tags_removed > 0:
            print(f"  - Removed {tags_removed} <Garrison> tags.")
            garrison_fix_count = tags_removed
        else:
            print("  - No <Garrison> tags found to remove.")

    # --- NEW: Single-Pass Faction Processing Loop for Generals, Knights, Levies, Garrisons ---
    ranked_units_count = 0
    levy_fix_count = 0
    # garrison_fix_count is handled above for no_garrison, and will be a faction count otherwise.
    if not no_garrison:
        garrison_fix_count = 0

    # Counters for consolidated operations
    duplicate_levies_fixed_count = 0
    duplicate_garrisons_fixed_count = 0
    duplicate_ranked_units_removed_count = 0
    per_faction_siege_attr_count = 0
    per_faction_siege_engine_per_unit_attr_count = 0
    per_faction_num_guns_attr_count = 0
    per_faction_max_attr_count = 0
    per_faction_subculture_attr_count = 0
    per_faction_excluded_keys_removed_count = 0
    per_faction_levy_garrison_max_fix_count = 0
    per_faction_levy_renormalize_count = 0
    per_faction_garrison_renormalize_count = 0
    per_faction_zero_percent_removed_count = 0
    per_faction_reorganized_count = 0
    per_faction_reordered_attr_count = 0


    print("\n--- Starting Consolidated Faction Processing Loop ---")
    for faction in all_faction_elements: # Use the cached list of faction elements
        faction_name = faction.get('name')
        if faction_name == "Default":
            continue

        # In submod mode, skip factions that are already in the main mod for core unit processing.
        if is_submod_mode and factions_in_main_mod and faction_name in factions_in_main_mod:
            # This faction will be processed by remove_core_unit_tags later if needed.
            continue

        # 1. Manage Generals and Knights for this faction
        units_added, g_k_failures = unit_management._manage_single_faction_generals_and_knights(
            faction, faction_pool_cache, categorized_units, general_units, unit_stats_map, unit_categories,
            screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map,
            subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map, tier,
            unit_to_tier_map, faction_to_json_map, all_units, unit_to_training_level, excluded_units_set,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, faction_culture_map,
            is_submod_mode, factions_in_main_mod
        )
        ranked_units_count += units_added
        all_llm_failures_to_process.extend(g_k_failures)

        # 2. Collect elite units for this faction to exclude from levies
        faction_elite_units = {el.get('key') for el in faction.findall('General')}
        faction_elite_units.update({el.get('key') for el in faction.findall('Knights')})
        faction_elite_units.discard(None)

        # 3. Manage Levies for this faction
        working_pool, log_faction_str, _ = faction_xml_utils.get_cached_faction_working_pool(
            faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Levies)"
        )
        json_group_data = faction_to_json_map.get(faction_name) if faction_to_json_map else None

        changes, levy_failures = unit_management.manage_faction_levies(
            faction, working_pool, unit_to_training_level, unit_categories, log_faction_str, unit_to_class_map,
            json_group_data, all_units, tier, faction_elite_units=faction_elite_units, faction_name=faction_name,
            excluded_units_set=excluded_units_set, destructive_on_failure=True, levy_composition_override=None
        )
        if changes:
            levy_fix_count += 1
        all_llm_failures_to_process.extend(levy_failures)

        # 4. Manage Garrisons for this faction (if not disabled)
        if not no_garrison:
            working_pool_garrison, log_faction_str_garrison, _ = faction_xml_utils.get_cached_faction_working_pool(
                faction_name, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
                faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
                culture_to_faction_map, excluded_units_set,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map, log_prefix="(Garrison)"
            )

            changes, garrison_failures = unit_management.manage_faction_garrisons(
                faction, working_pool_garrison, unit_to_training_level, unit_categories, log_faction_str_garrison,
                general_units, unit_to_class_map, tier, unit_to_tier_map, excluded_units_set=excluded_units_set,
                destructive_on_failure=True, faction_to_heritage_map=faction_to_heritage_map,
                heritage_to_factions_map=heritage_to_factions_map, screen_name_to_faction_key_map=screen_name_to_faction_key_map,
                faction_key_to_units_map=faction_key_to_units_map, json_group_data=json_group_data, all_units=all_units
            )
            if changes:
                garrison_fix_count += 1
            all_llm_failures_to_process.extend(garrison_failures)

        # --- Consolidated Faction-Specific Operations ---

        # 5. Fix duplicate levy units (per faction)
        duplicate_levies_fixed_count += faction_xml_utils._fix_duplicate_levy_units_for_faction(
            faction, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map, unit_to_training_level,
            unit_categories, faction_elite_units
        )

        # 6. Fix duplicate garrison units (per faction)
        duplicate_garrisons_fixed_count += faction_xml_utils._fix_duplicate_garrison_units_for_faction(
            faction, faction_pool_cache, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, excluded_units_set, faction_to_heritage_map,
            heritage_to_factions_map, faction_to_heritages_map, unit_categories, general_units
        )

        # 7. Remove duplicate ranked units (per faction)
        if not is_submod_mode:
            duplicate_ranked_units_removed_count += faction_xml_utils._remove_duplicate_ranked_units_for_faction(faction)

        # 8. Consolidated Attribute Management Pass (per faction)
        s, se, ng, m = _run_attribute_management_pass_for_faction(
            faction, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege
        )
        per_faction_siege_attr_count += s
        per_faction_siege_engine_per_unit_attr_count += se
        per_faction_num_guns_attr_count += ng
        per_faction_max_attr_count += m

        # 9. Ensure subculture attributes (per faction)
        if not no_subculture:
            per_faction_subculture_attr_count += faction_xml_utils._ensure_subculture_attributes_for_faction(
                faction, screen_name_to_faction_key_map, faction_to_subculture_map,
                most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map,
                faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map
            )

        # 10. Remove excluded unit keys (per faction)
        per_faction_excluded_keys_removed_count += faction_xml_utils._remove_excluded_unit_keys_for_faction(faction, excluded_units_set)

        # 11. Final safeguard for max='LEVY' (per faction)
        # This is already handled within _run_attribute_management_pass_for_faction for existing tags.
        # For newly created tags, it's handled at creation.
        # The global scan for './/Levies' and './/Garrison' will catch any remaining.
        # So, no specific per-faction loop needed here.

        # 12. Final re-normalization pass for Levies and Garrisons (per faction)
        if not is_submod_mode:
            levy_tags = faction.findall('Levies')
            if levy_tags:
                if unit_management._normalize_levy_percentages(levy_tags):
                    per_faction_levy_renormalize_count += 1

            if not no_garrison:
                garrisons_by_level = defaultdict(list)
                for g_tag in faction.findall('Garrison'):
                    level = g_tag.get('level')
                    if level:
                        garrisons_by_level[level].append(g_tag)

                faction_had_garrison_change = False
                for level, tags in garrisons_by_level.items():
                    if unit_management._normalize_levy_percentages(tags):
                        faction_had_garrison_change = True
                if faction_had_garrison_change:
                    per_faction_garrison_renormalize_count += 1

        # 13. Remove any remaining zero-percentage tags (per faction)
        per_faction_zero_percent_removed_count += faction_xml_utils._remove_zero_percentage_tags_for_faction(faction)

        # 14. Final Tag and Attribute Reordering (per faction)
        if faction_xml_utils.reorganize_faction_children(faction): # This function already takes a single faction
            per_faction_reorganized_count += 1
        per_faction_reordered_attr_count += faction_xml_utils._reorder_attributes_in_all_tags_for_element(faction)


    # NEW: After populating everyone, strip core tags from factions that are in the main mod.
    core_tags_removed_count = 0
    if is_submod_mode:
        core_tags_removed_count = faction_xml_utils.remove_core_unit_tags(root, factions_in_main_mod)


    # Stage 2: LLM Intervention Pass (Single-Pass Architecture)
    llm_replacements_made, llm_failures = llm_orchestrator.run_llm_unit_assignment_pass(
        llm_helper, all_llm_failures_to_process, time_period_context, llm_threads, llm_batch_size,
        faction_pool_cache, all_units, excluded_units_set, unit_to_tier_map, unit_to_class_map,
        unit_to_description_map, unit_stats_map, screen_name_to_faction_key_map, faction_key_to_units_map,
        faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
        culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map,
        ck3_maa_definitions
    )


    # Stage 3: Low-Confidence Procedural Pass (Role/Category-based)
    low_confidence_replacements = processing_passes.run_low_confidence_unit_pass(
        root, llm_failures, ck3_maa_definitions, unit_to_class_map, unit_variant_map, unit_to_description_map,
        categorized_units, unit_categories, unit_stats_map, all_units,
        excluded_units_set=excluded_units_set,
        faction_to_heritage_map=faction_to_heritage_map,
        heritage_to_factions_map=heritage_to_factions_map,
        screen_name_to_faction_key_map=screen_name_to_faction_key_map,
        faction_key_to_units_map=faction_key_to_units_map,
        llm_helper=llm_helper,
        unit_to_training_level=unit_to_training_level,
        faction_elite_units=defaultdict(set), # Pass empty dict, as this pass now handles factions that failed LLM
        faction_to_json_map=faction_to_json_map,
        faction_culture_map=faction_culture_map,
        faction_pool_cache=faction_pool_cache,
        faction_to_subculture_map=faction_to_subculture_map,
        subculture_to_factions_map=subculture_to_factions_map,
        faction_key_to_screen_name_map=faction_key_to_screen_name_map,
        culture_to_faction_map=culture_to_faction_map,
        faction_to_heritages_map=faction_to_heritages_map
    )

    # NEW: Merge duplicate factions AFTER adding and renaming.
    merged_duplicates_count = faction_xml_utils.merge_duplicate_factions(root, screen_name_to_faction_key_map)
    if merged_duplicates_count > 0:
        _recache_factions() # Re-cache after removals

    # NEW: Re-run duplicate removal for MenAtArm after merging factions
    if merged_duplicates_count > 0:
        print(f"\nRe-running duplicate MenAtArm removal after merging {merged_duplicates_count} factions...")
        post_merge_maa_removed_count = faction_xml_utils.remove_duplicate_men_at_arm_tags(root)
        if post_merge_maa_removed_count > 0:
            duplicate_maa_removed_count += post_merge_maa_removed_count

    # --- NEW: Consolidated Attribute Management Pass (Global) ---
    # This is now a wrapper around the per-faction version, but still called once globally.
    # The counts are already accumulated in the main loop.
    # print("\nRunning consolidated attribute management pass...")
    # (siege_attr_count, siege_engine_per_unit_attr_count,
    #  num_guns_attr_count, max_attr_count) = _run_attribute_management_pass(
    #     root, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege
    # )
    siege_attr_count = per_faction_siege_attr_count
    siege_engine_per_unit_attr_count = per_faction_siege_engine_per_unit_attr_count
    num_guns_attr_count = per_faction_num_guns_attr_count
    max_attr_count = per_faction_max_attr_count


    # Ensure all Levies and Garrison elements have 'max' attribute set to 'LEVY'
    # This was previously a separate loop, now mostly handled per-faction.
    # The global scan below will catch any remaining.
    levy_garrison_max_fix_count = per_faction_levy_garrison_max_fix_count
    # print("\nRunning final safeguard to ensure all Levies and Garrison elements have 'max' attribute...")
    # for faction in root.findall('Faction'):
    #     faction_name = faction.get('name')
    #     for element in faction:
    #         if element.tag in ['Levies', 'Garrison']:
    #             element_key = element.get('key', 'N/A')
    #             current_max = element.get('max')
    #             if current_max != 'LEVY':
    #                 print(f"  -> Fixing {element.tag} in faction '{faction_name}': <{element.tag} key='{element_key}' current_max='{current_max}'>. Setting max='LEVY'.")
    #                 element.set('max', 'LEVY')
    #                 levy_garrison_max_fix_count += 1
    #             else:
    #                 print(f"  -> {element.tag} in faction '{faction_name}': <{element.tag} key='{element_key}'> already has max='LEVY'.")

    # if levy_garrison_max_fix_count > 0:
    #     print(f"Fixed {levy_garrison_max_fix_count} Levies/Garrison elements missing 'max' attribute in final safeguard.")
    #     max_attr_count += levy_garrison_max_fix_count
    # else:
    #     print("No Levies/Garrison elements needed 'max' attribute fix in final safeguard.")

    # NEW: LLM Subculture Assignment Pass
    llm_subcultures_assigned_count = 0
    if llm_helper and not no_subculture:
        llm_subcultures_assigned_count = llm_orchestrator.run_llm_subculture_pass(
            root, llm_helper, time_period_context, llm_threads, llm_batch_size,
            faction_to_subculture_map, subculture_to_factions_map, screen_name_to_faction_key_map,
            all_faction_elements=all_faction_elements # Pass cached elements
        )
        if llm_subcultures_assigned_count > 0:
            total_changes += llm_subcultures_assigned_count
            print(f"LLM assigned {llm_subcultures_assigned_count} subcultures.")

    # NEW: Add subculture attributes (this will now act as a fallback for LLM failures)
    # This is now a wrapper around the per-faction version, but still called once globally.
    # The counts are already accumulated in the main loop.
    # subculture_attr_count = faction_xml_utils.ensure_subculture_attributes(root, screen_name_to_faction_key_map, faction_to_subculture_map, no_subculture=no_subculture, most_common_faction_key=most_common_faction_key, faction_key_to_screen_name_map=faction_key_to_screen_name_map, culture_to_faction_map=culture_to_faction_map, faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map, faction_to_heritages_map=faction_to_heritages_map)
    subculture_attr_count = per_faction_subculture_attr_count

    # NEW: Ensure Default faction is first
    default_moved_count = faction_xml_utils.ensure_default_faction_is_first(root)
    if default_moved_count > 0:
        _recache_factions() # Re-cache after reordering

    # NEW: Final scrub of excluded units before validation to ensure none slip through.
    # This forces reprocessing of any slots that were filled with an.
    # This is now a wrapper around the per-faction version, but still called once globally.
    # The counts are already accumulated in the main loop.
    # print("\nRunning final scrub to remove any excluded unit keys...")
    # final_scrub_removed_count = faction_xml_utils.remove_excluded_unit_keys(root, excluded_units_set)
    final_scrub_removed_count = per_faction_excluded_keys_removed_count
    if final_scrub_removed_count > 0:
        excluded_keys_removed_count += final_scrub_removed_count
        total_changes += final_scrub_removed_count
        print(f"Removed an additional {final_scrub_removed_count} excluded unit keys during final scrub, forcing reprocessing.")

    # Add safeguard to ensure ALL Levies/Garrison have 'max' attribute before validation
    # This was previously a separate loop, now mostly handled per-faction.
    # The global scan below will catch any remaining.
    pre_validation_levy_garrison_max_fix_count = per_faction_levy_garrison_max_fix_count
    # print("\nRunning safeguard to ensure ALL Levies/Garrison have 'max' attribute before validation...")
    # for faction in root.findall('Faction'):
    #     # Process Levies
    #     for levy in faction.findall('Levies'):
    #         levy.set('max', 'LEVY')
    #         levy_garrison_max_fix_count += 1
    #     # Process Garrison
    #     for garrison in faction.findall('Garrison'):
    #         garrison.set('max', 'LEVY')
    #         levy_garrison_max_fix_count += 1

    # if levy_garrison_max_fix_count > 0:
    #     print(f"Fixed {levy_garrison_max_fix_count} Levies/Garrison elements missing 'max' attribute in pre-validation safeguard.")
    #     max_attr_count += levy_garrison_max_fix_count
    # else:
    #     print("No Levies/Garrison elements needed 'max' attribute fix in pre-validation safeguard.")

    # Final validation step (First Pass)
    print("\nRunning initial XML validation...")
    initial_validation_failures = faction_xml_utils.validate_final_faction_completeness(root, ck3_maa_definitions, culture_factions, no_garrison_flag=no_garrison, main_mod_faction_maa_map=main_mod_faction_maa_map, is_submod_mode=is_submod_mode, unit_to_class_map=unit_to_class_map, unit_categories=unit_categories)

    if initial_validation_failures:
        print(f"\n--- Initial XML Validation Failed ({len(initial_validation_failures)} issues found). Attempting final fix pass... ---")
        # Attempt to fix failures
        final_fixes_applied = processing_passes.run_final_fix_pass(
            root,
            initial_validation_failures,
            categorized_units,
            all_units, # Use global all_units for best-effort
            unit_categories,
            tier, # Pass tier_arg
            unit_variant_map,
            ck3_maa_definitions,
            unit_to_description_map,
            unit_stats_map, # Passed as keyword
            general_units,
            unit_to_class_map,
            excluded_units_set,
            screen_name_to_faction_key_map, # NEW
            faction_key_to_units_map, # NEW
            faction_to_subculture_map, # NEW
            subculture_to_factions_map, # NEW
            faction_key_to_screen_name_map, # NEW
            culture_to_faction_map, # NEW
            faction_to_heritage_map, # NEW
            heritage_to_factions_map, # NEW
            faction_to_heritages_map, # NEW
            faction_pool_cache, # NEW
            faction_to_json_map=faction_to_json_map,
            faction_culture_map=faction_culture_map,
            llm_helper=llm_helper, # NEW
            unit_to_training_level=unit_to_training_level, # NEW
            faction_elite_units=defaultdict(set) # NEW
        )
        total_changes += final_fixes_applied
        print(f"Attempted to fix {len(initial_validation_failures)} issues, applied {final_fixes_applied} fixes.")

        # Run FINAL safeguard after fixes to ensure ALL Levies/Garrison have 'max' attribute
        # Use global XPath search to find ALL elements regardless of when they were created
        print("\nRunning global scan for Levies/Garrison missing 'max' attribute...")
        levy_count = 0
        garrison_count = 0
        for levy in root.findall('.//Levies'):
            levy.set('max', 'LEVY')
            levy_count += 1
        for garrison in root.findall('.//Garrison'):
            garrison.set('max', 'LEVY')
            garrison_count += 1
        final_safeguard_count = levy_count + garrison_count

        if final_safeguard_count > 0:
            print(f"FORCED 'max=LEVY' for {levy_count} Levies and {garrison_count} Garrison elements in global scan.")
            max_attr_count += final_safeguard_count
        else:
            print("All Levies/Garrison elements already have 'max=LEVY'")

        # Second validation pass after attempting fixes
        print("\nRunning second XML validation after fix attempt...")
        final_validation_failures = faction_xml_utils.validate_final_faction_completeness(root, ck3_maa_definitions, culture_factions, no_garrison_flag=no_garrison, main_mod_faction_maa_map=main_mod_faction_maa_map, is_submod_mode=is_submod_mode, unit_to_class_map=unit_to_class_map, unit_categories=unit_categories)

        if final_validation_failures:
            print("\n--- XML VALIDATION FAILED AFTER FINAL FIX ATTEMPT ---")
            for failure in final_validation_failures:
                error_msg = f"  - Faction '{failure.get('faction_element').get('name', 'Unknown')}': "
                if failure.get('validation_error') == 'missing_max_attribute':
                    element_key = failure.get('element').get('key', 'N/A') if failure.get('element') is not None else 'N/A'
                    element_type = failure.get('element').get('type', 'N/A') if failure.get('element') is not None and failure.get('tag_name') == 'MenAtArm' else 'N/A'
                    if failure.get('tag_name') == 'MenAtArm':
                        error_msg += f"<{failure['tag_name']} key='{element_key}' type='{element_type}'> is missing 'max' attribute."
                    else:
                        error_msg += f"<{failure['tag_name']} key='{element_key}'> is missing 'max' attribute."
                elif failure.get('tag_name') == 'General' or failure.get('tag_name') == 'Knights':
                    error_msg += f"Missing {failure['tag_name']} unit for rank '{failure['rank']}' or unit has no 'key'."
                elif failure.get('tag_name') == 'Levies':
                    if failure.get('element') is not None:
                        error_msg += f"Levies tag (key='{failure.get('element').get('key', 'N/A')}') is missing a 'key' attribute."
                    else:
                        error_msg += "Missing at least one Levies unit."
                elif failure.get('tag_name') == 'MenAtArm':
                    if failure.get('element') is not None:
                        error_msg += f"MenAtArm tag (type='{failure.get('element').get('type', 'N/A')}') is missing a 'key' attribute."
                    else:
                        error_msg += f"Missing MenAtArm definition '{failure.get('maa_definition_name', 'Unknown Type')}'."
                elif failure.get('tag_name') == 'Garrison':
                    if failure.get('element') is not None:
                        error_msg += f"Garrison tag (level='{failure.get('element').get('level', 'N/A')}') is missing a 'key' attribute."
                    else:
                        error_msg += f"Missing Garrison unit for level '{failure['level']}' or unit has no 'key'."
                else:
                    error_msg += f"Unknown validation error for tag '{failure.get('tag_name')}'."
                print(error_msg)
            raise ValueError("XML validation failed. Factions are incomplete. The output file was not saved.")
        else:
            print("Second validation pass successful. All issues resolved.")
    else:
        print("Initial XML validation successful. No issues found.")

    # --- Final Re-normalization Pass for Levies and Garrisons ---
    # This is now handled per-faction in the main loop.
    # The counts are already accumulated.
    levy_renormalize_count = per_faction_levy_renormalize_count
    garrison_renormalize_count = per_faction_garrison_renormalize_count
    if not is_submod_mode:
        # print("\nRunning final levy and garrison percentage re-normalization pass...")
        # levy_renormalize_count = 0
        # garrison_renormalize_count = 0
        # factions_with_levy_changes = set()
        # factions_with_garrison_changes = set()

        # for faction in root.findall('Faction'):
        #     faction_name = faction.get('name')
        #     if faction_name == "Default" or (is_submod_mode and faction_name in factions_in_main_mod):
        #         continue

        #     # Re-normalize levies
        #     levy_tags = faction.findall('Levies')
        #     if levy_tags:
        #         if unit_management._normalize_levy_percentages(levy_tags):
        #             factions_with_levy_changes.add(faction_name)

        #     # Re-normalize garrisons
        #     if not no_garrison:
        #         garrisons_by_level = defaultdict(list)
        #         for g_tag in faction.findall('Garrison'):
        #             level = g_tag.get('level')
        #             if level:
        #                 garrisons_by_level[level].append(g_tag)

        #         faction_had_garrison_change = False
        #         for level, tags in garrisons_by_level.items():
        #             if unit_management._normalize_levy_percentages(tags):
        #                 faction_had_garrison_change = True

        #         if faction_had_garrison_change:
        #             factions_with_garrison_changes.add(faction_name)

        # levy_renormalize_count = len(factions_with_levy_changes)
        # garrison_renormalize_count = len(factions_with_garrison_changes)

        if levy_renormalize_count > 0 or garrison_renormalize_count > 0:
            print(f"Re-normalized percentages for levies in {levy_renormalize_count} factions and garrisons in {garrison_renormalize_count} factions.")
        else:
            print("Final percentage re-normalization pass complete. No changes needed.")
    else:
        print("\nSubmod mode enabled: Skipping final levy and garrison re-normalization.")
        levy_renormalize_count = 0
        garrison_renormalize_count = 0

    # NEW: Remove any remaining zero-percentage tags
    # This is now handled per-faction in the main loop.
    # The counts are already accumulated.
    zero_percent_removed_count = per_faction_zero_percent_removed_count
    # zero_percent_removed_count = faction_xml_utils.remove_zero_percentage_tags(root)

    # --- MOVED: Final Tag and Attribute Reordering ---
    # This is now handled per-faction in the main loop.
    # The counts are already accumulated.
    reorganized_count = per_faction_reorganized_count
    reordered_attr_count = per_faction_reordered_attr_count
    # print("\nRunning final tag and attribute reordering...")
    # reorganized_count = faction_xml_utils.reorganize_faction_children(root)
    # reordered_attr_count = faction_xml_utils.reorder_attributes_in_all_tags(root)

    total_changes += (high_confidence_replacements + llm_replacements_made + low_confidence_replacements + max_attr_count +
                     factions_added + factions_fixed + default_created + faction_sync_count +
                     levy_fix_count + garrison_fix_count + merged_duplicates_count + default_moved_count +
                     factions_removed_count + reorganized_count + removed_excluded_factions_count +
                     porcentage_rename_count + ranked_units_count + excluded_keys_removed_count + siege_attr_count + num_guns_attr_count + siege_engine_per_unit_attr_count + subculture_attr_count + llm_subcultures_assigned_count + # subculture_attr_count is now the fallback count, llm_subcultures_assigned_count is new
                     factions_removed_from_main_mod + levy_renormalize_count + garrison_renormalize_count + reordered_attr_count +
                     zero_percent_removed_count + invalid_maa_removed_count + core_tags_removed_count + maa_tags_removed_from_submod +
                     duplicate_ranked_units_removed_count + stale_keys_removed_count + duplicate_maa_removed_count + procedural_keys_removed_count + duplicate_levies_fixed_count + duplicate_garrisons_fixed_count + maa_levy_conflicts_fixed)
    if submod_tag_added:
        total_changes += 1
    if submod_addon_for_added:
        total_changes += 1

    if total_changes > 0:
        summary_parts = []
        if submod_tag_added: summary_parts.append(f"added submod_tag='{submod_tag}'")
        if submod_addon_for_added: summary_parts.append(f"added/updated submod_addon_tag='{submod_addon_tag}'")
        if invalid_maa_removed_count > 0: summary_parts.append(f"removed {invalid_maa_removed_count} invalid MenAtArm tags")
        if duplicate_maa_removed_count > 0: summary_parts.append(f"removed {duplicate_maa_removed_count} duplicate MenAtArm tags")
        if maa_levy_conflicts_fixed > 0: summary_parts.append(f"fixed {maa_levy_conflicts_fixed} MAA/Levy key conflicts")
        if excluded_keys_removed_count > 0: summary_parts.append(f"removed {excluded_keys_removed_count} excluded unit keys for replacement")
        if stale_keys_removed_count > 0: summary_parts.append(f"removed {stale_keys_removed_count} stale unit keys for replacement")
        if procedural_keys_removed_count > 0: summary_parts.append(f"removed {procedural_keys_removed_count} procedural unit keys for re-evaluation")
        if porcentage_rename_count > 0: summary_parts.append(f"renamed {porcentage_rename_count} 'porcentage' attributes")
        if core_tags_removed_count > 0: summary_parts.append(f"removed {core_tags_removed_count} core unit tags from main mod factions")
        if removed_excluded_factions_count > 0: summary_parts.append(f"removed {removed_excluded_factions_count} excluded factions")
        if factions_removed_count > 0: summary_parts.append(f"removed {factions_removed_count} factions not in cultures")
        if factions_removed_from_main_mod > 0: summary_parts.append(f"removed {factions_removed_from_main_mod} factions present in main mod")
        if maa_tags_removed_from_submod > 0: summary_parts.append(f"removed {maa_tags_removed_from_submod} MenAtArm tags present in main mod")
        if default_created > 0: summary_parts.append("created new Default faction")
        if high_confidence_replacements > 0: summary_parts.append(f"made {high_confidence_replacements} high-confidence unit replacements")
        if llm_replacements_made > 0: summary_parts.append(f"made {llm_replacements_made} unit replacements via LLM")
        if low_confidence_replacements > 0: summary_parts.append(f"made {low_confidence_replacements} best-effort unit replacements")
        if duplicate_levies_fixed_count > 0: summary_parts.append(f"fixed {duplicate_levies_fixed_count} duplicate levy units")
        if duplicate_garrisons_fixed_count > 0: summary_parts.append(f"fixed {duplicate_garrisons_fixed_count} duplicate garrison units")
        if factions_added > 0: summary_parts.append(f"added {factions_added} factions")
        if factions_fixed > 0: summary_parts.append(f"fixed {factions_fixed} faction names")
        if merged_duplicates_count > 0: summary_parts.append(f"merged {merged_duplicates_count} duplicate factions")
        if faction_sync_count > 0: summary_parts.append(f"synced {faction_sync_count} MenAtArm tags")
        if ranked_units_count > 0: summary_parts.append(f"created {ranked_units_count} ranked General/Knights units")
        if duplicate_ranked_units_removed_count > 0: summary_parts.append(f"removed {duplicate_ranked_units_removed_count} duplicate ranked units")
        if max_attr_count > 0: summary_parts.append(f"added/removed {max_attr_count} 'max' attributes")
        if siege_attr_count > 0: summary_parts.append(f"added/removed {siege_attr_count} 'siege' attributes")
        if num_guns_attr_count > 0: summary_parts.append(f"added/removed {num_guns_attr_count} 'num_guns' attributes")
        if siege_engine_per_unit_attr_count > 0: summary_parts.append(f"added/removed {siege_engine_per_unit_attr_count} 'siege_engine_per_unit' attributes")
        if llm_subcultures_assigned_count > 0: summary_parts.append(f"assigned {llm_subcultures_assigned_count} subcultures via LLM") # NEW summary part
        if subculture_attr_count > 0: summary_parts.append(f"added/updated {subculture_attr_count} 'subculture' attributes (procedural fallback)") # Modified summary
        if levy_fix_count > 0: summary_parts.append(f"fixed levy structure for {levy_fix_count} factions")
        if garrison_fix_count > 0:
            if no_garrison:
                summary_parts.append(f"removed {garrison_fix_count} garrison tags")
            else:
                summary_parts.append(f"fixed garrison structure for {garrison_fix_count} factions")
        if default_moved_count > 0: summary_parts.append("moved Default faction to top")
        if reorganized_count > 0: summary_parts.append(f"reorganized tags for {reorganized_count} factions")
        if reordered_attr_count > 0: summary_parts.append(f"reordered attributes for {reordered_attr_count} tags")
        if levy_renormalize_count > 0: summary_parts.append(f"re-normalized levy percentages for {levy_renormalize_count} factions")
        if garrison_renormalize_count > 0: summary_parts.append(f"re-normalized garrison percentages for {garrison_renormalize_count} factions")
        if zero_percent_removed_count > 0: summary_parts.append(f"removed {zero_percent_removed_count} zero-percentage levy/garrison tags")


        print(f"Finished processing {units_xml_path}. Summary: {', '.join(summary_parts)}.")

        if faction_name_map:
            print("\n--- Details of Faction Names Fixed ---")
            for old_name, new_name in sorted(faction_name_map.items()):
                old_key = screen_name_to_faction_key_map.get(old_name, "N/A")
                new_key = screen_name_to_faction_key_map.get(new_name, "N/A")
                print(f"  - Changed faction '{old_name}' (key: {old_key}) to '{new_name}' (key: {new_key})")
            print("--------------------------------------")

        # --- XML Schema Validation ---
        print("\nValidating final XML against schema 'schemas/factions.xsd'...")
        if lxml_etree is None:
            print("  -> WARNING: 'lxml' library not installed. Skipping schema validation.")
        else:
            try:
                schema_path = 'schemas/factions.xsd'
                if not os.path.exists(schema_path):
                    print(f"  -> WARNING: Schema file not found at '{schema_path}'. Skipping validation.")
                else:
                    xmlschema_doc = lxml_etree.parse(schema_path)
                    xmlschema = lxml_etree.XMLSchema(xmlschema_doc)

                    # Convert the ElementTree object to a string to be parsed by lxml
                    xml_string = ET.tostring(root, encoding='unicode')
                    xml_doc_to_validate = lxml_etree.fromstring(xml_string.encode('utf-8'))

                    xmlschema.assertValid(xml_doc_to_validate)
                    print("  -> SUCCESS: XML validation passed.")

            except lxml_etree.DocumentInvalid:
                print("\n--- XML VALIDATION FAILED ---")
                print(f"The generated XML for '{units_xml_path}' is not valid according to the schema.")
                # The error log is an attribute of the schema object after validation
                for error in xmlschema.error_log:
                    print(f"  - Line {error.line}, Column {error.column}: {error.message}")
                raise ValueError("XML validation failed. The output file was not saved.")
            except Exception as e:
                print(f"\n--- An unexpected error occurred during XML validation ---")
                print(f"Error: {e}")
                raise

        shared_utils.indent_xml(root)
        tree.write(units_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Successfully updated '{units_xml_path}'.")
    else:
        print(f"No changes were made to {units_xml_path}.")

    return total_changes


def main():
    shared_utils.setup_logging()

    # --- NEW: Set a fixed seed for reproducibility ---
    random.seed(42)
    print("Set random seed to 42 for reproducibility.")
    parser = argparse.ArgumentParser(description="Intelligently fix and update Attila unit XML files.")
    parser.add_argument("--cultures-xml-path", required=True, help="Path to the target Cultures XML file to be modified.")
    parser.add_argument("--factions-xml-path", required=True, help="Path to the Attila Factions XML file for faction reconciliation.")
    parser.add_argument("--factions-xml-path-main-mod", help="Path to a main mod's Factions XML file to check for existing faction/unit combinations.")
    parser.add_argument("--ck3-common-path", required=True, help="Path to the Crusader Kings III 'common' directory.")
    parser.add_argument("--attila-db-path", required=True, help="Path to the Attila debug database directory (e.g., 'debug/919ad/attila/db').")
    parser.add_argument("--attila-text-path", help="Path to the Attila text directory containing .tsv files with unit screen names.")
    parser.add_argument("--tier", type=int, help="Specify a unit tier to use (e.g., 1, 2).")
    parser.add_argument("--exclude-factions", nargs='+', help="List of faction screen names to exclude and remove from the Factions XML.")
    parser.add_argument("--exclude-units-file", help="Path to a text file containing unit keys to exclude from all unit selections, one per line.")
    parser.add_argument("--exclude-units-prefix", action='append', help="A unit key prefix to exclude. Can be specified multiple times (e.g., --exclude-units-prefix att_ --exclude-units-prefix cha_).")
    parser.add_argument("--faction-culture-map", help="Path to a JSON file containing faction to CK3 culture mappings for high-priority overrides.")
    parser.add_argument("--faction-name-overrides", help="Path to a JSON file to override faction screen names from the TSV files.")
    parser.add_argument("--time-period-xml-path", help="Path to the Time Period XML file for historical context.")
    parser.add_argument("--no-siege", action='store_true', help="If set, do not add the 'siege' attribute to siege units.")
    parser.add_argument("--no-subculture", action='store_true', help="If set, do not add the 'subculture' attribute to factions.")
    parser.add_argument("--no-garrison", action='store_true', help="If set, do not add garrison units and remove any existing <Garrison> tags.")
    # LLM Arguments
    parser.add_argument("--use-llm", action='store_true', help="Enable LLM network calls to fix missing units. Requires --llm-cache-tag.")
    parser.add_argument("--g4f", action='store_true', help="Use the g4f library for LLM calls instead of litellm.")
    parser.add_argument("--llm-model", default="ollama/llama3", help="The model name to use with the LLM helper (e.g., 'ollama/llama3', 'gpt-4').")
    parser.add_argument("--llm-api-base", help="The base URL for the LLM API server (for local models).")
    parser.add_argument("--llm-api-key", help="The API key for the LLM service.")
    parser.add_argument("--llm-cache-dir", default="mapper_tools/llm_cache", help="Directory for LLM cache files.")
    parser.add_argument("--llm-cache-tag", help="A unique tag for partitioning the LLM cache (e.g., 'AGOT'). If set without --use-llm, enables cache-only mode.")
    parser.add_argument("--llm-batch-size", type=int, default=200, help="The maximum number of unit replacement requests to send to the LLM in a single batch.")
    parser.add_argument("--llm-force-refresh", action='store_true', help="If set, the LLM will ignore existing cache entries for the current run and re-query for all requests, but will still save new results to the cache.")
    parser.add_argument("--clear-llm-cache-units-file", help="Path to a text file containing unit keys to recache, one per line. LLM cache entries for these specific units will be cleared and re-queried.")
    parser.add_argument("--llm-clear-null-cache", action='store_true', help="If set, clears all 'null' entries from all LLM cache files before processing.")
    parser.add_argument("--llm-threads", type=int, default=1, help="The number of parallel threads to use for LLM API calls.")
    parser.add_argument("--llm-update-mappings-only", action='store_true', help="Use the LLM only to update ck3_to_attila_mappings.json with new CK3 MAA types. Disables LLM network calls for unit replacement.")
    parser.add_argument("--force-procedural-recache", action='store_true', help="Force reprocessing of all units that are not found in the LLM cache.")
    parser.add_argument("--first-pass-threshold", type=float, default=0.90, help="Fuzzy matching threshold for the high-confidence procedural pass (0.0 to 1.0). Default is 0.90.")
    parser.add_argument("--submod-tag", help="An optional tag to add to the root <Factions> element as submod_tag=\"value\".")
    parser.add_argument("--submod-addon-tag", help="An optional tag to add to the root <Factions> element as submod_addon_tag=\"value\".")
    # --- Roster Fixing and Reviewing Arguments ---
    parser.add_argument("--fix-rosters-only", action='store_true', help="Run only the procedural and LLM-based roster fixing process.")
    parser.add_argument("--llm-review-rosters-only", action='store_true', help="Skip the roster fixing process and only run the LLM-based roster review on the existing file.")
    parser.add_argument("--update-subcultures-only", action='store_true', help="Run only the subculture assignment process.") # NEW ARGUMENT
    args = parser.parse_args()

    # --- Construct dynamic paths right after parsing args ---
    tsv_dir = os.path.join(args.attila_db_path, "main_units_tables")
    land_units_tsv_dir = os.path.join(args.attila_db_path, "land_units_tables")
    permissions_tsv_dir = os.path.join(args.attila_db_path, "units_custom_battle_permissions_tables")
    faction_tables_dir = os.path.join(args.attila_db_path, "factions_tables")
    men_at_arms_dir = os.path.join(args.ck3_common_path, "men_at_arms_types")
    unit_variants_tables_dir = os.path.join(args.attila_db_path, "unit_variants_tables")

    # --- Dynamically reload mappings at the start of every run ---
    mappings.load_mappings()
    print("Dynamically loaded CK3 to Attila mappings from JSON.")

    if not os.path.isdir(args.attila_db_path):
        raise FileNotFoundError(f"Attila DB path not found or is not a directory: {args.attila_db_path}")
    if not os.path.isdir(args.ck3_common_path):
        raise FileNotFoundError(f"CK3 common path not found or is not a directory: {args.ck3_common_path}")

    if args.use_llm and not args.llm_cache_tag:
        parser.error("--llm-cache-tag is required when --use-llm is specified.")

    if not shared_utils.prompt_to_create_xml(args.cultures_xml_path, 'Cultures'):
        print("Cultures XML file is required to proceed. Aborting.")
        return

    if not shared_utils.prompt_to_create_xml(args.factions_xml_path, 'Factions'):
        print("Factions XML file is required to proceed. Aborting.")
        return

    # Read time period context
    time_period_context = ""
    if args.time_period_xml_path:
        try:
            if os.path.exists(args.time_period_xml_path):
                tree = ET.parse(args.time_period_xml_path)
                root = tree.getroot()
                start_date = root.findtext('StartDate')
                end_date = root.findtext('EndDate')
                if start_date and end_date:
                    time_period_context = f"The historical time period is from {start_date} AD to {end_date} AD."
                    print(f"Loaded time period context: {time_period_context}")
                else:
                    print(f"Warning: Could not find <StartDate> or <EndDate> in '{args.time_period_xml_path}'.")
            else:
                print(f"Warning: Time period XML file not found at '{args.time_period_xml_path}'.")
        except ET.ParseError as e:
            print(f"Warning: Could not parse time period XML '{args.time_period_xml_path}': {e}. Proceeding without it.")
            raise

    # The tier argument is now an integer, no need to normalize or lower()
    tier_arg = args.tier

    print("Starting parallel data loading...")

    # --- Group 1: Independent data sources ---
    group1_tasks = {
        'faction_key_to_screen_name_map': (shared_utils.get_faction_key_to_screen_name_map, [faction_tables_dir]),
        'culture_factions': (shared_utils.get_factions_from_cultures_xml, [args.cultures_xml_path]),
        'all_units': (shared_utils.get_all_land_units_keys, [land_units_tsv_dir]),
        'unit_to_screen_name_map': (shared_utils.get_unit_screen_name_map, [args.attila_text_path]),
        'culture_to_faction_map': (shared_utils.get_culture_to_faction_map_from_xml, [args.cultures_xml_path]),
        'unit_categories': (shared_utils.get_unit_land_categories, [land_units_tsv_dir]),
        'unit_to_class_map': (shared_utils.get_unit_classes, [land_units_tsv_dir]),
        'unit_to_description_map': (shared_utils.get_unit_descriptions, [land_units_tsv_dir]),
        'unit_to_num_guns_map': (shared_utils.get_unit_num_guns, [land_units_tsv_dir]),
        'unit_to_training_level': (shared_utils.get_unit_training_levels, [land_units_tsv_dir]),
        'unit_to_faction_key_map': (shared_utils.get_unit_to_faction_key_map, [permissions_tsv_dir]),
        'ck3_maa_definitions': (shared_utils.get_ck3_maa_definitions, [men_at_arms_dir]),
        'faction_key_to_units_map': (shared_utils.get_faction_key_to_units_map, [permissions_tsv_dir]),
        'subculture_maps': (shared_utils.get_faction_subculture_maps, [faction_tables_dir]),
        'heritage_maps': (shared_utils.get_faction_heritage_maps_from_xml, [args.cultures_xml_path]),
    }
    g1_results = _load_data_in_parallel(group1_tasks)

    # --- Unpack and process Group 1 results ---
    faction_key_to_screen_name_map = g1_results['faction_key_to_screen_name_map']
    culture_factions = g1_results['culture_factions']
    all_units = g1_results['all_units']
    unit_to_screen_name_map = g1_results['unit_to_screen_name_map']
    culture_to_faction_map = g1_results['culture_to_faction_map']
    unit_categories = g1_results['unit_categories']
    unit_to_class_map = g1_results['unit_to_class_map']
    unit_to_description_map = g1_results['unit_to_description_map']
    unit_to_num_guns_map = g1_results['unit_to_num_guns_map']
    unit_to_training_level = g1_results['unit_to_training_level']
    unit_to_faction_key_map = g1_results['unit_to_faction_key_map']
    ck3_maa_definitions = g1_results['ck3_maa_definitions']
    faction_key_to_units_map = g1_results['faction_key_to_units_map']
    faction_to_subculture_map, subculture_to_factions_map = g1_results['subculture_maps']
    faction_to_heritage_map, heritage_to_factions_map = g1_results['heritage_maps']

    # --- Sequential processing for data that depends on Group 1 or has complex logic ---

    # 2. Apply faction name overrides if provided
    if args.faction_name_overrides:
        try:
            with open(args.faction_name_overrides, 'r', encoding='utf-8-sig') as f:
                overrides = json.load(f)
            print(f"Applying {len(overrides)} faction name overrides from '{args.faction_name_overrides}'.")

            # Create a temporary reverse map for easier override application
            temp_screen_name_to_key_map = {v: k for k, v in faction_key_to_screen_name_map.items()}

            for identifier, new_screen_name in overrides.items():
                faction_key = None
                old_screen_name = None

                if identifier in faction_key_to_screen_name_map: # identifier is a key
                    faction_key = identifier
                    old_screen_name = faction_key_to_screen_name_map.get(faction_key)
                elif identifier in temp_screen_name_to_key_map: # identifier is a screen name
                    old_screen_name = identifier
                    faction_key = temp_screen_name_to_key_map.get(old_screen_name)
                else:
                    print(f"  -> WARNING: Override identifier '{identifier}' not found as a faction key or screen name. Skipping.")
                    continue

                if old_screen_name != new_screen_name and faction_key:
                    faction_key_to_screen_name_map[faction_key] = new_screen_name
                    print(f"  - Overrode faction '{faction_key}': '{old_screen_name}' -> '{new_screen_name}'.")

        except FileNotFoundError:
            print(f"Error: Faction name overrides file not found at '{args.faction_name_overrides}'. Proceeding without overrides.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from '{args.faction_name_overrides}': {e}. Proceeding without overrides.")
            raise

    # 3. Create the primary screen_name_to_key_map
    screen_name_to_faction_key_map = {v: k for k, v in faction_key_to_screen_name_map.items()}

    # 4. Validate and augment faction maps
    if not culture_factions:
        print("Could not load any factions from Cultures.xml. Aborting.")
        return
    print(f"Loaded {len(culture_factions)} factions from '{os.path.basename(args.cultures_xml_path)}' as the source of truth.")

    db_factions = set(faction_key_to_screen_name_map.values())
    missing_from_db = culture_factions - db_factions
    if missing_from_db:
        print(f"Found {len(missing_from_db)} factions in Cultures.xml that are not in the Attila DB. They will be added to the processing maps.")
        for faction_name in sorted(list(missing_from_db)):
            generated_key = f"cw_gen_{faction_name.lower().replace(' ', '_').replace('-', '_')}"
            if generated_key in faction_key_to_screen_name_map:
                generated_key = f"cw_gen_{faction_name.lower().replace(' ', '_').replace('-', '_')}_{random.randint(1000, 9999)}"
            print(f"  - Adding '{faction_name}' with generated key '{generated_key}'.")
            faction_key_to_screen_name_map[generated_key] = faction_name
            screen_name_to_faction_key_map[faction_name] = generated_key

    # 5. Define and apply excluded factions
    excluded_factions_set = set(args.exclude_factions) if args.exclude_factions else set()
    if excluded_factions_set:
        print(f"The following factions will be excluded and removed from the Factions XML: {', '.join(sorted(list(excluded_factions_set)))}")
        culture_factions = culture_factions - excluded_factions_set
        print(f"Removed {len(excluded_factions_set)} excluded factions from the culture_factions set for validation.")

    # Load units to recache
    units_to_recache = set()
    if args.clear_llm_cache_units_file:
        try:
            with open(args.clear_llm_cache_units_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        units_to_recache.add(line)
            print(f"Loaded {len(units_to_recache)} unit keys for targeted cache clearing from '{args.clear_llm_cache_units_file}'.")
        except FileNotFoundError:
            print(f"Warning: File for --clear-llm-cache-units-file not found at '{args.clear_llm_cache_units_file}'. No units will be specifically recached.")

    # Load excluded units file
    excluded_units_set = set()
    if args.exclude_units_file:
        try:
            with open(args.exclude_units_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        excluded_units_set.add(line)
            print(f"Loaded {len(excluded_units_set)} unit keys to exclude from all unit selections from '{args.exclude_units_file}'.")
        except FileNotFoundError:
            print(f"Warning: Exclude units file not found at '{args.exclude_units_file}'. No units will be excluded.")

    # Validate all_units
    if not all_units:
        print("Could not load any valid land unit keys from land_units_tables. Aborting.")
        return
    print(f"Loaded {len(all_units)} definitive valid unit keys from land_units_tables.")

    # Validate faction_key_to_units_map
    if not faction_key_to_units_map:
        print("Warning: Could not load faction to unit mappings. Faction-specific unit validation will be skipped.")
        faction_key_to_units_map = {}
    else:
        print(f"Loaded {len(faction_key_to_units_map)} factions with their unit permissions.")

    # --- Unit Exclusion Logic ---
    all_possible_units_from_permissions = set()
    for unit_set in faction_key_to_units_map.values():
        all_possible_units_from_permissions.update(unit_set)
    print(f"Created a comprehensive set of {len(all_possible_units_from_permissions)} units from permissions for prefix exclusion.")

    if args.exclude_units_prefix:
        prefix_excluded_units = set()
        for unit_key in all_possible_units_from_permissions:
            for prefix in args.exclude_units_prefix:
                if unit_key.startswith(prefix):
                    prefix_excluded_units.add(unit_key)
                    break
        if prefix_excluded_units:
            excluded_units_set.update(prefix_excluded_units)
            print(f"Identified {len(prefix_excluded_units)} units for exclusion based on prefixes: {', '.join(args.exclude_units_prefix)}.")
        else:
            print(f"No units found matching the provided prefixes: {', '.join(args.exclude_units_prefix)}.")

    if excluded_units_set:
        for faction_key in faction_key_to_units_map:
            faction_key_to_units_map[faction_key] = {unit for unit in faction_key_to_units_map[faction_key] if unit not in excluded_units_set}
        print(f"Filtered faction_key_to_units_map against {len(excluded_units_set)} final excluded units.")
        original_unit_count = len(all_units)
        all_units = all_units - excluded_units_set
        print(f"Filtered out {original_unit_count - len(all_units)} excluded units from the main unit pool.")

    # Load and validate faction_culture_map if provided
    faction_culture_map = {}
    if args.faction_culture_map:
        try:
            with open(args.faction_culture_map, 'r', encoding='utf-8') as f:
                faction_culture_map = json.load(f)
            print(f"Loaded {len(faction_culture_map)} entries from faction-culture map: '{args.faction_culture_map}'.")
        except FileNotFoundError:
            print(f"Error: Faction-culture map file not found at '{args.faction_culture_map}'. Proceeding without it.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from '{args.faction_culture_map}': {e}. Proceeding without it.")
            raise

    # Pre-calculate a definitive mapping from each faction to its JSON data group.
    faction_to_json_map = {}
    if faction_culture_map:
        print("\nPre-calculating definitive faction-to-JSON-group mappings...")
        faction_to_culture_list_map = shared_utils.get_faction_to_culture_list_map_from_xml(args.cultures_xml_path)
        culture_to_json_group_key_map = {}
        for group_key, group_data in faction_culture_map.items():
            for culture_name in group_data.get("cultures", []):
                culture_to_json_group_key_map[culture_name.lower()] = group_key
        for faction_name in culture_factions:
            json_group_key, json_group_data = shared_utils.find_best_fuzzy_match_in_dict(faction_name, faction_culture_map, threshold=0.80)
            if json_group_data:
                faction_to_json_map[faction_name] = json_group_data
                print(f"  - Mapped faction '{faction_name}' to JSON group '{json_group_key}' via direct name match.")
                continue
            faction_cultures = faction_to_culture_list_map.get(faction_name, [])
            if not faction_cultures:
                continue
            possible_group_keys = []
            for culture in faction_cultures:
                group_key = culture_to_json_group_key_map.get(culture.lower())
                if group_key:
                    possible_group_keys.append(group_key)
            if possible_group_keys:
                most_common_group_key = Counter(possible_group_keys).most_common(1)[0][0]
                faction_to_json_map[faction_name] = faction_culture_map[most_common_group_key]
                print(f"  - Mapped faction '{faction_name}' to JSON group '{most_common_group_key}' via reverse culture lookup.")
        print(f"Successfully created {len(faction_to_json_map)} definitive faction-to-JSON mappings.")

    # --- Group 2: Data sources dependent on all_units ---
    group2_tasks = {
        'unit_stats_map': (shared_utils.get_unit_stats_map, [tsv_dir, land_units_tsv_dir, all_units]),
        'unit_to_tier_map': (shared_utils.get_unit_to_tier_map, [tsv_dir, all_units]),
        'categorized_units': (shared_utils.get_tsv_units, [tsv_dir, all_units]),
        'general_units': (shared_utils.get_general_units, [permissions_tsv_dir, all_units]),
    }
    g2_results = _load_data_in_parallel(group2_tasks)
    unit_stats_map = g2_results['unit_stats_map']
    unit_to_tier_map = g2_results['unit_to_tier_map']
    categorized_units = g2_results['categorized_units']
    general_units = g2_results['general_units']

    # --- Group 3: Data sources dependent on Group 1 & 2 results ---
    unit_variant_map = shared_utils.get_unit_variant_map(unit_variants_tables_dir, all_units, unit_to_tier_map)
    print("  -> Loaded unit_variant_map")
    variant_to_base_map = shared_utils.create_variant_to_base_map(unit_variant_map)

    # --- Initialize LLM Helper (after loading necessary data) ---
    llm_helper = None
    if args.llm_cache_tag:
        try:
            from mapper_tools.llm_helper import LLMHelper
            network_calls_for_units = args.use_llm and not args.llm_update_mappings_only
            llm_helper = LLMHelper(
                model=args.llm_model, cache_dir=args.llm_cache_dir, api_base=args.llm_api_base,
                api_key=args.llm_api_key, cache_tag=args.llm_cache_tag, force_refresh=args.llm_force_refresh,
                units_to_recache=units_to_recache, excluded_units_set=excluded_units_set,
                network_calls_enabled=network_calls_for_units, unit_to_screen_name_map=unit_to_screen_name_map,
                use_g4f=args.g4f, clear_null_cache=args.llm_clear_null_cache, faction_to_json_map=faction_to_json_map,
                faction_culture_map=faction_culture_map, all_units=all_units, unit_to_class_map=unit_to_class_map
            )
            if network_calls_for_units:
                print("LLM Helper initialized (network calls ENABLED for unit replacement).")
            else:
                print("LLM Helper initialized (cache-only mode for unit replacement, network calls DISABLED).")
        except ImportError:
            print("Warning: 'litellm' or 'g4f' library not found. Please install required libraries to use the LLM feature.")
            llm_helper = None
        except Exception as e:
            print(f"Error initializing LLM Helper: {e}")
            llm_helper = None

    # --- Final validation and print summaries for loaded data ---
    if not categorized_units: print("Could not load main unit data from TSV files. Aborting."); return
    if general_units is None: print("Warning: Could not load general-eligible units. General tags will not be processed correctly."); general_units = set()
    if not unit_categories: print("Warning: Could not load unit categories. Category-based matching will be disabled."); unit_categories = {}
    if not unit_to_class_map: print("Warning: Could not load unit classes. High-precision matching will be disabled."); unit_to_class_map = {}
    if not unit_to_description_map: print("Warning: Could not load unit descriptions. Thematic matching will be less accurate."); unit_to_description_map = {}
    if not unit_to_training_level: print("Warning: Could not load unit training levels. Levy identification will be less accurate."); unit_to_training_level = {}
    if not faction_key_to_screen_name_map: print("Warning: Could not load faction key to screen name map. Faction name validation will be skipped.")
    if not unit_to_faction_key_map: print("Warning: Could not load unit to faction key map. Faction name validation will be skipped.")
    if not unit_variant_map and tier_arg: print(f"Tier '{tier_arg}' was specified, but no unit variant data was loaded. Tier-based selection will be disabled.")
    if not ck3_maa_definitions: print("Warning: Could not load CK3 Men-at-Arms definitions. MAA processing will be limited."); ck3_maa_definitions = {}

    # --- LLM Mapping Update Logic ---
    if (args.use_llm or args.llm_update_mappings_only) and args.llm_cache_tag:
        print("\nChecking for new CK3 Men-at-Arms types to map...")
        all_ck3_maa_definitions = set(ck3_maa_definitions.keys())
        missing_types = all_ck3_maa_definitions - set(mappings.CK3_TYPE_TO_ATTILA_ROLES.keys())
        if missing_types:
            print(f"Found {len(missing_types)} new CK3 MAA types: {', '.join(sorted(list(missing_types)))}")
            all_attila_classes = set(unit_to_class_map.values())
            all_attila_roles = set(categorized_units.keys())
            all_attila_max_categories = set(mappings.MAX_TO_CATEGORY.keys())
            llm_requests = []
            for maa_definition_name in sorted(list(missing_types)):
                internal_type = ck3_maa_definitions.get(maa_definition_name)
                llm_requests.append({
                    'id': maa_definition_name, 'ck3_maa_definition': maa_definition_name,
                    'ck3_internal_type': internal_type, 'attila_classes': list(all_attila_classes),
                    'attila_roles': list(all_attila_roles), 'attila_max_categories': list(all_attila_max_categories)
                })
            mapping_helper = llm_helper
            if mapping_helper and not mapping_helper.network_calls_enabled:
                print("  -> Main LLM Helper has network calls disabled for unit replacement. Initializing a temporary LLM Helper for mapping updates.")
                try:
                    from mapper_tools.llm_helper import LLMHelper
                    mapping_helper = LLMHelper(
                        model=args.llm_model, cache_dir=args.llm_cache_dir, api_base=args.llm_api_base,
                        api_key=args.llm_api_key, cache_tag=args.llm_cache_tag, force_refresh=args.llm_force_refresh,
                        units_to_recache=units_to_recache, excluded_units_set=excluded_units_set,
                        network_calls_enabled=True, use_g4f=args.g4f
                    )
                except Exception as e:
                    print(f"  -> ERROR: Could not initialize temporary LLM Helper for mapping updates: {e}. Skipping mapping updates.")
                    mapping_helper = None
            if mapping_helper:
                new_mappings = mapping_helper.get_batch_mapping_updates(llm_requests)
                if new_mappings:
                    print(f"LLM provided {len(new_mappings)} new mappings. Updating and saving...")
                    updated = False
                    for ck3_type, mapping_data in new_mappings.items():
                        mappings.CK3_TYPE_TO_ATTILA_CLASS[ck3_type] = mapping_data['attila_class']
                        mappings.CK3_TYPE_TO_ATTILA_ROLES[ck3_type] = mapping_data['attila_roles']
                        mappings.CK3_TYPE_TO_ATTILA_MAX_CATEGORY[ck3_type] = mapping_data['attila_max_category']
                        updated = True
                    if updated:
                        mappings.save_mappings()
                else:
                    print("LLM did not provide any valid new mappings.")
            else:
                print("Skipping LLM mapping updates due to initialization error.")
        else:
            print("All CK3 MAA types are already present in the mappings file.")

    # --- Print final load summaries ---
    print(f"Loaded {len(categorized_units)} categorized units from TSV files.")
    print(f"Loaded {len(general_units)} unique general-eligible units.")
    print(f"Loaded {len(unit_categories)} unit categories from land_units TSV files.")
    print(f"Loaded {len(unit_to_class_map)} unit class mappings from land_units TSV files.")
    print(f"Loaded {len(unit_to_description_map)} unit descriptions from land_units TSV files.")
    print(f"Loaded num_guns for {len(unit_to_num_guns_map)} units from land_units.tsv.")
    print(f"Loaded {len(unit_to_training_level)} unit training levels from land_units TSV files.")
    print(f"Loaded {len(faction_key_to_screen_name_map)} faction key to screen name mappings.")
    print(f"Loaded {len(unit_to_faction_key_map)} unit to faction key mappings.")
    print(f"Loaded {len(unit_variant_map)} base units with variants from unit_variants TSV files.")
    print(f"Loaded {len(variant_to_base_map)} variant to base unit mappings.")
    print(f"Loaded {len(ck3_maa_definitions)} CK3 Men-at-Arms definitions.")

    # --- Template Pool and Sub/Heritage Map Creation ---
    template_faction_unit_pool = all_units
    most_common_faction_key = None
    most_common_faction_name = shared_utils.get_most_common_faction_from_cultures(args.cultures_xml_path)
    if most_common_faction_name:
        most_common_faction_key = screen_name_to_faction_key_map.get(most_common_faction_name)
        if most_common_faction_key:
            print(f"\nFound most common faction in Cultures.xml: '{most_common_faction_name}' (key: {most_common_faction_key}).")
            representative_pool = faction_key_to_units_map.get(most_common_faction_key)
            if representative_pool:
                template_faction_unit_pool = representative_pool
                print(f"Using a representative pool of {len(template_faction_unit_pool)} units from '{most_common_faction_name}' for template creation.")
            else:
                print(f"  -> WARNING: Could not find unit pool for most common faction '{most_common_faction_name}'. Falling back to global pool of {len(all_units)} units.")
        else:
            print(f"  -> WARNING: Could not find a faction key for the most common faction name '{most_common_faction_name}'. Falling back to global pool.")
    else:
        print(f"\nWARNING: Could not determine most common faction from Cultures.xml. Falling back to global pool of {len(all_units)} units for template creation.")

    faction_to_heritages_map = shared_utils.create_faction_to_heritages_map(heritage_to_factions_map)
    is_submod_mode = bool(args.factions_xml_path_main_mod)
    main_mod_faction_maa_map = None
    if is_submod_mode:
        print(f"\n--- Submod Mode Enabled ---")
        print(f"Loading main mod faction data from: {args.factions_xml_path_main_mod}")
        main_mod_faction_maa_map = shared_utils.get_main_mod_faction_maa_map(args.factions_xml_path_main_mod)
        if main_mod_faction_maa_map:
            print(f"Loaded MenAtArm definitions for {len(main_mod_faction_maa_map)} factions from the main mod.")
        else:
            print("Warning: Could not load any faction data from the main mod's Factions.xml.")

    # --- Main Execution Logic ---
    if args.update_subcultures_only:
        if not llm_helper:
            print("ERROR: LLM Helper is not initialized. Subculture update requires LLM functionality (--llm-cache-tag is required).")
            return
        update_subcultures_only(
            args.factions_xml_path, llm_helper, time_period_context, args.llm_threads, args.llm_batch_size,
            faction_to_subculture_map, subculture_to_factions_map, screen_name_to_faction_key_map,
            args.no_subculture, most_common_faction_key, faction_key_to_screen_name_map, culture_to_faction_map,
            faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map
        )
        return

    run_fix = not args.llm_review_rosters_only
    run_review = not args.fix_rosters_only

    if run_fix:
        print("\n--- Starting Roster Fixing Pass ---")
        process_units_xml(
            args.factions_xml_path, categorized_units, all_units, general_units, unit_categories,
            faction_key_to_screen_name_map, unit_to_faction_key_map, template_faction_unit_pool,
            culture_factions, tier_arg, unit_variant_map, unit_to_tier_map, variant_to_base_map,
            unit_to_training_level, ck3_maa_definitions, screen_name_to_faction_key_map,
            faction_key_to_units_map, args.submod_tag, excluded_factions_set, unit_to_class_map,
            faction_to_subculture_map, subculture_to_factions_map, culture_to_faction_map,
            unit_to_description_map, unit_stats_map, faction_culture_map, llm_helper,
            excluded_units_set, unit_to_num_guns_map, llm_batch_size=args.llm_batch_size,
            no_siege=args.no_siege, no_subculture=args.no_subculture, no_garrison=args.no_garrison,
            most_common_faction_key=most_common_faction_key, main_mod_faction_maa_map=main_mod_faction_maa_map,
            llm_threads=args.llm_threads, faction_to_heritage_map=faction_to_heritage_map,
            heritage_to_factions_map=heritage_to_factions_map, faction_to_heritages_map=faction_to_heritages_map,
            first_pass_threshold=args.first_pass_threshold, is_submod_mode=is_submod_mode,
            submod_addon_tag=args.submod_addon_tag, faction_to_json_map=faction_to_json_map,
            time_period_context=time_period_context, force_procedural_recache=args.force_procedural_recache
        )

    if run_review:
        if not llm_helper or not llm_helper.network_calls_enabled:
            print("\n--- Skipping LLM Roster Review Pass ---")
            print("ERROR: LLM Roster Review requires --use-llm and --llm-cache-tag to be configured with network calls enabled.")
        else:
            try:
                tree = ET.parse(args.factions_xml_path)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"Error parsing XML file {args.factions_xml_path} for review: {e}. Aborting review.")
                raise
            review_faction_pool_cache = {}
            # Cache faction elements for the review pass
            all_faction_elements_review = list(root.findall('Faction'))
            review_changes = llm_orchestrator.run_llm_roster_review_pass(
                root, llm_helper, time_period_context, args.llm_threads, args.llm_batch_size,
                review_faction_pool_cache, all_units, excluded_units_set, screen_name_to_faction_key_map,
                faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map,
                faction_key_to_screen_name_map, culture_to_faction_map, faction_to_heritage_map,
                heritage_to_factions_map, faction_to_heritages_map, ck3_maa_definitions,
                all_faction_elements=all_faction_elements_review # Pass cached elements
            )
            if review_changes > 0:
                print(f"\nLLM Roster Review applied {review_changes} corrections. Saving file...")
                shared_utils.indent_xml(root)
                tree.write(args.factions_xml_path, encoding='utf-8', xml_declaration=True)
                print(f"Successfully saved updated rosters to '{args.factions_xml_path}'.")
            else:
                print("\nLLM Roster Review complete. No changes were made.")

if __name__ == "__main__":
    main()
