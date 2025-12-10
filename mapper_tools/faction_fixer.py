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


def _run_attribute_management_pass(root, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege, all_faction_elements):
    """
    Consolidated pass to manage all unit attributes in a single loop.
    Handles 'max', 'siege', 'siege_engine_per_unit', and 'num_guns'.
    """
    siege_attr_count = 0
    siege_engine_per_unit_attr_count = 0
    num_guns_attr_count = 0
    max_attr_count = 0

    factions_to_iterate = all_faction_elements if all_faction_elements is not None else root.findall('Faction')

    for faction in factions_to_iterate:
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
                      faction_to_heritage_map=None, heritage_to_factions_map=None, faction_to_heritages_map=None, first_pass_threshold=0.90, is_submod_mode=False, submod_addon_tag=None, faction_to_json_map=None, time_period_context="", force_procedural_recache=False, faction_elite_units=None): # Added heritage maps and first_pass_threshold, is_submod_mode, faction_elite_units
    """
    Processes a single Attila Factions XML file to fix and update unit entries.
    """
    total_changes = 0 # Initialize total_changes here
    faction_pool_cache = {} # Initialize the cache for this run
    print(f"\nProcessing file: {units_xml_path}")

    # Initialize faction_elite_units if not provided
    if faction_elite_units is None:
        faction_elite_units = defaultdict(set)

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
        removed_excluded_factions_count = faction_xml_utils.remove_excluded_factions(root, excluded_factions, screen_name_to_faction_key_map, all_faction_elements)
        if removed_excluded_factions_count > 0:
            _recache_factions() # Re-cache after removals

    # NEW: Prune factions not in Cultures.xml (NOW runs after name correction)
    factions_removed_count = faction_xml_utils.remove_factions_not_in_cultures(root, culture_factions, screen_name_to_faction_key_map, all_faction_elements)
    if factions_removed_count > 0:
        _recache_factions() # Re-cache after removals

    # NEW: Prune factions present in main mod that have no new MenAtArm types
    factions_removed_from_main_mod, removed_faction_names_for_sync = 0, set()
    if is_submod_mode:
        factions_removed_from_main_mod, removed_faction_names_for_sync = faction_xml_utils.remove_factions_in_main_mod(root, main_mod_faction_maa_map, all_faction_elements)
        if factions_removed_from_main_mod > 0:
            _recache_factions() # Re-cache after removals

    default_created = faction_xml_utils.create_default_faction_if_missing(root, categorized_units, unit_categories, general_units, template_faction_unit_pool, all_units, tier, unit_variant_map, unit_to_tier_map, variant_to_base_map, ck3_maa_definitions, unit_to_class_map=unit_to_class_map, unit_to_description_map=unit_to_description_map, unit_stats_map=unit_stats_map, excluded_units_set=excluded_units_set, is_submod_mode=is_submod_mode)
    if default_created > 0:
        _recache_factions() # Re-cache after addition

    factions_added = faction_xml_utils.sync_factions_from_cultures(root, culture_factions, explicitly_removed_factions=removed_faction_names_for_sync, all_faction_elements=all_faction_elements)
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
    # Process factions in parallel where possible
    print("\nProcessing factions in parallel...")

    # Clear faction_pool_cache before parallel processing to free memory
    faction_pool_cache.clear()
    print("  -> Cleared faction pool cache before parallel processing")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # Create a lock for thread-safe access to the XML tree
    xml_lock = threading.Lock()

    # Prepare a list of faction elements to process
    factions_to_process = list(all_faction_elements)

    # Function to process a single faction
    def process_faction(faction_element):
        faction_name = faction_element.get('name')
        print(f"  -> Processing faction: '{faction_name}'")

        # Create a local copy of the faction element for thread-safe processing
        # We'll process it in isolation and return the modified element
        # Make a deep copy to avoid modifying the original during processing
        import copy
        try:
            faction_copy = copy.deepcopy(faction_element)
            if faction_copy is None:
                print(f"    -> ERROR: Failed to create deep copy of faction '{faction_name}'. Skipping processing.")
                return faction_element, faction_element, 0, []
        except Exception as e:
            print(f"    -> ERROR: Exception during deep copy of faction '{faction_name}': {e}. Skipping processing.")
            return faction_element, faction_element, 0, []

        # Validate that the faction_copy has the expected structure
        if not faction_copy.tag == 'Faction' or not faction_copy.get('name'):
            print(f"    -> ERROR: Invalid faction copy for '{faction_name}'. Skipping processing.")
            return faction_element, faction_element, 0, []

        # Process MenAtArm units for this faction
        try:
            high_confidence_replacements, high_confidence_failures = processing_passes.run_high_confidence_unit_pass(
                None, tier, unit_variant_map, ck3_maa_definitions, unit_to_class_map, unit_to_description_map,
                screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map,
                subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map,
                faction_culture_map, categorized_units, unit_categories, unit_stats_map, all_units,
                excluded_units_set=excluded_units_set, faction_pool_cache=faction_pool_cache,
                faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map,
                faction_to_heritages_map=faction_to_heritages_map, first_pass_threshold=first_pass_threshold,
                llm_helper=llm_helper, faction_to_json_map=faction_to_json_map, all_faction_elements=[faction_copy]
            )
        except Exception as e:
            print(f"    -> ERROR: Exception during high confidence pass for faction '{faction_name}': {e}")
            high_confidence_replacements, high_confidence_failures = 0, []

        # Process Generals and Knights for this faction
        try:
            general_knight_changes, general_knight_failures = unit_management.manage_all_generals_and_knights(
                None, categorized_units, general_units, unit_stats_map, unit_categories,
                screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool,
                faction_to_subculture_map=faction_to_subculture_map, subculture_to_factions_map=subculture_to_factions_map,
                faction_key_to_screen_name_map=faction_key_to_screen_name_map, culture_to_faction_map=culture_to_faction_map,
                tier=tier, unit_to_tier_map=unit_to_tier_map, faction_to_json_map=faction_to_json_map,
                all_units=all_units, unit_to_training_level=unit_to_training_level, excluded_units_set=excluded_units_set,
                faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map,
                faction_to_heritages_map=faction_to_heritages_map, faction_culture_map=faction_culture_map,
                is_submod_mode=is_submod_mode, factions_in_main_mod=factions_in_main_mod, all_faction_elements=[faction_copy]
            )
        except Exception as e:
            print(f"    -> ERROR: Exception during generals/knights pass for faction '{faction_name}': {e}")
            general_knight_changes, general_knight_failures = 0, []

        # Process Levies and Garrisons for this faction
        levy_changes, levy_failures = (0, [])
        garrison_changes, garrison_failures = (0, [])
        if not no_garrison:
            try:
                levy_changes, levy_failures = processing_passes.ensure_levy_structure_and_percentages(
                    None, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool,
                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map,
                    unit_to_class_map, faction_to_json_map, all_units, unit_to_training_level, tier, faction_elite_units,
                    excluded_units_set, faction_pool_cache, faction_to_heritage_map, heritage_to_factions_map,
                    faction_to_heritages_map, destructive_on_failure=False, faction_culture_map=faction_culture_map,
                    is_submod_mode=is_submod_mode, factions_in_main_mod=factions_in_main_mod, all_faction_elements=[faction_copy]
                )
            except Exception as e:
                print(f"    -> ERROR: Exception during levy pass for faction '{faction_name}': {e}")
                levy_changes, levy_failures = 0, []

            try:
                garrison_changes, garrison_failures = processing_passes.ensure_garrison_structure(
                    None, unit_categories, screen_name_to_faction_key_map, faction_key_to_units_map, template_faction_unit_pool,
                    faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map, culture_to_faction_map,
                    unit_to_class_map, general_units, unit_to_training_level, tier, unit_to_tier_map, excluded_units_set,
                    faction_pool_cache, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map,
                    destructive_on_failure=False, faction_to_json_map=faction_to_json_map, all_units=all_units,
                    faction_culture_map=faction_culture_map, is_submod_mode=is_submod_mode, factions_in_main_mod=factions_in_main_mod, all_faction_elements=[faction_copy]
                )
            except Exception as e:
                print(f"    -> ERROR: Exception during garrison pass for faction '{faction_name}': {e}")
                garrison_changes, garrison_failures = 0, []

        # Collect all failures for this faction
        all_failures = high_confidence_failures + general_knight_failures + levy_failures + garrison_failures

        # Apply final fixes for this faction
        try:
            final_fix_changes = processing_passes.run_final_fix_pass(
                None, all_failures, categorized_units, all_units, unit_categories, tier, unit_variant_map, ck3_maa_definitions,
                unit_to_description_map, unit_stats_map, general_units, unit_to_class_map, excluded_units_set,
                screen_name_to_faction_key_map, faction_key_to_units_map, faction_to_subculture_map, subculture_to_factions_map,
                faction_key_to_screen_name_map, culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map,
                faction_to_heritages_map, faction_pool_cache, faction_to_json_map, faction_culture_map, llm_helper,
                unit_to_training_level, faction_elite_units, all_faction_elements=[faction_copy]
            )
        except Exception as e:
            print(f"    -> ERROR: Exception during final fix pass for faction '{faction_name}': {e}")
            final_fix_changes = 0

        total_faction_changes = (
            high_confidence_replacements + general_knight_changes +
            levy_changes + garrison_changes + final_fix_changes
        )

        return faction_element, faction_copy, total_faction_changes, all_failures

    # Process factions in parallel
    total_parallel_changes = 0
    all_parallel_failures = []
    processed_factions = []

    with ThreadPoolExecutor(max_workers=llm_threads) as executor:
        future_to_faction = {executor.submit(process_faction, faction): faction for faction in factions_to_process}
        for future in as_completed(future_to_faction):
            try:
                original_faction, processed_faction, faction_changes, faction_failures = future.result()
                total_parallel_changes += faction_changes
                all_parallel_failures.extend(faction_failures)
                processed_factions.append((original_faction, processed_faction))
            except Exception as exc:
                faction_name = future_to_faction[future].get('name', 'Unknown')
                print(f"Faction '{faction_name}' processing generated an exception: {exc}")
                # Continue processing other factions instead of failing completely
                continue

    # Replace original faction elements with processed ones
    for original_faction, processed_faction in processed_factions:
        # Find the position of the original faction
        for i, child in enumerate(root):
            if child is original_faction:
                # Replace the element
                root[i] = processed_faction
                break

    total_changes += total_parallel_changes

    # --- LLM Pass (Consolidated requests for all failures) ---
    llm_replacements = 0
    if llm_helper and all_parallel_failures and len(all_parallel_failures) < MAX_LLM_FAILURES_THRESHOLD:
        llm_replacements, llm_final_failures = llm_orchestrator.run_llm_unit_assignment_pass(
            llm_helper, all_parallel_failures, time_period_context, llm_threads, llm_batch_size,
            faction_pool_cache, all_units, excluded_units_set, unit_to_tier_map, unit_to_class_map,
            unit_to_description_map, unit_stats_map, screen_name_to_faction_key_map, faction_key_to_units_map,
            faction_to_subculture_map, subculture_to_factions_map, faction_key_to_screen_name_map,
            culture_to_faction_map, faction_to_heritage_map, heritage_to_factions_map, faction_to_heritages_map,
            ck3_maa_definitions
        )
        total_changes += llm_replacements

    # --- Low-Confidence Procedural Fallback (for LLM failures) ---
    low_confidence_replacements = processing_passes.run_low_confidence_unit_pass(
        root, all_parallel_failures, ck3_maa_definitions, unit_to_class_map, unit_variant_map, unit_to_description_map,
        categorized_units, unit_categories, unit_stats_map, all_units, excluded_units_set=excluded_units_set,
        faction_to_heritage_map=faction_to_heritage_map, heritage_to_factions_map=heritage_to_factions_map,
        screen_name_to_faction_key_map=screen_name_to_faction_key_map, faction_key_to_units_map=faction_key_to_units_map,
        llm_helper=llm_helper, unit_to_training_level=unit_to_training_level, faction_elite_units=faction_elite_units,
        faction_to_json_map=faction_to_json_map, faction_culture_map=faction_culture_map, faction_pool_cache=faction_pool_cache,
        faction_to_subculture_map=faction_to_subculture_map, subculture_to_factions_map=subculture_to_factions_map,
        faction_key_to_screen_name_map=faction_key_to_screen_name_map, culture_to_faction_map=culture_to_faction_map,
        faction_to_heritages_map=faction_to_heritages_map, all_faction_elements=all_faction_elements
    )
    total_changes += low_confidence_replacements

    # --- Final Attribute Management Pass ---
    print("\nRunning final attribute management pass...")
    s, se, ng, m = _run_attribute_management_pass(
        root, ck3_maa_definitions, unit_to_class_map, unit_categories, unit_to_num_guns_map, no_siege, all_faction_elements
    )
    total_changes += s + se + ng + m
    print(f"  -> Applied {s} siege, {se} siege_engine_per_unit, {ng} num_guns, and {m} max attribute changes.")

    # --- Final Normalization Pass ---
    # This pass ensures Levy/Garrison percentages sum to 100% and removes any remaining invalid tags.
    print("\nRunning final normalization pass...")
    normalization_changes = unit_management.normalize_all_levy_percentages(root, all_faction_elements)
    total_changes += normalization_changes
    print(f"  -> Applied {normalization_changes} normalization changes.")

    # --- Final XML Output ---
    if total_changes > 0 or submod_tag_added or submod_addon_for_added:
        print(f"\nProcessing complete. Applied {total_changes} total changes. Saving file...")
        shared_utils.indent_xml(root)
        tree.write(units_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Successfully updated '{units_xml_path}'.")
    else:
        print("\nProcessing complete. No changes were made to the XML content.")

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
            print(f"Error: Faction