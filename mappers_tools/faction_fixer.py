import argparse
import concurrent.futures
import os
from mappers_tools import llm_helper
from mappers_tools import unit_management

# Global variables to hold master lists, loaded once at the start of main
_MASTER_UNIT_LIST = []
_CULTURAL_AFFINITY_MAP = {}

def is_unit_culturally_appropriate(unit_key, culture):
    """
    Checks if a unit key is culturally appropriate for a given culture.

    Args:
        unit_key (str): The key of the unit to check.
        culture (str): The culture string (e.g., 'roman', 'barbarian').

    Returns:
        bool: True if the unit is culturally appropriate, False otherwise.
    """
    if culture not in _CULTURAL_AFFINITY_MAP:
        # If culture is unknown, no specific affinity rules apply, so it's not "culturally appropriate"
        # in the context of needing a fix.
        return False 
    
    allowed_prefixes = _CULTURAL_AFFINITY_MAP[culture]
    for prefix in allowed_prefixes:
        if unit_key.startswith(prefix):
            return True
    return False

def process_faction(file_path):
    """
    Encapsulates the entire logic for processing a single faction file.
    This function is designed to be run in parallel by a ThreadPoolExecutor.

    Args:
        file_path (str): The path to the faction XML file.
    """
    print(f"\nProcessing faction file: {file_path}")
    faction_data = unit_management.parse_faction_data(file_path)

    if faction_data is None:
        print(f"  Skipping {file_path} due to parsing errors.")
        return

    faction_name = faction_data['name']
    culture = faction_data['culture']
    current_units = faction_data['units']

    print(f"  Faction: {faction_name}, Culture: {culture}")

    # Phase 1: Core Roster Correction
    flagged_units = []
    for unit_type in ['General', 'Knights', 'Levies', 'Garrison']:
        for unit_key in current_units[unit_type]:
            if not is_unit_culturally_appropriate(unit_key, culture):
                flagged_units.append(unit_key)
    
    if flagged_units:
        print(f"  Flagged core units for correction: {flagged_units}")
        # Filter master unit list to only include culturally appropriate units for the prompt
        cultural_unit_pool = [
            unit for unit in _MASTER_UNIT_LIST 
            if is_unit_culturally_appropriate(unit, culture) or unit.startswith('generic_') # Allow generics as fallback
        ]
        
        roster_correction_prompt = llm_helper.create_roster_correction_prompt(
            culture, flagged_units, cultural_unit_pool
        )
        roster_corrections_response = llm_helper.get_llm_response(roster_correction_prompt)

        if roster_corrections_response and not roster_corrections_response.get("error"):
            unit_management.apply_roster_corrections(file_path, roster_corrections_response)
        else:
            print(f"  Error or no valid corrections from LLM for core roster: {roster_corrections_response.get('error', 'Unknown error')}")
    else:
        print("  No flagged core units found. Skipping core roster correction.")

    # Phase 2: MenAtArm Remapping
    print(f"  Remapping Men-at-Arms for {faction_name}...")
    maa_roster = current_units['MenAtArm']
    
    # For MAA remapping, the cultural pool should be more flexible, including generics
    cultural_unit_pool_for_maa = [
        unit for unit in _MASTER_UNIT_LIST 
        if is_unit_culturally_appropriate(unit, culture) or unit.startswith('generic_')
    ]

    maa_remapping_prompt = llm_helper.create_maa_remapping_prompt(
        culture, maa_roster, cultural_unit_pool_for_maa, _MASTER_UNIT_LIST
    )
    maa_remapping_response = llm_helper.get_llm_response(maa_remapping_prompt)

    if maa_remapping_response and not maa_remapping_response.get("error"):
        unit_management.apply_maa_remapping(file_path, maa_remapping_response)
    else:
        print(f"  Error or no valid remapping from LLM for MenAtArm: {maa_remapping_response.get('error', 'Unknown error')}")

def main():
    """
    Main function to parse arguments, orchestrate file discovery, and manage multi-threaded processing.
    """
    parser = argparse.ArgumentParser(description="Fix faction rosters and Men-at-Arms using LLM assistance.")
    parser.add_argument('--fix-rosters', action='store_true', help="Enable fixing of faction rosters and Men-at-Arms.")
    parser.add_argument('--base-path', type=str, default='.', help="Base path to search for faction XML files.")
    args = parser.parse_args()

    if not args.fix_rosters:
        print("Roster fixing is disabled. Use --fix-rosters to enable.")
        return
    
    print("Starting roster fixing process...")

    # Load master lists once into global variables
    global _MASTER_UNIT_LIST, _CULTURAL_AFFINITY_MAP
    _MASTER_UNIT_LIST = unit_management.get_master_unit_list()
    _CULTURAL_AFFINITY_MAP = unit_management.get_cultural_affinity_map()
    print("Master unit list and cultural affinity map loaded.")

    faction_files = unit_management.find_faction_files(args.base_path)
    if not faction_files:
        print(f"No faction XML files found in '{args.base_path}'. Please check the path and file naming convention (e.g., faction_*.xml).")
        return

    print(f"Found {len(faction_files)} faction files to process.")

    # Multi-threading Logic
    # Use a reasonable number of workers, typically CPU count or a bit more for I/O-bound tasks
    max_workers = os.cpu_count() * 2 if os.cpu_count() else 4 
    print(f"Using ThreadPoolExecutor with {max_workers} workers.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applies the function to each item in the iterable in parallel
        # and returns results in the order the calls were made.
        # We iterate through the results to ensure all futures are completed and exceptions are raised.
        for _ in executor.map(process_faction, faction_files):
            pass # Results are printed within process_faction, so we just need to consume the iterator

    print("\nProcess complete: All faction files have been processed.")

if __name__ == '__main__':
    main()
