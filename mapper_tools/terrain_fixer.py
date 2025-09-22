import os
import re
import csv
import xml.etree.ElementTree as ET
import argparse
import Levenshtein
import json
from collections import defaultdict
import random

# --- Constants ---
NORMAL_MAP_COORDS_COUNT = 50

# --- Helper functions (copied from faction_fixer.py) ---

def prompt_to_create_xml(file_path, root_tag_name):
    """
    Checks if an XML file exists. If not, prompts the user to create it.
    Returns True if the file exists or was created, False otherwise.
    """
    if not os.path.exists(file_path):
        clean_path = os.path.normpath(file_path)
        response = input(f"File not found: '{clean_path}'.\nDo you want to create it? (y/n): ").lower()
        if response == 'y':
            try:
                dir_name = os.path.dirname(file_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                root = ET.Element(root_tag_name)
                tree = ET.ElementTree(root)
                indent_xml(root)
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
                print(f"Successfully created '{clean_path}'.")
                return True
            except Exception as e:
                print(f"Error creating file '{clean_path}': {e}")
                return False
        else:
            print("File creation skipped.")
            return False
    return True

def indent_xml(elem, level=0):
    """Adds indentation to the XML tree for pretty printing."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent_xml(subelem, level + 1)
        # After the loop, the tail of the last subelement should be the parent's closing tag indentation.
        elem[-1].tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# --- New Data Loading Functions ---

def get_faction_key_to_screen_name_map(tsv_dir):
    """Parses the faction_tables TSV files to get a map of faction key to screen name."""
    faction_map = {}
    if not os.path.isdir(tsv_dir):
        print(f"Error: Faction tables directory not found at {tsv_dir}")
        return faction_map

    for filename in os.listdir(tsv_dir):
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None); next(reader, None) # Skip metadata
                    header = next(reader, None)
                    if not header: continue

                    try:
                        key_idx = header.index("key")
                        screen_name_idx = header.index("screen_name")
                    except ValueError:
                        continue

                    for row in reader:
                        if len(row) > key_idx and len(row) > screen_name_idx and row[key_idx]:
                            faction_map[row[key_idx]] = row[screen_name_idx]
            except Exception as e:
                print(f"Error processing faction table TSV file {filename}: {e}")
    return faction_map

def get_faction_to_subculture_map(tsv_dir):
    """
    Parses the faction_tables TSV files to get maps between faction keys and subcultures.
    Returns {faction_key: subculture_key}
    """
    faction_to_subculture_map = {}

    if not os.path.isdir(tsv_dir):
        print(f"Error: Faction tables directory not found at {tsv_dir}")
        return faction_to_subculture_map

    for filename in os.listdir(tsv_dir):
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None); next(reader, None) # Skip metadata
                    header = next(reader, None)
                    if not header: continue

                    try:
                        key_idx = header.index("key")
                        subculture_idx = header.index("subculture")
                    except ValueError:
                        continue

                    for row in reader:
                        if len(row) > key_idx and len(row) > subculture_idx and row[key_idx]:
                            faction_key = row[key_idx]
                            subculture_key = row[subculture_idx]
                            if faction_key and subculture_key:
                                faction_to_subculture_map[faction_key] = subculture_key
            except Exception as e:
                print(f"Error processing faction table TSV file {filename}: {e}")
    return faction_to_subculture_map

def get_all_factions_from_units_xml(xml_file_path):
    """
    Parses the Attila Factions XML to get a set of all faction names.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        factions = {
            faction.get('name')
            for faction in root.findall('.//Faction')
            if faction.get('name')
        }
        return factions
    except ET.ParseError as e:
        print(f"Error parsing Attila Factions XML file {xml_file_path}: {e}")
        return set()
    except FileNotFoundError:
        print(f"Error: Attila Factions XML file not found at {xml_file_path}")
        return set()


def get_ck3_building_keys(directory):
    """
    Extracts all building definition keys from CK3 game files.
    """
    building_keys = set()
    # This regex finds potential building definition starts: `building_key = {`
    building_start_pattern = re.compile(r'^([a-zA-Z0-9_]+)\s*=\s*\{', re.MULTILINE) # MODIFIED REGEX

    if not os.path.isdir(directory):
        print(f"Error: CK3 buildings directory not found: {directory}")
        return set()

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
                    content = f.read()
                    for match in building_start_pattern.finditer(content):
                        building_keys.add(match.group(1))
            except Exception as e:
                print(f"Error reading or parsing {filename}: {e}")

    return building_keys

def get_ck3_terrain_types(directory):
    """
    Extracts all terrain type keys from CK3 game files.
    """
    terrain_keys = set()
    # This regex finds potential terrain definition starts: `terrain_key = {`
    terrain_start_pattern = re.compile(r'^([a-zA-Z0-9_]+)\s*=\s*\{', re.MULTILINE) # MODIFIED REGEX

    if not os.path.isdir(directory):
        print(f"Error: CK3 terrain types directory not found: {directory}")
        return set()

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
                    content = f.read()
                    for match in terrain_start_pattern.finditer(content):
                        terrain_keys.add(match.group(1))
            except Exception as e:
                print(f"Error reading or parsing {filename}: {e}")

    return terrain_keys

def get_map_index(directory, terrain_folder_name):
    """
    Extracts the map index for a given terrain folder name from campaign_map_playable_areas_tables.
    """
    if not os.path.isdir(directory):
        print(f"Error: Attila playable areas directory not found: {directory}")
        return None

    lower_terrain_folder_name = terrain_folder_name.lower()

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None); next(reader, None) # Skip metadata
                    header = next(reader, None)
                    if not header: continue

                    try:
                        terrain_folder_idx = header.index("terrain_folder")
                        index_idx = header.index("index")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if len(row) > terrain_folder_idx and len(row) > index_idx and row[terrain_folder_idx]:
                            if row[terrain_folder_idx].lower() == lower_terrain_folder_name:
                                return row[index_idx] # Return the index as a string
            except Exception as e:
                print(f"Error processing playable areas TSV file {filename}: {e}")
    return None

def get_attila_preset_coords(directory, required_map_index):
    """
    Extracts all Attila battle preset keys and their coordinates, filtered by a specific campaign map index.
    """
    preset_coords = {}

    if required_map_index is None:
        print("Warning: No required_map_index provided. Returning empty preset coordinates.")
        return {}

    if not os.path.isdir(directory):
        print(f"Error: Attila campaign_battle_presets_tables directory not found: {directory}")
        return {}

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None); next(reader, None) # Skip metadata
                    header = next(reader, None)
                    if not header: continue

                    try:
                        key_idx = header.index("key")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if (len(row) > key_idx and len(row) > coord_x_idx and
                            len(row) > coord_y_idx and len(row) > campaign_map_idx and
                            row[key_idx]):
                            
                            # Filter by campaign_map index
                            if row[campaign_map_idx] == required_map_index:
                                preset_key = row[key_idx]
                                coord_x = row[coord_x_idx]
                                coord_y = row[coord_y_idx]
                                preset_coords[preset_key] = {'x': coord_x, 'y': coord_y}
            except Exception as e:
                print(f"Error processing preset TSV file {filename}: {e}")
    return preset_coords

def get_attila_settlement_presets(directory, required_map_index):
    """
    Extracts Attila settlement battle presets, filtered by battle_type and campaign_map index.
    """
    settlement_presets = []

    if required_map_index is None:
        print("Warning: No required_map_index provided. Returning empty settlement presets.")
        return []

    if not os.path.isdir(directory):
        print(f"Error: Attila campaign_battle_presets_tables directory not found: {directory}")
        return []

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    next(reader, None); next(reader, None) # Skip metadata
                    header = next(reader, None)
                    if not header: continue

                    try:
                        key_idx = header.index("key")
                        battle_type_idx = header.index("battle_type")
                        tile_upgrade_idx = header.index("tile_upgrade")
                        is_unique_settlement_idx = header.index("is_unique_settlement")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if (len(row) > key_idx and len(row) > battle_type_idx and
                            len(row) > tile_upgrade_idx and len(row) > is_unique_settlement_idx and
                            len(row) > coord_x_idx and len(row) > coord_y_idx and
                            len(row) > campaign_map_idx and row[key_idx]):
                            
                            # Filter by campaign_map index and battle_type
                            if row[campaign_map_idx] == required_map_index and \
                               (row[battle_type_idx] == 'settlement_standard' or row[battle_type_idx] == 'settlement_unfortified'):
                                
                                preset_data = {
                                    'key': row[key_idx],
                                    'battle_type': row[battle_type_idx],
                                    'tile_upgrade': row[tile_upgrade_idx],
                                    'is_unique_settlement': row[is_unique_settlement_idx],
                                    'x': row[coord_x_idx],
                                    'y': row[coord_y_idx]
                                }
                                settlement_presets.append(preset_data)
            except Exception as e:
                print(f"Error processing settlement preset TSV file {filename}: {e}")
    return settlement_presets


# --- Matching Logic ---

def _normalize_name_for_match(name, prefixes_to_remove=None):
    """Normalizes a string for Levenshtein comparison."""
    if not name:
        return ""
    normalized = name.lower().replace("_", " ")
    if prefixes_to_remove:
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
    return normalized

def _find_best_preset_match(ck3_key, attila_preset_keys, threshold=0.85):
    """
    Finds the best matching Attila preset key for a given CK3 building key or terrain type
    using Levenshtein distance. Normalizes names by removing common prefixes.
    """
    if not ck3_key or not attila_preset_keys:
        return None

    ck3_normalized = _normalize_name_for_match(ck3_key, prefixes_to_remove=['wonder_', 'building_', 'terrain_'])

    best_match_preset = None
    highest_ratio = 0

    for preset_key in attila_preset_keys:
        preset_normalized = _normalize_name_for_match(preset_key, prefixes_to_remove=['preset_'])
        ratio = Levenshtein.ratio(ck3_normalized, preset_normalized)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match_preset = preset_key

    if best_match_preset and highest_ratio >= threshold:
        return best_match_preset
    return None

def _generate_nearby_coords(base_x_str, base_y_str, count=NORMAL_MAP_COORDS_COUNT, max_delta=0.1):
    """
    Generates a list of randomized coordinate pairs around a central point.
    Coordinates are clamped to [0.0, 0.999] and formatted to 3 decimal places.
    """
    try:
        base_x = float(base_x_str)
        base_y = float(base_y_str)
    except ValueError:
        print(f"Error: Invalid base coordinates '{base_x_str}', '{base_y_str}'. Cannot generate nearby coordinates.")
        return []

    nearby_coords = []
    for _ in range(count):
        delta_x = random.uniform(-max_delta, max_delta)
        delta_y = random.uniform(-max_delta, max_delta)

        new_x = base_x + delta_x
        new_y = base_y + delta_y

        # Clamp coordinates to [0.0, 0.999]
        new_x = max(0.0, min(0.999, new_x))
        new_y = max(0.0, min(0.999, new_y))

        nearby_coords.append((f"{new_x:.3f}", f"{new_y:.3f}"))
    return nearby_coords

# --- Core Processing Function ---

def process_settlement_maps(root, settlement_presets, all_valid_factions, screen_name_to_key_map, faction_to_subculture_map):
    """
    Processes settlement map configurations, matching factions to settlement presets
    and generating the XML structure.
    """
    print("\n--- Processing Settlement Maps ---")
    settlement_changes = 0

    # 1. Find or create <Settlement_Maps> element
    settlement_maps_element = root.find('Settlement_Maps')
    if settlement_maps_element is None:
        settlement_maps_element = ET.SubElement(root, 'Settlement_Maps')
        settlement_changes += 1
        print("Created new <Settlement_Maps> element.")
    else:
        # Remove existing children to ensure a clean build
        removed_count = 0
        for child in list(settlement_maps_element):
            settlement_maps_element.remove(child)
            removed_count += 1
        if removed_count > 0:
            print(f"Removed {removed_count} existing child elements from <Settlement_Maps>.")
            settlement_changes += removed_count

    if not settlement_presets:
        print("Warning: No Attila settlement presets found. Skipping settlement map generation.")
        return settlement_changes

    if not all_valid_factions:
        print("Warning: No valid factions found. Skipping settlement map generation.")
        return settlement_changes

    # Create reverse map for screen names to faction keys
    key_to_screen_name_map = {v: k for k, v in screen_name_to_key_map.items()}

    print(f"Found {len(all_valid_factions)} valid factions and {len(settlement_presets)} settlement presets.")

    # Iterate through each faction
    for faction_screen_name in sorted(list(all_valid_factions)):
        faction_key = screen_name_to_key_map.get(faction_screen_name)
        if not faction_key:
            print(f"  -> WARNING: Could not find faction key for screen name '{faction_screen_name}'. Skipping.")
            continue

        faction_subculture = faction_to_subculture_map.get(faction_key)
        
        matched_presets_for_faction = defaultdict(list) # {battle_type: [preset1, preset2, ...]}

        # Filter settlement presets for the current faction
        for preset in settlement_presets:
            tile_upgrade_normalized = _normalize_name_for_match(preset['tile_upgrade'])
            faction_screen_normalized = _normalize_name_for_match(faction_screen_name)
            
            # Check against faction screen name
            ratio_screen = Levenshtein.ratio(tile_upgrade_normalized, faction_screen_normalized)
            
            # Check against subculture if available
            ratio_subculture = 0
            if faction_subculture:
                faction_subculture_normalized = _normalize_name_for_match(faction_subculture, prefixes_to_remove=['subculture_'])
                ratio_subculture = Levenshtein.ratio(tile_upgrade_normalized, faction_subculture_normalized)

            # Use the higher ratio for matching
            if max(ratio_screen, ratio_subculture) >= 0.7: # Threshold for matching
                matched_presets_for_faction[preset['battle_type']].append(preset)
        
        if not matched_presets_for_faction:
            # print(f"  -> INFO: No settlement presets found for faction '{faction_screen_name}'.")
            continue

        print(f"  - Processing settlement maps for faction '{faction_screen_name}' (key: {faction_key}).")

        # XML Generation for each battle_type
        for battle_type in sorted(matched_presets_for_faction.keys()):
            presets_for_battle_type = matched_presets_for_faction[battle_type]
            
            # Create <Settlement> element
            settlement_element = ET.SubElement(settlement_maps_element, 'Settlement', {
                'faction': faction_screen_name,
                'battle_type': battle_type
            })
            settlement_changes += 1

            # Variant Selection: Max 10 variants, prioritize unique, then random
            unique_presets = [p for p in presets_for_battle_type if p['is_unique_settlement'].lower() == 'true']
            non_unique_presets = [p for p in presets_for_battle_type if p['is_unique_settlement'].lower() != 'true']

            selected_variants = []
            selected_variants.extend(unique_presets)

            remaining_slots = 10 - len(selected_variants)
            if remaining_slots > 0:
                # Randomly sample from non-unique presets if needed
                if len(non_unique_presets) > remaining_slots:
                    selected_variants.extend(random.sample(non_unique_presets, remaining_slots))
                else:
                    selected_variants.extend(non_unique_presets)
            
            # If still more than 10 (e.g., many unique presets), trim to 10
            selected_variants = selected_variants[:10]

            if not selected_variants:
                print(f"    -> WARNING: No suitable variants selected for battle_type '{battle_type}' for faction '{faction_screen_name}'. Removing empty <Settlement> tag.")
                settlement_maps_element.remove(settlement_element)
                settlement_changes -= 1
                continue

            for variant_preset in selected_variants:
                variant_element = ET.SubElement(settlement_element, 'Variant', {
                    'tile_upgrade': variant_preset['tile_upgrade'],
                    'is_unique_settlement': variant_preset['is_unique_settlement']
                })
                settlement_changes += 1

                map_element = ET.SubElement(variant_element, 'Map', {
                    'x': variant_preset['x'],
                    'y': variant_preset['y']
                })
                settlement_changes += 1
    
    if settlement_changes > 0:
        print(f"Successfully generated settlement maps. Total changes: {settlement_changes}.")
    else:
        print("No settlement maps generated or updated.")

    return settlement_changes


def process_terrains_xml(terrains_xml_path, ck3_building_keys, attila_preset_coords, attila_map_name, llm_helper=None, llm_batch_size=50, ck3_terrain_types=None,
                         settlement_presets=None, all_valid_factions=None, screen_name_to_key_map=None, faction_to_subculture_map=None):
    """
    Processes the terrains XML file to map CK3 buildings to Attila battle presets
    and CK3 terrain types to Attila normal map presets, and generates settlement maps.
    """
    print(f"\nProcessing file: {terrains_xml_path}")

    try:
        tree = ET.parse(terrains_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file {terrains_xml_path}: {e}. Skipping.")
        return 0

    total_changes = 0

    # NEW: Create or update the top-level <Map> tag for the campaign map name
    top_level_map_element = root.find('Map')
    if top_level_map_element is None:
        top_level_map_element = ET.SubElement(root, 'Map')
        total_changes += 1
        print("Created new top-level <Map> element for campaign map name.")

    if attila_map_name:
        if top_level_map_element.get('name') != attila_map_name:
            top_level_map_element.set('name', attila_map_name)
            print(f"Set top-level <Map> name attribute to '{attila_map_name}'.")
            total_changes += 1
    elif 'name' not in top_level_map_element.attrib:
        print("Warning: No --attila-map specified and top-level <Map> has no existing name. Consider adding one.")


    # --- Historic Maps (Buildings) Processing ---
    # Ensure <Terrains><Historic_maps> structure
    historic_maps_container = root.find('Historic_maps')
    if historic_maps_container is None:
        historic_maps_container = ET.SubElement(root, 'Historic_maps')
        total_changes += 1 # Count creation of new element

    # Remove all existing <Map> tags from within <Historic_maps> to start fresh
    removed_maps_count = 0
    for map_element_to_remove in list(historic_maps_container.findall('Map')):
        historic_maps_container.remove(map_element_to_remove)
        removed_maps_count += 1
    if removed_maps_count > 0:
        print(f"Removed {removed_maps_count} existing historic map tags from the XML.")
        total_changes += removed_maps_count

    if attila_map_name:
        # The old logic for setting map_element name is removed as it's now handled by top_level_map_element
        pass
    elif 'name' not in historic_maps_container.attrib: # Check if historic_maps_container itself needs a name if no top-level map
        pass # No longer setting name here, handled by top-level map

    # Get all available Attila preset keys
    attila_preset_keys = set(attila_preset_coords.keys())
    if not attila_preset_keys:
        print("Error: No Attila battle presets found. Cannot map buildings or terrains. Aborting.")
        return 0

    print(f"Found {len(ck3_building_keys)} CK3 building keys and {len(attila_preset_keys)} Attila battle presets.")

    matched_buildings = {} # {ck3_key: attila_preset_key}
    high_confidence_building_failures = []

    # --- Stage 1: High-Confidence Procedural Pass for Buildings ---
    print("\nRunning high-confidence procedural pass for building assignments...")
    for ck3_key in sorted(list(ck3_building_keys)): # Sort for deterministic output
        best_match = _find_best_preset_match(ck3_key, attila_preset_keys, threshold=0.85)
        if best_match:
            matched_buildings[ck3_key] = best_match
            print(f"  -> High-confidence match: CK3 '{ck3_key}' -> Attila '{best_match}'.")
        else:
            high_confidence_building_failures.append({'id': ck3_key, 'preset_pool': sorted(list(attila_preset_keys))})
            print(f"  -> No high-confidence match for CK3 '{ck3_key}'. Queued for LLM/low-confidence processing.")

    print(f"High-confidence pass complete. Matched {len(matched_buildings)} buildings. {len(high_confidence_building_failures)} failures.")

    # --- Stage 2: LLM Pass for Buildings ---
    llm_building_replacements_made = 0
    llm_building_failures = []
    if llm_helper and high_confidence_building_failures:
        print(f"\nAttempting to resolve {len(high_confidence_building_failures)} missing building assignments with LLM...")

        requests_for_llm = high_confidence_building_failures # Already in the correct format
        element_map = {req['id']: req for req in high_confidence_building_failures} # Map ID back to original request data

        if requests_for_llm:
            all_llm_results = {}
            num_batches = (len(requests_for_llm) + llm_batch_size - 1) // llm_batch_size

            for i in range(num_batches):
                start_index = i * llm_batch_size
                end_index = min((i + 1) * llm_batch_size, len(requests_for_llm))
                current_batch = requests_for_llm[start_index:end_index]
                print(f"  -> Sending LLM batch {i+1}/{num_batches} with {len(current_batch)} requests...")

                batch_llm_results = llm_helper.get_batch_building_assignments(current_batch)
                if batch_llm_results:
                    all_llm_results.update(batch_llm_results)
                else:
                    print(f"  -> WARNING: LLM did not provide any valid replacements for batch {i+1}.")

            if all_llm_results:
                print(f"Applying LLM suggestions from {len(all_llm_results)} total resolved requests...")
                resolved_ids = set()
                for req_id, chosen_preset in all_llm_results.items():
                    if chosen_preset:
                        matched_buildings[req_id] = chosen_preset
                        llm_building_replacements_made += 1
                        resolved_ids.add(req_id)
                        print(f"  -> LLM SUCCESS: CK3 '{req_id}' -> Attila '{chosen_preset}'.")
                    else:
                        print(f"  -> LLM WARNING: No chosen preset for '{req_id}'.")

                for req_data in high_confidence_building_failures:
                    if req_data['id'] not in resolved_ids:
                        llm_building_failures.append(req_data)
            else:
                print("LLM did not provide any valid replacements.")
                llm_building_failures.extend(high_confidence_building_failures)
        else:
            print("\nLLM integration is disabled or no buildings required LLM intervention.")
            llm_building_failures.extend(high_confidence_building_failures)

    print(f"LLM pass complete. Matched {llm_building_replacements_made} buildings. {len(llm_building_failures)} remaining failures.")

    # --- Stage 3: Low-Confidence Procedural Pass for Buildings ---
    low_confidence_building_replacements = 0
    unmatchable_buildings = []
    if llm_building_failures:
        print("\nRunning low-confidence procedural pass for remaining building assignments...")
        for failure_data in llm_building_failures:
            ck3_key = failure_data['id']
            best_match = _find_best_preset_match(ck3_key, attila_preset_keys, threshold=0.60) # Lower threshold
            if best_match:
                matched_buildings[ck3_key] = best_match
                low_confidence_building_replacements += 1
                print(f"  -> Low-confidence match: CK3 '{ck3_key}' -> Attila '{best_match}'.")
            else:
                unmatchable_buildings.append(ck3_key)
                print(f"  -> WARNING: No low-confidence match found for CK3 '{ck3_key}'. This building will not be mapped.")

    print(f"Low-confidence pass complete. Matched {low_confidence_building_replacements} buildings. {len(unmatchable_buildings)} unmatchable buildings.")

    total_changes += len(matched_buildings)

    # --- Final XML Population for Buildings ---
    if matched_buildings:
        print("\nPopulating XML with matched buildings...")
        for ck3_key, attila_preset_key in sorted(matched_buildings.items()): # Sort for deterministic output
            coords = attila_preset_coords.get(attila_preset_key)
            if coords:
                # MODIFIED: Create <Map> tag directly under historic_maps_container
                map_element_for_building = ET.SubElement(historic_maps_container, 'Map', {
                    'ck3_building_key': ck3_key,
                    'x': coords['x'],
                    'y': coords['y']
                })
                # print(f"  - Added <Map ck3_building_key='{ck3_key}' x='{coords['x']}' y='{coords['y']}'/>")
            else:
                print(f"  -> ERROR: Coordinates not found for Attila preset '{attila_preset_key}' (matched from CK3 '{ck3_key}'). Skipping.")
                total_changes -= 1 # Decrement if we couldn't add it

    # --- Normal Maps (Terrains) Processing ---
    normal_maps_changes = 0
    if ck3_terrain_types:
        print("\n--- Processing Normal Maps (Terrains) ---")
        normal_maps_element = root.find('Normal_maps')
        if normal_maps_element is None:
            normal_maps_element = ET.SubElement(root, 'Normal_maps')
            normal_maps_changes += 1

        # Remove all existing <Terrain> tags to start fresh
        removed_terrains_count = 0
        for terrain_element in list(normal_maps_element.findall('Terrain')):
            normal_maps_element.remove(terrain_element)
            removed_terrains_count += 1
        if removed_terrains_count > 0:
            print(f"Removed {removed_terrains_count} existing <Terrain> tags from the XML.")
            normal_maps_changes += removed_terrains_count

        print(f"Found {len(ck3_terrain_types)} CK3 terrain types.")

        matched_terrains = {} # {ck3_terrain_type: attila_preset_key}
        high_confidence_terrain_failures = []

        # --- Stage 1: High-Confidence Procedural Pass for Terrains ---
        print("\nRunning high-confidence procedural pass for terrain assignments...")
        for ck3_terrain_key in sorted(list(ck3_terrain_types)): # Sort for deterministic output
            best_match = _find_best_preset_match(ck3_terrain_key, attila_preset_keys, threshold=0.85)
            if best_match:
                matched_terrains[ck3_terrain_key] = best_match
                print(f"  -> High-confidence match: CK3 '{ck3_terrain_key}' -> Attila '{best_match}'.")
            else:
                high_confidence_terrain_failures.append({'id': ck3_terrain_key, 'preset_pool': sorted(list(attila_preset_keys))})
                print(f"  -> No high-confidence match for CK3 '{ck3_terrain_key}'. Queued for LLM/low-confidence processing.")

        print(f"High-confidence pass complete. Matched {len(matched_terrains)} terrains. {len(high_confidence_terrain_failures)} failures.")

        # --- Stage 2: LLM Pass for Terrains ---
        llm_terrain_replacements_made = 0
        llm_terrain_failures = []
        if llm_helper and high_confidence_terrain_failures:
            print(f"\nAttempting to resolve {len(high_confidence_terrain_failures)} missing terrain assignments with LLM...")

            requests_for_llm = high_confidence_terrain_failures # Already in the correct format
            element_map = {req['id']: req for req in high_confidence_terrain_failures} # Map ID back to original request data

            if requests_for_llm:
                all_llm_results = {}
                num_batches = (len(requests_for_llm) + llm_batch_size - 1) // llm_batch_size

                for i in range(num_batches):
                    start_index = i * llm_batch_size
                    end_index = min((i + 1) * llm_batch_size, len(requests_for_llm))
                    current_batch = requests_for_llm[start_index:end_index]
                    print(f"  -> Sending LLM batch {i+1}/{num_batches} with {len(current_batch)} requests...")

                    batch_llm_results = llm_helper.get_batch_terrain_assignments(current_batch)
                    if batch_llm_results:
                        all_llm_results.update(batch_llm_results)
                    else:
                        print(f"  -> WARNING: LLM did not provide any valid replacements for batch {i+1}.")

                if all_llm_results:
                    print(f"Applying LLM suggestions from {len(all_llm_results)} total resolved requests...")
                    resolved_ids = set()
                    for req_id, chosen_preset in all_llm_results.items():
                        if chosen_preset:
                            matched_terrains[req_id] = chosen_preset
                            llm_terrain_replacements_made += 1
                            resolved_ids.add(req_id)
                            print(f"  -> LLM SUCCESS: CK3 Terrain '{req_id}' -> Attila '{chosen_preset}'.")
                        else:
                            print(f"  -> LLM WARNING: No chosen preset for CK3 Terrain '{req_id}'.")

                    for req_data in high_confidence_terrain_failures:
                        if req_data['id'] not in resolved_ids:
                            llm_terrain_failures.append(req_data)
                else:
                    print("LLM did not provide any valid replacements for terrains.")
                    llm_terrain_failures.extend(high_confidence_terrain_failures)
            else:
                print("No terrain assignment failures suitable for LLM processing.")
        else:
            print("\nLLM integration is disabled or no terrains required LLM intervention.")
            llm_terrain_failures.extend(high_confidence_terrain_failures)

        print(f"LLM pass complete. Matched {llm_terrain_replacements_made} terrains. {len(llm_terrain_failures)} remaining failures.")

        # --- Stage 3: Low-Confidence Procedural Pass for Terrains ---
        low_confidence_terrain_replacements = 0
        unmatchable_terrains = []
        if llm_terrain_failures:
            print("\nRunning low-confidence procedural pass for remaining terrain assignments...")
            for failure_data in llm_terrain_failures:
                ck3_terrain_key = failure_data['id']
                best_match = _find_best_preset_match(ck3_terrain_key, attila_preset_keys, threshold=0.60) # Lower threshold
                if best_match:
                    matched_terrains[ck3_terrain_key] = best_match
                    low_confidence_terrain_replacements += 1
                    print(f"  -> Low-confidence match: CK3 Terrain '{ck3_terrain_key}' -> Attila '{best_match}'.")
                else:
                    unmatchable_terrains.append(ck3_terrain_key)
                    print(f"  -> WARNING: No low-confidence match found for CK3 Terrain '{ck3_terrain_key}'. This terrain will not be mapped.")

        print(f"Low-confidence pass complete. Matched {low_confidence_terrain_replacements} terrains. {len(unmatchable_terrains)} unmatchable terrains.")

        normal_maps_changes += len(matched_terrains) * (1 + NORMAL_MAP_COORDS_COUNT) # 1 Terrain tag + NORMAL_MAP_COORDS_COUNT Map tags per terrain

        # --- Final XML Population for Terrains ---
        if matched_terrains:
            print("\nPopulating XML with matched terrains...")
            for ck3_terrain_key, attila_preset_key in sorted(matched_terrains.items()): # Sort for deterministic output
                coords = attila_preset_coords.get(attila_preset_key)
                if coords:
                    terrain_element = ET.SubElement(normal_maps_element, 'Terrain', {
                        'ck3_name': ck3_terrain_key
                    })
                    nearby_coords = _generate_nearby_coords(coords['x'], coords['y'])
                    for x, y in nearby_coords:
                        ET.SubElement(terrain_element, 'Map', {'x': x, 'y': y})
                    # print(f"  - Added <Terrain ck3_name='{ck3_terrain_key}'> with {len(nearby_coords)} map points.")
                else:
                    print(f"  -> ERROR: Coordinates not found for Attila preset '{attila_preset_key}' (matched from CK3 Terrain '{ck3_terrain_key}'). Skipping.")
                    normal_maps_changes -= (1 + NORMAL_MAP_COORDS_COUNT) # Decrement if we couldn't add it

    total_changes += normal_maps_changes

    # --- Settlement Maps Processing ---
    settlement_maps_changes = process_settlement_maps(root, settlement_presets, all_valid_factions, screen_name_to_key_map, faction_to_subculture_map)
    total_changes += settlement_maps_changes

    if total_changes > 0:
        summary_parts = []
        if removed_maps_count > 0: summary_parts.append(f"removed {removed_maps_count} old historic map tags") # MODIFIED LOGGING
        if len(matched_buildings) > 0: summary_parts.append(f"added {len(matched_buildings)} new historic map tags") # MODIFIED LOGGING
        if llm_building_replacements_made > 0: summary_parts.append(f"LLM resolved {llm_building_replacements_made} building assignments")
        if len(unmatchable_buildings) > 0: summary_parts.append(f"{len(unmatchable_buildings)} buildings unmatchable")

        if ck3_terrain_types: # Only add terrain summary if terrain processing was enabled
            if removed_terrains_count > 0: summary_parts.append(f"removed {removed_terrains_count} old terrain tags")
            if len(matched_terrains) > 0: summary_parts.append(f"added {len(matched_terrains)} new terrain tags")
            if llm_terrain_replacements_made > 0: summary_parts.append(f"LLM resolved {llm_terrain_replacements_made} terrain assignments")
            if len(unmatchable_terrains) > 0: summary_parts.append(f"{len(unmatchable_terrains)} terrains unmatchable")

        if settlement_maps_changes > 0: summary_parts.append(f"generated {settlement_maps_changes} settlement map elements")


        print(f"Finished processing {terrains_xml_path}. Summary: {', '.join(summary_parts)}.")

        indent_xml(root)
        tree.write(terrains_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"Successfully updated '{terrains_xml_path}'.")
    else:
        print(f"No changes were made to {terrains_xml_path}.")

    return total_changes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligently map CK3 buildings to Attila battle preset coordinates in terrains.xml.")
    parser.add_argument("--cultures-xml-path", required=True, help="Path to the target Cultures XML file (for context, not modified).")
    parser.add_argument("--terrains-xml-path", required=True, help="Path to the target Terrains XML file to be modified.")
    parser.add_argument("--factions-xml-path", required=True, help="Path to the Attila Factions XML file for faction reconciliation.") # NEW ARG
    parser.add_argument("--ck3-common-path", required=True, help="Path to the Crusader Kings III 'common' directory.")
    parser.add_argument("--attila-db-path", required=True, help="Path to the Attila debug database directory (e.g., 'debug/919ad/attila/db').")
    parser.add_argument("--attila-map", required=True, help="Name to set for the <Map> element in terrains.xml (e.g., 'campaign_map').")
    # LLM Arguments
    parser.add_argument("--use-llm", action='store_true', help="Enable LLM integration to fix missing building assignments.")
    parser.add_argument("--llm-model", default="ollama/llama3", help="The model name to use with the LLM helper (e.g., 'ollama/llama3', 'gpt-4').")
    parser.add_argument("--llm-api-base", help="The base URL for the LLM API server (for local models).")
    parser.add_argument("--llm-api-key", help="The API key for the LLM service.")
    parser.add_argument("--llm-cache-dir", default="mapper_tools/llm_cache", help="Directory for LLM cache files.")
    parser.add_argument("--llm-cache-tag", help="A unique tag for partitioning the LLM cache (e.g., 'AGOT').")
    parser.add_argument("--llm-batch-size", type=int, default=50, help="The maximum number of building assignment requests to send to the LLM in a single batch.")
    parser.add_argument("--clear-llm-cache", action='store_true', help="If set, clears the LLM cache for the specified --llm-cache-tag before processing.")
    args = parser.parse_args()

    if args.use_llm and not args.llm_cache_tag:
        parser.error("--llm-cache-tag is required when --use-llm is specified.")

    if not prompt_to_create_xml(args.terrains_xml_path, 'Terrains'):
        print("Terrains XML file is required to proceed. Aborting.")
        exit()

    # Construct dynamic paths
    ck3_buildings_dir = os.path.join(args.ck3_common_path, "buildings")
    attila_presets_dir = os.path.join(args.attila_db_path, "campaign_battle_presets_tables")
    ck3_terrain_types_dir = os.path.join(args.ck3_common_path, "terrain_types") # NEW: Add terrain types dir
    attila_playable_areas_dir = os.path.join(args.attila_db_path, "campaign_map_playable_areas_tables") # NEW: Add playable areas directory
    factions_tables_dir = os.path.join(args.attila_db_path, "factions_tables") # NEW: Add factions tables dir

    print("Starting terrain building mapping process...")

    # Initialize LLM Helper if requested
    llm_helper = None
    if args.use_llm:
        try:
            from mapper_tools.llm_helper import LLMHelper
            llm_helper = LLMHelper(
                model=args.llm_model,
                cache_dir=args.llm_cache_dir,
                api_base=args.llm_api_base,
                api_key=args.llm_api_key,
                cache_tag=args.llm_cache_tag
            )
            print("LLM Helper initialized.")
            if args.clear_llm_cache:
                print(f"Clearing LLM cache for tag '{args.llm_cache_tag}'...")
                llm_helper.clear_cache()
                print("LLM cache cleared.")
        except ImportError:
            print("Warning: 'litellm' library not found. Please run 'pip install litellm' to use the LLM feature.")
            llm_helper = None
        except Exception as e:
            print(f"Error initializing LLM Helper: {e}")
            llm_helper = None

    # Load data
    ck3_building_keys = get_ck3_building_keys(ck3_buildings_dir)
    if not ck3_building_keys:
        print("No CK3 building keys found. Aborting.")
        exit()
    print(f"Loaded {len(ck3_building_keys)} CK3 building keys.")

    # NEW: Get map index
    map_index = get_map_index(attila_playable_areas_dir, args.attila_map)
    if map_index is None:
        print(f"Error: The specified --attila-map '{args.attila_map}' could not be found in the_campaign_map_playable_areas_tables. Aborting.")
        exit()
    print(f"Found map index '{map_index}' for Attila map '{args.attila_map}'.")

    attila_preset_coords = get_attila_preset_coords(attila_presets_dir, map_index) # Modified call
    if not attila_preset_coords:
        print(f"No Attila battle presets found for map '{args.attila_map}'. Aborting.") # Updated log message
        exit()
    print(f"Loaded {len(attila_preset_coords)} Attila battle presets for map '{args.attila_map}' with coordinates.") # Updated log message

    # NEW: Load CK3 terrain types
    ck3_terrain_types = get_ck3_terrain_types(ck3_terrain_types_dir)
    if not ck3_terrain_types:
        print("No CK3 terrain types found. Normal maps will not be generated.")
    else:
        print(f"Loaded {len(ck3_terrain_types)} CK3 terrain types.")

    # NEW: Load data for settlement maps
    faction_key_to_screen_name_map = get_faction_key_to_screen_name_map(factions_tables_dir)
    screen_name_to_key_map = {v: k for k, v in faction_key_to_screen_name_map.items()}
    faction_to_subculture_map = get_faction_to_subculture_map(factions_tables_dir)
    all_valid_factions = get_all_factions_from_units_xml(args.factions_xml_path)
    settlement_presets = get_attila_settlement_presets(attila_presets_dir, map_index)

    if not all_valid_factions:
        print("Warning: No valid factions loaded from Factions XML. Settlement maps will be limited.")
    else:
        print(f"Loaded {len(all_valid_factions)} valid factions from '{args.factions_xml_path}'.")

    if not settlement_presets:
        print("Warning: No Attila settlement presets loaded. Settlement maps will not be generated.")
    else:
        print(f"Loaded {len(settlement_presets)} Attila settlement presets.")


    # Process the XML
    process_terrains_xml(
        args.terrains_xml_path,
        ck3_building_keys,
        attila_preset_coords,
        args.attila_map,
        llm_helper,
        llm_batch_size=args.llm_batch_size,
        ck3_terrain_types=ck3_terrain_types, # NEW: Pass terrain types
        settlement_presets=settlement_presets, # NEW: Pass settlement presets
        all_valid_factions=all_valid_factions, # NEW: Pass all valid factions
        screen_name_to_key_map=screen_name_to_key_map, # NEW: Pass screen name to key map
        faction_to_subculture_map=faction_to_subculture_map # NEW: Pass faction to subculture map
    )
