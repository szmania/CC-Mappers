import os
import xml.etree.ElementTree as ET
import Levenshtein
import re
import csv
import json # Added for parse_llm_json_response
from collections import defaultdict
import sys
from PIL import Image # For coastal province detection
from lxml import etree # For XML schema validation
import hashlib # Added for calculate_sha256
import random # Added for _generate_nearby_coords

# --- Constants ---
# Prefixes to remove for cleaner fuzzy matching
COMMON_PREFIXES = ['ck3_', 'att_', 'wonder_', 'building_', 'terrain_', 'subculture_']

# --- Helper Functions ---

def find_files(directory, filename):
    """
    Recursively searches for a file with the given filename within a directory.
    Returns a list of full paths to all matching files.
    """
    found_files = []
    for root, _, files in os.walk(directory):
        if filename in files:
            found_files.append(os.path.join(root, filename))
    return found_files

def calculate_sha256(file_path):
    """
    Calculates the SHA-256 hash of a file.
    Reads the file in chunks to handle large files efficiently.
    Returns the hexadecimal digest of the hash, or None if the file cannot be read.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"Error reading file '{file_path}' for SHA-256 calculation: {e}")
        return None

def _clean_name_for_fuzzy_match(name):
    """Cleans a name for fuzzy matching by lowercasing, replacing underscores, and removing common prefixes."""
    cleaned_name = name.lower().replace('_', ' ')
    for prefix in COMMON_PREFIXES:
        if cleaned_name.startswith(prefix.replace('_', ' ')):
            cleaned_name = cleaned_name[len(prefix.replace('_', ' ')):].strip()
    return cleaned_name

def find_best_fuzzy_match(target_name, candidates, threshold=0.8):
    """
    Finds the best fuzzy match for a target name within a list of candidates.
    Returns the best matching candidate if its Levenshtein ratio is above the threshold,
    otherwise returns None.
    """
    if not target_name or not candidates:
        return None

    cleaned_target = _clean_name_for_fuzzy_match(target_name)
    best_match = None
    highest_ratio = 0

    for candidate in candidates:
        cleaned_candidate = _clean_name_for_fuzzy_match(candidate)
        ratio = Levenshtein.ratio(cleaned_target, cleaned_candidate)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = candidate

    if best_match and highest_ratio >= threshold:
        return best_match
    return None

def find_best_fuzzy_match_in_dict(target_name, candidate_dict, threshold=0.8):
    """
    Finds the best fuzzy match for a target name within the keys of a dictionary.
    Returns (best_matching_key, value_of_best_matching_key) if ratio > threshold,
    otherwise returns (None, None).
    """
    if not target_name or not candidate_dict:
        return None, None

    cleaned_target = _clean_name_for_fuzzy_match(target_name)
    best_match_key = None
    highest_ratio = 0

    for key in candidate_dict.keys():
        cleaned_key = _clean_name_for_fuzzy_match(key)
        ratio = Levenshtein.ratio(cleaned_target, cleaned_key)
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match_key = key

    if best_match_key and highest_ratio >= threshold:
        return best_match_key, candidate_dict[best_match_key]
    return None, None

def _generate_keywords(name):
    """Generates a set of keywords from a cleaned name."""
    return set(name.split())

def _normalize_name_for_match(name, prefixes_to_remove=None):
    """Normalizes a name for matching by lowercasing, replacing underscores, and removing specified prefixes."""
    normalized = name.lower().replace('_', ' ')
    if prefixes_to_remove:
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix.replace('_', ' ')):
                normalized = normalized[len(prefix.replace('_', ' ')):].strip()
    return normalized

def parse_tier(tier_str):
    """Parses a tier string (e.g., 'tier_1', '1') into an integer, or returns None."""
    if tier_str is None:
        return None
    if isinstance(tier_str, int):
        return tier_str
    match = re.match(r'(?:tier_)?(\d+)', str(tier_str), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def indent_xml(elem, level=0, indent_str='  '):
    """Adds pretty indentation to XML elements."""
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent_xml(subelem, level + 1, indent_str)
        if not subelem.tail or not subelem.tail.strip():
            subelem.tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

def validate_xml_with_schema(xml_root, schema_path):
    """
    Validates an XML ElementTree root against an XSD schema.
    Returns (True, None) if valid, or (False, error_message) if invalid.
    """
    if not os.path.exists(schema_path):
        return False, f"Schema file not found at: {schema_path}"

    try:
        schema_doc = etree.parse(schema_path)
        schema = etree.XMLSchema(schema_doc)

        # Convert standard ElementTree object to lxml Element object for validation
        xml_string = ET.tostring(xml_root, encoding='utf-8')
        lxml_doc = etree.fromstring(xml_string)

        schema.assertValid(lxml_doc)
        return True, None
    except etree.DocumentInvalid as e:
        return False, str(e)
    except etree.XMLSyntaxError as e:
        return False, f"XML Syntax Error in schema or document: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred during XML validation: {e}"

def prompt_to_create_xml(file_path, root_tag_name):
    """
    Prompts the user to create an XML file if it doesn't exist.
    Returns True if the file exists or was created, False if not.
    """
    if not os.path.exists(file_path):
        print(f"The file '{file_path}' does not exist.")
        response = input(f"Would you like to create it with a basic <{root_tag_name}> root element? (y/n): ").lower()
        if response == 'y':
            root = ET.Element(root_tag_name)
            tree = ET.ElementTree(root)
            try:
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
                print(f"'{file_path}' created successfully.")
                return True
            except IOError as e:
                print(f"Error creating file: {e}")
                return False
        else:
            print(f"Cannot proceed without '{file_path}'.")
            return False
    return True

# --- CK3 Data Loading ---

def get_ck3_building_keys(ck3_buildings_dir):
    """
    Scans CK3 building definition files and extracts unique building keys.
    """
    building_keys = set()
    if not os.path.exists(ck3_buildings_dir):
        print(f"Warning: CK3 buildings directory not found at '{ck3_buildings_dir}'. Skipping building key extraction.")
        return building_keys

    for root, _, files in os.walk(ck3_buildings_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Regex to find building keys (e.g., "building_castle_barracks = {")
                        # Also captures wonder keys (e.g., "wonder_pyramids_giza = {")
                        matches = re.findall(r'^\s*([a-zA-Z0-9_]+)\s*=\s*{', content, re.MULTILINE)
                        for match in matches:
                            building_keys.add(match)
                except Exception as e:
                    print(f"Error reading CK3 building file '{file_path}': {e}")
    return building_keys

def get_ck3_terrain_types(ck3_terrain_types_dir):
    """
    Scans CK3 terrain type definition files and extracts unique terrain type keys.
    """
    terrain_types = set()
    if not os.path.exists(ck3_terrain_types_dir):
        print(f"Warning: CK3 terrain types directory not found at '{ck3_terrain_types_dir}'. Skipping terrain type extraction.")
        return terrain_types

    for root, _, files in os.walk(ck3_terrain_types_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Regex to find terrain type keys (e.g., "plains = {", "desert_mountains = {")
                        matches = re.findall(r'^\s*([a-zA-Z0-9_]+)\s*=\s*{', content, re.MULTILINE)
                        for match in matches:
                            terrain_types.add(match)
                except Exception as e:
                    print(f"Error reading CK3 terrain type file '{file_path}': {e}")
    return terrain_types

def get_ck3_adjacencies(ck3_map_data_dir):
    """
    Parses adjacencies.csv to extract CK3 adjacency data.
    """
    adjacencies = []
    adjacencies_file = os.path.join(ck3_map_data_dir, 'adjacencies.csv')
    if not os.path.exists(adjacencies_file):
        print(f"Warning: adjacencies.csv not found at '{adjacencies_file}'. Skipping adjacency data extraction.")
        return adjacencies

    try:
        with open(adjacencies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Only consider valid adjacencies with 'from' and 'to' provinces
                if row.get('from') and row.get('to') and row.get('type'):
                    adjacencies.append({
                        'from': row['from'],
                        'to': row['to'],
                        'type': row['type'],
                        'name': row.get('name', f"adjacency_{row['from']}_{row['to']}") # Use a default name if missing
                    })
    except Exception as e:
        print(f"Error reading adjacencies.csv: {e}")
    return adjacencies

# --- Attila Data Loading ---

def get_map_index(attila_playable_areas_dir, attila_map_name):
    """
    Finds the map index and canonical map name for a given Attila campaign map name from
    the_campaign_map_playable_areas_tables. Supports both XML and TSV files.
    Returns (map_index, canonical_map_name) or (None, None) if not found.
    """
    if not os.path.exists(attila_playable_areas_dir):
        print(f"Error: Attila playable areas directory not found at '{attila_playable_areas_dir}'.")
        return None, None

    for root, _, files in os.walk(attila_playable_areas_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith('.xml'):
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//campaign_map_playable_areas_tables_entry'):
                        map_name_xml = entry.findtext('map_name')
                        if map_name_xml == attila_map_name:
                            return (entry.findtext('map_index'), map_name_xml)
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila playable areas XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila playable areas XML file '{file_path}': {e}")

            elif file.endswith('.tsv'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Skip first two metadata lines
                        f.readline()
                        f.readline()
                        reader = csv.reader(f, delimiter='\t')
                        header = next(reader)

                        # Find column indices
                        try:
                            index_col = header.index('index')
                            mapname_col = header.index('mapname')
                            meaningful_id_col = header.index('meaningful_id')
                            terrain_folder_col = header.index('terrain_folder')
                        except ValueError as e:
                            print(f"Warning: Missing expected column in TSV file '{file_path}': {e}. Skipping.")
                            continue

                        for row in reader:
                            if len(row) > max(index_col, mapname_col, meaningful_id_col, terrain_folder_col):
                                # Check for match against multiple columns
                                if (row[mapname_col] == attila_map_name or
                                    row[meaningful_id_col] == attila_map_name or
                                    row[terrain_folder_col] == attila_map_name):
                                    return (row[index_col], row[mapname_col]) # Return canonical mapname
                except Exception as e:
                    print(f"Error reading Attila playable areas TSV file '{file_path}': {e}")

    return None, None

def get_attila_preset_coords(attila_presets_dir, map_index):
    """
    Scans Attila battle preset files and extracts preset keys and their coordinates
    for a specific map_index.
    """
    preset_coords = {}
    if not os.path.exists(attila_presets_dir):
        print(f"Warning: Attila presets directory not found at '{attila_presets_dir}'. Skipping preset coordinate extraction.")
        return preset_coords

    for root, _, files in os.walk(attila_presets_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//campaign_battle_presets_tables_entry'):
                        if entry.findtext('map_index') == map_index:
                            key = entry.findtext('preset_key')
                            x = entry.findtext('x')
                            y = entry.findtext('y')
                            if key and x is not None and y is not None:
                                preset_coords[key] = {'x': x, 'y': y}
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila preset XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
    return preset_coords

def get_attila_settlement_presets(attila_presets_dir, map_index):
    """
    Scans Attila battle preset files and extracts settlement preset keys, coordinates,
    battle types, and unique status for a specific map_index.
    """
    settlement_presets = []
    if not os.path.exists(attila_presets_dir):
        print(f"Warning: Attila presets directory not found at '{attila_presets_dir}'. Skipping settlement preset extraction.")
        return settlement_presets

    for root, _, files in os.walk(attila_presets_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//campaign_battle_presets_tables_entry'):
                        if entry.findtext('map_index') == map_index:
                            preset_key = entry.findtext('preset_key')
                            battle_type = entry.findtext('battle_type')
                            is_unique_settlement = entry.findtext('is_unique_settlement')
                            x = entry.findtext('x')
                            y = entry.findtext('y')

                            if preset_key and battle_type and is_unique_settlement is not None and x is not None and y is not None:
                                # Only include settlement-related battle types
                                if battle_type.startswith('settlement_'):
                                    settlement_presets.append({
                                        'key': preset_key,
                                        'battle_type': battle_type,
                                        'is_unique_settlement': is_unique_settlement,
                                        'x': x,
                                        'y': y
                                    })
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila preset XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
    return settlement_presets

def get_attila_land_bridge_presets(attila_presets_dir, map_index):
    """
    Scans Attila battle preset files and extracts 'land_bridge' preset keys and coordinates
    for a specific map_index.
    """
    land_bridge_presets = []
    if not os.path.exists(attila_presets_dir):
        print(f"Warning: Attila presets directory not found at '{attila_presets_dir}'. Skipping land bridge preset extraction.")
        return land_bridge_presets

    for root, _, files in os.walk(attila_presets_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//campaign_battle_presets_tables_entry'):
                        if entry.findtext('map_index') == map_index:
                            preset_key = entry.findtext('preset_key')
                            battle_type = entry.findtext('battle_type')
                            x = entry.findtext('x')
                            y = entry.findtext('y')

                            if preset_key and battle_type == 'land_bridge' and x is not None and y is not None:
                                land_bridge_presets.append({
                                    'key': preset_key,
                                    'battle_type': battle_type,
                                    'x': x,
                                    'y': y
                                })
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila preset XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
    return land_bridge_presets

def get_attila_coastal_battle_presets(attila_presets_dir, map_index):
    """
    Scans Attila battle preset files and extracts 'coastal_battle' preset keys and coordinates
    for a specific map_index.
    """
    coastal_battle_presets = []
    if not os.path.exists(attila_presets_dir):
        print(f"Warning: Attila presets directory not found at '{attila_presets_dir}'. Skipping coastal battle preset extraction.")
        return coastal_battle_presets

    for root, _, files in os.walk(attila_presets_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//campaign_battle_presets_tables_entry'):
                        if entry.findtext('map_index') == map_index:
                            preset_key = entry.findtext('preset_key')
                            battle_type = entry.findtext('battle_type')
                            x = entry.findtext('x')
                            y = entry.findtext('y')

                            if preset_key and battle_type == 'coastal_battle' and x is not None and y is not None:
                                coastal_battle_presets.append({
                                    'key': preset_key,
                                    'battle_type': battle_type,
                                    'x': x,
                                    'y': y
                                })
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila preset XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
    return coastal_battle_presets

def get_siege_engines_data(attila_siege_engines_dir):
    """
    Scans Attila battlefield_deployable_siege_items_tables and extracts siege engine data.
    Returns a set of tuples: (key, type, siege_effort_cost).
    """
    siege_engines_data = set()
    if not os.path.exists(attila_siege_engines_dir):
        print(f"Warning: Attila siege engines directory not found at '{attila_siege_engines_dir}'. Skipping siege engine data extraction.")
        return siege_engines_data

    for root, _, files in os.walk(attila_siege_engines_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//battlefield_deployable_siege_items_tables_entry'):
                        key = entry.findtext('key')
                        type_val = entry.findtext('type')
                        cost = entry.findtext('siege_effort_cost')
                        if key and type_val and cost is not None:
                            siege_engines_data.add((key, type_val, cost))
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila siege engines XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila siege engines XML file '{file_path}': {e}")
    return siege_engines_data

def get_faction_key_to_screen_name_map(factions_tables_dir):
    """
    Scans Attila factions_tables and extracts faction keys and their screen names.
    """
    faction_map = {}
    if not os.path.exists(factions_tables_dir):
        print(f"Warning: Attila factions tables directory not found at '{factions_tables_dir}'. Skipping faction data extraction.")
        return faction_map

    for root, _, files in os.walk(factions_tables_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//factions_tables_entry'):
                        key = entry.findtext('faction_key')
                        screen_name = entry.findtext('screen_name')
                        if key and screen_name:
                            faction_map[key] = screen_name
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila factions XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila factions XML file '{file_path}': {e}")
    return faction_map

def get_faction_subculture_maps(factions_tables_dir):
    """
    Scans Attila factions_tables and extracts faction keys and their subcultures.
    Returns (faction_key_to_subculture_map, subculture_to_factions_map).
    """
    faction_key_to_subculture_map = {}
    subculture_to_factions_map = defaultdict(list)

    if not os.path.exists(factions_tables_dir):
        print(f"Warning: Attila factions tables directory not found at '{factions_tables_dir}'. Skipping subculture data extraction.")
        return faction_key_to_subculture_map, subculture_to_factions_map

    for root, _, files in os.walk(factions_tables_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//factions_tables_entry'):
                        key = entry.findtext('faction_key')
                        subculture = entry.findtext('subculture')
                        if key and subculture:
                            faction_key_to_subculture_map[key] = subculture
                            subculture_to_factions_map[subculture].append(key)
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila factions XML file '{file_path}': {e}")
                except Exception as e:
                    print(f"Error reading Attila factions XML file '{file_path}': {e}")
    return faction_key_to_subculture_map, subculture_to_factions_map

def parse_factions_xml_for_faction_names(factions_xml_path):
    """
    Parses the Factions.xml file to extract all faction names.
    """
    faction_names = set()
    if not os.path.exists(factions_xml_path):
        print(f"Warning: Factions XML file not found at '{factions_xml_path}'. Cannot extract faction names.")
        return faction_names

    try:
        tree = ET.parse(factions_xml_path)
        root = tree.getroot()
        for faction_element in root.findall('Faction'):
            name = faction_element.get('name')
            if name and name != "Default":
                faction_names.add(name)
    except ET.ParseError as e:
        print(f"Error parsing Factions XML file '{factions_xml_path}': {e}")
    return faction_names

def get_factions_from_cultures_xml(cultures_xml_path):
    """
    Parses Cultures.xml to get a set of all faction names defined within it.
    """
    factions = set()
    if not os.path.exists(cultures_xml_path):
        print(f"Warning: Cultures XML file not found at '{cultures_xml_path}'. Cannot extract faction names.")
        return factions

    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for culture_element in root.findall('.//culture'):
            for faction_element in culture_element.findall('faction'):
                faction_name = faction_element.get('name')
                if faction_name:
                    factions.add(faction_name)
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file '{cultures_xml_path}': {e}")
    return factions

def get_faction_to_culture_list_map_from_xml(cultures_xml_path):
    """
    Parses Cultures.xml to create a map from faction name to a list of cultures it belongs to.
    """
    faction_to_culture_map = defaultdict(list)
    if not os.path.exists(cultures_xml_path):
        print(f"Warning: Cultures XML file not found at '{cultures_xml_path}'. Cannot extract faction-culture map.")
        return faction_to_culture_map

    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for culture_element in root.findall('.//culture'):
            culture_name = culture_element.get('name')
            if culture_name:
                for faction_element in culture_element.findall('faction'):
                    faction_name = faction_element.get('name')
                    if faction_name:
                        faction_to_culture_map[faction_name].append(culture_name)
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file '{cultures_xml_path}': {e}")
    return faction_to_culture_map

def _generate_nearby_coords(center_x, center_y, count=50, radius=10):
    """
    Generates a list of `count` unique (x, y) coordinates clustered around a center point.
    Ensures coordinates are within reasonable bounds (0-2000).
    """
    coords = set()
    center_x_int = int(float(center_x))
    center_y_int = int(float(center_y))

    # Add the center itself
    coords.add((center_x_int, center_y_int))

    while len(coords) < count:
        # Generate random offsets within the radius
        offset_x = random.randint(-radius, radius)
        offset_y = random.randint(-radius, radius)

        new_x = max(0, min(2000, center_x_int + offset_x))
        new_y = max(0, min(2000, center_y_int + offset_y))

        coords.add((new_x, new_y))

        # Gradually increase radius if struggling to find enough unique points
        if len(coords) < count / 2 and len(coords) % 10 == 0:
            radius += 1

    return list(coords)

# --- Coastal Province Detection ---

def parse_default_map_for_sea_zones(ck3_map_data_dir):
    """
    Parses default.map to find all province IDs that are considered sea zones.
    """
    sea_zone_ids = set()
    default_map_path = os.path.join(ck3_map_data_dir, 'default.map')
    if not os.path.exists(default_map_path):
        print(f"Warning: default.map not found at '{default_map_path}'. Cannot identify sea zones.")
        return sea_zone_ids

    try:
        with open(default_map_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find the block for 'sea_zones'
            sea_zones_match = re.search(r'sea_zones\s*=\s*{([^}]+)}', content, re.DOTALL)
            if sea_zones_match:
                # Extract all numbers within the sea_zones block
                ids = re.findall(r'\b(\d+)\b', sea_zones_match.group(1))
                sea_zone_ids.update(ids)
    except Exception as e:
        print(f"Error reading default.map: {e}")
    return sea_zone_ids

def parse_definition_csv(ck3_map_data_dir):
    """
    Parses definition.csv to get province data (ID, R, G, B, Name).
    Returns (id_to_data_map, color_to_id_map).
    """
    id_to_data = {}
    color_to_id = {}
    definition_csv_path = os.path.join(ck3_map_data_dir, 'definition.csv')
    if not os.path.exists(definition_csv_path):
        print(f"Warning: definition.csv not found at '{definition_csv_path}'. Cannot parse province definitions.")
        return id_to_data, color_to_id

    try:
        with open(definition_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) >= 5:
                    prov_id = row[0]
                    r, g, b = row[1], row[2], row[3]
                    name = row[4]
                    id_to_data[prov_id] = {'r': int(r), 'g': int(g), 'b': int(b), 'name': name}
                    color_to_id[(int(r), int(g), int(b))] = prov_id
    except Exception as e:
        print(f"Error reading definition.csv: {e}")
    return id_to_data, color_to_id

def find_coastal_provinces_from_image(ck3_map_data_dir, sea_zone_ids, id_to_data, color_to_id):
    """
    Analyzes provinces.png to find land provinces that border sea zones.
    Returns a dictionary of {province_id: province_name} for coastal land provinces.
    """
    coastal_provinces = {}
    provinces_image_path = os.path.join(ck3_map_data_dir, 'provinces.png')
    if not os.path.exists(provinces_image_path):
        print(f"Warning: provinces.png not found at '{provinces_image_path}'. Cannot detect coastal provinces.")
        return coastal_provinces

    if not sea_zone_ids or not id_to_data or not color_to_id:
        print("Warning: Missing sea zone IDs or province definitions. Cannot detect coastal provinces.")
        return coastal_provinces

    try:
        img = Image.open(provinces_image_path)
        width, height = img.size
        pixels = img.load()

        # Create a set of RGB tuples for all sea zones
        sea_zone_colors = set()
        for prov_id in sea_zone_ids:
            if prov_id in id_to_data:
                data = id_to_data[prov_id]
                sea_zone_colors.add((data['r'], data['g'], data['b']))

        # Iterate through each pixel to find land provinces bordering sea
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x]
                current_color = (r, g, b)

                # If it's a land province (not a sea zone itself)
                if current_color in color_to_id and color_to_id[current_color] not in sea_zone_ids:
                    current_prov_id = color_to_id[current_color]

                    # Check 8-directional neighbors
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue # Skip self

                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                neighbor_color = pixels[nx, ny]
                                if neighbor_color in sea_zone_colors:
                                    coastal_provinces[current_prov_id] = id_to_data[current_prov_id]['name']
                                    break # Found a sea neighbor, move to next pixel
                        if current_prov_id in coastal_provinces:
                            break # Move to next pixel if already identified as coastal
    except Exception as e:
        print(f"Error processing provinces.png: {e}")
    return coastal_provinces

def parse_llm_json_response(response_content, expected_key, request_id, validation_pool=None):
    """
    Parses a JSON response from an LLM, validates it, and returns the value of the expected key.
    Returns None if parsing or validation fails.
    """
    if not response_content:
        print(f"  -> WARNING: LLM call returned no content for request {request_id}.")
        return None

    try:
        json_str_match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
        if json_str_match:
            response_data = json.loads(json_str_match.group(1))
            chosen_value = response_data.get(expected_key)

            if not chosen_value:
                print(f"  -> WARNING: LLM response for request {request_id} was missing the key '{expected_key}' or its value was null.")
                return None

            if validation_pool and chosen_value not in validation_pool:
                print(f"  -> WARNING: LLM chose value '{chosen_value}' which is not in the validation pool for request {request_id}. Discarding.")
                return None

            return chosen_value
        else:
            print(f"  -> WARNING: Could not find JSON block in LLM response for request {request_id}.")
            return None
    except json.JSONDecodeError:
        print(f"  -> WARNING: Failed to decode JSON from LLM response for request {request_id}.")
        return None
