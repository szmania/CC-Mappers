import datetime
import os
import xml.etree.ElementTree as ET
import Levenshtein
import re
import csv
import json # Added for parse_llm_json_response
from collections import defaultdict, Counter
import sys
from PIL import Image # For coastal province detection
from lxml import etree # For XML schema validation
import hashlib # Added for calculate_sha256
import random # Added for _generate_nearby_coords
import threading

# --- Constants ---
# Prefixes to remove for cleaner fuzzy matching
COMMON_PREFIXES = ['ck3_', 'att_', 'wonder_', 'building_', 'terrain_', 'subculture_']

# --- NEW: Cache for fuzzy matching ---
_fuzzy_match_clean_cache = {}

try:
    from lxml import etree
except ImportError:
    print("Error: The 'lxml' library is required for XML schema validation.")
    print("Please install it by running: pip install lxml")
    raise

try:
    from PIL import Image
except ImportError:
    print("Warning: The 'Pillow' library is required for coastal province detection.")
    print("Please install it by running: pip install Pillow")
    Image = None

# --- Logging Utilities ---

class Tee:
    """A class to redirect stdout to both console and a log file, prefixing console output."""
    def __init__(self, log_filename):
        self.terminal = sys.stdout
        self.log_file_path = log_filename
        self.log = open(log_filename, "w", encoding='utf-8')
        self.line_started = False
        self.lock = threading.Lock() # Add a lock for thread safety

    def write(self, message):
        with self.lock: # Acquire the lock before writing
            # Write the original, unmodified message to the log file
            self.log.write(message)

            # Process and write to the terminal, handling partial lines correctly
            # to ensure the prefix is only added at the start of a new line.
            # This is crucial for clean traceback formatting.
            for char in message:
                if not self.line_started:
                    # Only add prefix if the message part contains non-whitespace
                    if char.strip():
                        self.terminal.write(f"LOGFILE: {self.log_file_path} | ")
                        self.line_started = True
                self.terminal.write(char)
                if char == '\n':
                    self.line_started = False

            self.flush() # Flush while holding the lock

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        # Delegate other attribute calls to the original stdout to maintain its behavior
        return getattr(self.terminal, attr)

def setup_logging():
    """Initializes the Tee logging system."""
    log_dir = "mapper_tools/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid()
    log_filename = os.path.join(log_dir, f"faction_fixer_run_{timestamp}_pid{pid}.log")

    # Redirect stdout and stderr to our Tee object
    tee = Tee(log_filename)
    sys.stdout = tee
    sys.stderr = tee

    # This first print will now automatically include the log file path
    print(f"Faction Fixer run started. All subsequent output will be logged.")


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
    if name in _fuzzy_match_clean_cache:
        return _fuzzy_match_clean_cache[name]
    cleaned_name = name.lower().replace('_', ' ')
    for prefix in COMMON_PREFIXES:
        if cleaned_name.startswith(prefix.replace('_', ' ')):
            cleaned_name = cleaned_name[len(prefix.replace('_', ' ')):].strip()
    _fuzzy_match_clean_cache[name] = cleaned_name
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


def detect_factions_schema(filename: str, xml_root):
    """
    Determines the appropriate factions schema based on the XML root element's attributes.

    Args:
        xml_root: The root element of the parsed Factions XML.

    Returns:
        str: The path to the appropriate schema file.
             Returns 'schemas/factions.xsd' for main mods (identified by submod_tag="DahanChina").
             Returns 'schemas/factions_addons.xsd' for addon/submod files or if submod_tag is not "DahanChina".
    """
    # Check if the submod_tag attribute of the root element is "DahanChina"
    # This identifies the file as a "main" or "base" mod file requiring strict validation
    if filename.startswith('OfficialCC_') or (filename.startswith('Submod_') and not xml_root.get('submod_addon_tag')):
        return 'schemas/factions.xsd'
    else:
        # Otherwise, use the addon schema
        return 'schemas/factions_addons.xsd'

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

def _find_header(reader, required_columns):
    """
    Finds the header row in a TSV file, skipping metadata and comments.
    A row is considered a header if it contains all `required_columns`.
    Returns the header row as a list, or None if not found.
    The reader object is advanced past the header row.
    """
    for row in reader:
        if not row:
            continue  # Skip empty rows
        if row[0].strip().startswith('#'):
            continue  # Skip comment/metadata lines

        # Check if all required columns are in this row
        row_set = set(row)
        if all(col in row_set for col in required_columns):
            return row

    return None

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

def create_variant_to_base_map(unit_variant_map):
    """
    Creates a reverse map from a tiered variant unit key to its base unit key.
    Example: {'mk_nor_t1_candle_men': 'mk_nor_candle_men', 'mk_nor_t2_candle_men': 'mk_nor_candle_men'}
    """
    variant_to_base = {}
    for base_unit, variants in unit_variant_map.items():
        for tier_str, variant_unit_key in variants.items():
            variant_to_base[variant_unit_key] = base_unit
    return variant_to_base

def create_faction_to_heritages_map(heritage_to_factions_map):
    """
    Creates a reverse map from a faction name to a list of its heritages.
    """
    faction_to_heritages = defaultdict(list)
    if not heritage_to_factions_map:
        return faction_to_heritages

    for heritage, factions in heritage_to_factions_map.items():
        for faction in factions:
            if heritage not in faction_to_heritages[faction]:
                faction_to_heritages[faction].append(heritage)

    return dict(faction_to_heritages)

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
                    raise
    return building_keys

def get_ck3_terrain_types(ck3_terrain_types_dir):
    """
    Scans CK3 terrain type definition files and extracts unique terrain type keys.
    """
    terrain_types = set()
    if not os.path.exists(ck3_terrain_types_dir):
        raise FileNotFoundError(f"CK3 terrain types directory not found at '{ck3_terrain_types_dir}'.")

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
                    raise
    return terrain_types

def get_ck3_adjacencies(ck3_map_data_dir):
    """
    Parses adjacencies.csv to extract CK3 adjacency data.
    """
    adjacencies = []
    adjacencies_file = os.path.join(ck3_map_data_dir, 'adjacencies.csv')
    if not os.path.exists(adjacencies_file):
        raise FileNotFoundError(f"adjacencies.csv not found at '{adjacencies_file}'.")

    try:
        with open(adjacencies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Normalize row keys to lowercase for case-insensitive access
                row_lower = {k.lower().strip(): v for k, v in row.items() if k}

                # Only consider valid adjacencies with 'from' and 'to' provinces
                if row_lower.get('from') and row_lower.get('to') and row_lower.get('type'):
                    adjacencies.append({
                        'from': row_lower['from'],
                        'to': row_lower['to'],
                        'type': row_lower['type'],
                        'name': row_lower.get('name', f"adjacency_{row_lower['from']}_{row_lower['to']}") # Use a default name if missing
                    })
    except Exception as e:
        print(f"Error reading adjacencies.csv: {e}")
        raise
    return adjacencies

def get_ck3_maa_definitions(directory):
    """
    Parses CK3 Men-at-Arms .txt files to extract MAA definition names and their internal types.
    Example: {'qarlons_fallen': 'skirmishers', 'gondorian_footmen': 'heavy_infantry'}
    """
    maa_definitions = {}
    # This regex finds potential MAA definition starts: `maa_name = {`
    maa_start_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*=\s*\{', re.MULTILINE)
    type_pattern = re.compile(r'\s*type\s*=\s*([a-zA-Z0-9_]+)')

    if not os.path.isdir(directory):
        print(f"Error: CK3 Men-at-Arms directory not found: {directory}")
        return {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
                    content = f.read()
                    for match in maa_start_pattern.finditer(content):
                        maa_definition_name = match.group(1)
                        content_after_match = content[match.end():]

                        open_braces = 1
                        end_pos = -1
                        for i, char in enumerate(content_after_match):
                            if char == '{':
                                open_braces += 1
                            elif char == '}':
                                open_braces -= 1
                                if open_braces == 0:
                                    end_pos = i
                                    break

                        if end_pos != -1:
                            block_content = content_after_match[:end_pos]
                            type_match = type_pattern.search(block_content)
                            if type_match:
                                internal_type = type_match.group(1)
                                maa_definitions[maa_definition_name] = internal_type
            except Exception as e:
                print(f"Error reading or parsing {filename}: {e}")
                raise

    return maa_definitions

def _parse_culture_file_content(content):
    """
    Parses the string content of a CK3 culture file.
    This is a bit tricky due to the non-standard format. We'll use regex and brace counting.
    """
    cultures = {}
    # This regex finds potential culture definition starts: `culture_name = {`
    culture_start_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*=\s*\{', re.MULTILINE)
    heritage_pattern = re.compile(r'\s*heritage\s*=\s*([a-zA-Z0-9_]+)')

    for match in culture_start_pattern.finditer(content):
        culture_name = match.group(1)
        content_after_match = content[match.end():]

        open_braces = 1
        end_pos = -1
        for i, char in enumerate(content_after_match):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    end_pos = i
                    break

        if end_pos != -1:
            block_content = content_after_match[:end_pos]
            heritage_match = heritage_pattern.search(block_content)
            if heritage_match:
                heritage_name = heritage_match.group(1)
                cultures[culture_name] = heritage_name

    return cultures

def get_all_ck3_cultures(directory):
    """
    Scans a directory for .txt files and parses them to extract culture and heritage info.
    Returns a dict mapping culture_name to heritage_name, and another mapping filename to list of cultures.
    """
    all_cultures = {}
    file_to_cultures = defaultdict(dict)

    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return {}, {}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
                    content = f.read()
                    parsed_cultures = _parse_culture_file_content(content) # Call the moved helper
                    all_cultures.update(parsed_cultures)
                    file_to_cultures[filename] = parsed_cultures
            except Exception as e:
                print(f"Error reading or parsing {filename}: {e}")
                raise

    return all_cultures, file_to_cultures

def parse_creation_name_cultures(creation_names_dir):
    """
    Parses culture files from common/culture/creation_names/*.txt.
    These files can define both hybrid and non-hybrid cultures. A culture can have one or more heritages.
    Returns a dictionary mapping culture_name -> list of heritage_names.
    """
    creation_cultures = defaultdict(list)
    if not os.path.isdir(creation_names_dir):
        print(f"Info: CK3 creation_names directory not found at {creation_names_dir}. Skipping.")
        return creation_cultures

    print(f"Parsing creation name cultures from: {creation_names_dir}")
    heritage_pattern = re.compile(r"has_cultural_pillar\s*=\s*(\w+)")

    for filename in os.listdir(creation_names_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(creation_names_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()

                # Find top-level culture blocks: culture_name = { ... }
                # This is a simplified parser that uses brace counting to find block boundaries.
                i = 0
                while i < len(content):
                    match = re.search(r"^\s*(\w+)\s*=\s*\{", content[i:], re.MULTILINE)
                    if not match:
                        break

                    culture_name = match.group(1)
                    block_start = i + match.end()
                    brace_level = 1
                    block_end = -1

                    for j in range(block_start, len(content)):
                        if content[j] == '{':
                            brace_level += 1
                        elif content[j] == '}':
                            brace_level -= 1
                        if brace_level == 0:
                            block_end = j
                            break

                    if block_end != -1:
                        block_content = content[block_start:block_end]
                        # Find all cultural pillars and filter for only heritages
                        pillars = heritage_pattern.findall(block_content)
                        heritages = {pillar for pillar in pillars if pillar.startswith("heritage_")}
                        if heritages:
                            creation_cultures[culture_name].extend(list(heritages))
                        i = block_end + 1
                    else:
                        # Could not find matching brace, advance past the match to avoid infinite loop
                        i += match.start() + 1

            except Exception as e:
                print(f"Error parsing creation name culture file {file_path}: {e}")
                raise e

    # Deduplicate heritages for each culture
    for culture_name in creation_cultures:
        creation_cultures[culture_name] = sorted(list(set(creation_cultures[culture_name])))

    return creation_cultures

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
                    raise
                except Exception as e:
                    print(f"Error reading Attila playable areas XML file '{file_path}': {e}")
                    raise e

            elif file.endswith('.tsv'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        required_columns = ['index', 'mapname', 'meaningful_id', 'terrain_folder']
                        header = _find_header(reader, required_columns)
                        if not header:
                            print(f"Warning: Could not find a valid header in TSV file '{file_path}'. Skipping.")
                            continue

                        # Find column indices
                        try:
                            index_col = header.index('index')
                            mapname_col = header.index('mapname')
                            meaningful_id_col = header.index('meaningful_id')
                            terrain_folder_col = header.index('terrain_folder')
                        except ValueError as e:
                            print(f"Warning: Missing expected column in TSV file '{file_path}': {e}. Skipping.")
                            raise e

                        for row in reader:
                            if not row or row[0].strip().startswith('#'):
                                continue
                            if len(row) > max(index_col, mapname_col, meaningful_id_col, terrain_folder_col):
                                # Check for match against multiple columns
                                if (row[mapname_col] == attila_map_name or
                                    row[meaningful_id_col] == attila_map_name or
                                    row[terrain_folder_col] == attila_map_name):
                                    return (row[index_col], row[mapname_col]) # Return canonical mapname
                except Exception as e:
                    print(f"Error reading Attila playable areas TSV file '{file_path}': {e}")
                    raise

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
                    raise e
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
                    raise e
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
                    raise e
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
                    raise e
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
                    raise e
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
                    raise e
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
                    raise e
                except Exception as e:
                    print(f"Error reading Attila preset XML file '{file_path}': {e}")
                    raise e
    return coastal_battle_presets

def get_siege_engines_data(attila_siege_engines_dir):
    """
    Scans Attila battlefield_deployable_siege_items_tables and extracts siege engine data.
    Returns a set of tuples: (key, type, siege_effort_cost).
    """
    siege_engines_data = set()
    if not os.path.exists(attila_siege_engines_dir):
        raise FileNotFoundError(f"Attila siege engines directory not found at '{attila_siege_engines_dir}'.")

    for root, _, files in os.walk(attila_siege_engines_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.xml'):
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
                    raise e
                except Exception as e:
                    print(f"Error reading Attila siege engines XML file '{file_path}': {e}")
                    raise e
            elif file.endswith('.tsv'):
                try:
                    with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                        reader = csv.reader(f, delimiter='\t')
                        header = _find_header(reader, ["key", "type", "siege_effort_cost"])
                        if not header:
                            continue

                        try:
                            key_idx = header.index("key")
                            type_idx = header.index("type")
                            siege_effort_cost_idx = header.index("siege_effort_cost")
                        except ValueError:
                            continue

                        for row in reader:
                            if not row or row[0].strip().startswith('#'):
                                continue
                            if (len(row) > key_idx and len(row) > type_idx and
                                len(row) > siege_effort_cost_idx and row[key_idx]):
                                key = row[key_idx]
                                type_val = row[type_idx]
                                siege_effort_cost = row[siege_effort_cost_idx]
                                siege_engines_data.add((key, type_val, siege_effort_cost))
                except Exception as e:
                    print(f"Error processing siege engines TSV file {file}: {e}")
                    raise
    return siege_engines_data

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
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "screen_name"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        screen_name_idx = header.index("screen_name")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and len(row) > screen_name_idx and row[key_idx]:
                            faction_map[row[key_idx]] = row[screen_name_idx]
            except Exception as e:
                print(f"Error processing faction table TSV file {filename}: {e}")
    return faction_map

def get_faction_subculture_maps(factions_tables_dir):
    """
    Scans Attila factions_tables (XML and TSV) and extracts faction keys and their subcultures.
    Returns (faction_key_to_subculture_map, subculture_to_factions_map).
    """
    faction_key_to_subculture_map = {}
    subculture_to_factions_map = defaultdict(list)

    if not os.path.exists(factions_tables_dir):
        print(f"Warning: Attila factions tables directory not found at '{factions_tables_dir}'. Skipping subculture data extraction.")
        return faction_key_to_subculture_map, subculture_to_factions_map

    for root, _, files in os.walk(factions_tables_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.xml'):
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    for entry in root_elem.findall('.//factions_tables_entry'):
                        key = entry.findtext('faction_key')
                        subculture = entry.findtext('subculture')
                        if key and subculture:
                            faction_key_to_subculture_map[key] = subculture
                            if key not in subculture_to_factions_map[subculture]:
                                subculture_to_factions_map[subculture].append(key)
                except ET.ParseError as e:
                    print(f"Warning: Could not parse Attila factions XML file '{file_path}': {e}")
                    raise e
                except Exception as e:
                    print(f"Error reading Attila factions XML file '{file_path}': {e}")
                    raise e
            elif file.endswith('.tsv'):
                try:
                    with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                        reader = csv.reader(f, delimiter='\t')
                        header = _find_header(reader, ["key", "subculture"])
                        if not header:
                            continue

                        try:
                            key_idx = header.index("key")
                            subculture_idx = header.index("subculture")
                        except ValueError:
                            continue

                        for row in reader:
                            if not row or row[0].strip().startswith('#'):
                                continue
                            if len(row) > key_idx and len(row) > subculture_idx and row[key_idx]:
                                faction_key = row[key_idx]
                                subculture_key = row[subculture_idx]
                                if faction_key and subculture_key:
                                    faction_key_to_subculture_map[faction_key] = subculture_key
                                    if faction_key not in subculture_to_factions_map[subculture_key]:
                                        subculture_to_factions_map[subculture_key].append(faction_key)
                except Exception as e:
                    print(f"Error processing faction table TSV file {file}: {e}")
                    raise
    return faction_key_to_subculture_map, subculture_to_factions_map

def parse_factions_xml_for_faction_names(factions_xml_path):
    """
    Parses the Factions.xml file to extract all faction names.
    """
    faction_names = set()
    if not os.path.exists(factions_xml_path):
        raise FileNotFoundError(f"Factions XML file not found at '{factions_xml_path}'.")

    try:
        tree = ET.parse(factions_xml_path)
        root = tree.getroot()
        for element in root.findall('Faction'):
            faction_name = element.get('name')
            if faction_name:
                faction_names.add(faction_name)
    except ET.ParseError as e:
        print(f"Error parsing Factions XML file '{factions_xml_path}': {e}")
        raise
    return faction_names

def get_factions_from_cultures_xml(xml_file_path):
    """Parses the Cultures XML to get a set of all faction names used."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    factions = set()

    for element in root.findall('.//*[@faction]'):
        faction_name = element.get('faction')
        if faction_name:
            factions.add(faction_name)

    return factions

def get_faction_to_culture_list_map_from_xml(cultures_xml_path):
    """
    Parses Cultures.xml to create a map from faction name to a list of cultures it belongs to.
    """
    faction_to_culture_map = defaultdict(list)
    if not os.path.exists(cultures_xml_path):
        print(f"Warning: Cultures XML file not found at '{cultures_xml_path}'. Cannot extract faction-culture map.")
        return dict(faction_to_culture_map)

    try:
        tree = ET.parse(cultures_xml_path)
        root = tree.getroot()
        for culture_element in root.findall('.//Culture'):
            culture_name = culture_element.get('name')
            if culture_name:
                for faction_element in culture_element.findall('Faction'):
                    faction_name = faction_element.get('name')
                    if faction_name:
                        faction_to_culture_map[faction_name].append(culture_name)
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file '{cultures_xml_path}': {e}")
        raise
    return dict(faction_to_culture_map)

def get_culture_to_faction_map_from_xml(xml_file_path):
    """
    Parses Cultures.xml and creates a dictionary mapping each CK3 culture name to its assigned TWA faction name.
    Returns {culture_name: faction_name}.
    """
    culture_to_faction_map = {}
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        for heritage_element in root.findall('Heritage'):
            for culture_element in heritage_element.findall('Culture'):
                culture_name = culture_element.get('name')
                faction_name = culture_element.get('faction')
                if culture_name and faction_name:
                    culture_to_faction_map[culture_name.lower()] = faction_name
    except ET.ParseError as e:
        print(f"Error parsing Cultures XML file {xml_file_path}: {e}")
        raise

    return culture_to_faction_map

def get_culture_names_from_xml(xml_file):
    """
    Parses a Cultures XML file and returns a set of all culture names.
    """
    if not os.path.exists(xml_file):
        return set()

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        culture_names = {culture.get('name') for culture in root.findall('.//Culture') if culture.get('name')}
        return culture_names
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Warning: Could not parse main mod cultures XML file '{xml_file}': {e}. Proceeding without it.")
        raise
    return set()

def parse_cultures_xml(xml_file):
    """
    Parses the target Cultures XML file to get existing heritages and cultures.
    """
    # The file is guaranteed to exist by prompt_to_create_xml in main.
    # The try-except block below will handle parsing errors.

    tree = ET.parse(xml_file)
    root = tree.getroot()

    heritages = {} # heritage_name -> faction
    cultures = defaultdict(set) # culture_name -> set of heritage_names it belongs to

    for heritage_element in root.findall('Heritage'):
        heritage_name = heritage_element.get('name')
        faction_name = heritage_element.get('faction')
        if heritage_name and faction_name:
            heritages[heritage_name] = faction_name

        for culture_element in heritage_element.findall('Culture'):
            culture_name = culture_element.get('name')
            if culture_name and heritage_name:
                cultures[culture_name].add(heritage_name)

    return tree, heritages, cultures

def get_main_mod_faction_maa_map(xml_file_path):
    """
    Parses a Factions.xml file and creates a map of faction names to a set of their MenAtArm types.
    """
    faction_maa_map = defaultdict(set)
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for faction in root.findall('Faction'):
            faction_name = faction.get('name')
            if faction_name:
                for maa in faction.findall('MenAtArm'):
                    maa_type = maa.get('type')
                    if maa_type:
                        faction_maa_map[faction_name].add(maa_type)
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing main mod Factions XML file {xml_file_path}: {e}")
        raise
    return dict(faction_maa_map)

def get_most_common_faction_from_cultures(xml_file_path):
    """Parses the Cultures XML to find the most frequently assigned faction name."""
    faction_counts = Counter()
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for element in root.findall('.//*[@faction]'):
            faction_name = element.get('faction')
            if faction_name and faction_name != "Default": # Exclude "Default" from counting
                faction_counts[faction_name] += 1
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing Cultures XML for faction counting: {e}")
        return None

    if not faction_counts:
        return None

    most_common_faction_name = faction_counts.most_common(1)[0][0]
    return most_common_faction_name

def get_unit_to_faction_key_map(tsv_dir):
    """Parses the units_custom_battle_permissions TSV files to map unit keys to faction keys."""
    unit_faction_map = {}
    if not os.path.isdir(tsv_dir):
        print(f"Error: Custom battle permissions directory not found at {tsv_dir}")
        return unit_faction_map

    for filename in os.listdir(tsv_dir):
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["unit", "faction"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("unit")
                        faction_idx = header.index("faction")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and len(row) > faction_idx and row[unit_idx]:
                            unit_faction_map[row[unit_idx]] = row[faction_idx]
            except Exception as e:
                print(f"Error processing custom battle permissions TSV file {filename}: {e}")
                raise
    return unit_faction_map

def get_faction_key_to_units_map(tsv_dir):
    """Parses the units_custom_battle_permissions TSV files to map faction keys to a set of their unit keys."""
    faction_units_map = defaultdict(set)
    if not os.path.isdir(tsv_dir):
        print(f"Error: Custom battle permissions directory not found at {tsv_dir}")
        return faction_units_map

    for filename in os.listdir(tsv_dir):
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["unit", "faction"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("unit")
                        faction_idx = header.index("faction")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and len(row) > faction_idx and row[unit_idx] and row[faction_idx]:
                            faction_units_map[row[faction_idx]].add(row[unit_idx])
            except Exception as e:
                print(f"Error processing custom battle permissions TSV file {filename}: {e}")
                raise
    return faction_units_map

def get_all_land_units_keys(tsv_dir):
    """
    Reads all unit keys from the land_units_tables TSV files.
    This serves as the definitive source of truth for valid unit keys.
    """
    all_land_units_keys = set()

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Land units TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and row[key_idx]:
                            all_land_units_keys.add(row[key_idx])
            except Exception as e:
                print(f"Error processing land units TSV file {filename}: {e}")
                raise
    return all_land_units_keys

def get_unit_land_categories(tsv_dir):
    """Scans a directory for .tsv files and extracts the 'category' for each unit 'key'."""
    unit_categories = {}

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Land units TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "category"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        category_idx = header.index("category")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and len(row) > category_idx and row[key_idx]:
                            unit_categories[row[key_idx]] = row[category_idx]
            except Exception as e:
                print(f"Error processing land units TSV file {filename}: {e}")
                raise
    return unit_categories

def get_unit_classes(tsv_dir):
    """Scans a directory for .tsv files and extracts the 'class' for each unit 'key'."""
    unit_classes = {}

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Land units TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "class"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        class_idx = header.index("class")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and len(row) > class_idx and row[key_idx]:
                            unit_classes[row[key_idx]] = row[class_idx]
            except Exception as e:
                print(f"Error processing land units TSV file {filename}: {e}")
                raise
    return unit_classes

def get_unit_descriptions(tsv_dir):
    """Scans a directory for .tsv files and extracts the 'historical_description_text' for each unit 'key'."""
    unit_descriptions = {}

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Land units TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "historical_description_text"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        desc_idx = header.index("historical_description_text")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and len(row) > desc_idx and row[key_idx]:
                            unit_descriptions[row[key_idx]] = row[desc_idx]
            except Exception as e:
                print(f"Error processing land units TSV file {filename}: {e}")
                raise
    return unit_descriptions

def get_unit_screen_name_map(attila_text_path):
    """
    Recursively scans a directory for .tsv files and builds a map of
    Attila unit keys to their human-readable screen names.
    """
    screen_name_map = {}
    if not attila_text_path or not os.path.isdir(attila_text_path):
        print(f"Warning: Attila text path not provided or not found at '{attila_text_path}'. Cannot load unit screen names.")
        return screen_name_map

    print(f"Loading unit screen names from: {attila_text_path}")
    prefix = "land_units_onscreen_name_"
    prefix_len = len(prefix)

    for root_dir, _, files in os.walk(attila_text_path):
        for filename in files:
            if filename.endswith('.tsv'):
                file_path = os.path.join(root_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        # Use csv reader to handle potential quotes or special characters
                        reader = csv.reader(f, delimiter='\t')
                        for row in reader:
                            # Skip comments, headers, or malformed rows
                            if not row or len(row) < 2 or row[0].strip().startswith('#') or not row[0].startswith(prefix):
                                continue

                            loc_key = row[0]
                            screen_name = row[1]

                            unit_key = loc_key[prefix_len:]
                            screen_name_map[unit_key] = screen_name
                except Exception as e:
                    print(f"Error reading screen name file {file_path}: {e}")
                    raise

    print(f"Loaded {len(screen_name_map)} unit screen names.")
    return screen_name_map

def get_unit_num_guns(land_units_tsv_dir):
    """
    Reads all .tsv files in land_units_tables to get the number of guns for each unit.
    """
    num_guns_map = {}
    if not os.path.isdir(land_units_tsv_dir):
        print(f"Warning: land_units_tables directory not found at {land_units_tsv_dir}")
        return num_guns_map

    for filename in os.listdir(land_units_tsv_dir):
        if not filename.endswith('.tsv'):
            continue

        file_path = os.path.join(land_units_tsv_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f, delimiter='\t')
                header = _find_header(reader, ['key', 'num_guns'])
                if not header:
                    continue

                try:
                    key_idx = header.index('key')
                    num_guns_idx = header.index('num_guns')
                except ValueError:
                    # Silently skip files that don't have the required columns.
                    continue

                for row in reader:
                    if not row or row[0].strip().startswith('#'):
                        continue
                    if len(row) > max(key_idx, num_guns_idx):
                        unit_key = row[key_idx]
                        try:
                            num_guns = int(row[num_guns_idx])
                            num_guns_map[unit_key] = num_guns
                        except (ValueError, IndexError):
                            # Ignore if num_guns is not a valid integer or row is malformed
                            pass
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            raise

    return num_guns_map

def get_unit_training_levels(tsv_dir):
    """
    Parses land_units_tables TSV files to get the 'training_level' for each unit 'key'.
    Returns a dictionary mapping unit_key to its training_level string.
    """
    unit_training_levels = {}

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Land units TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "training_level"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        training_level_idx = header.index("training_level")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and len(row) > training_level_idx and row[key_idx]:
                            unit_training_levels[row[key_idx]] = row[training_level_idx]
            except Exception as e:
                print(f"Error processing land units TSV file {filename}: {e}")
                raise
    return unit_training_levels

def get_tsv_units(tsv_dir, valid_units_set):
    """
    Scans a directory for .tsv files and extracts all valid land units, categorized by role.
    Filters units against the provided valid_units_set.
    Returns a dictionary mapping role to a list of unit keys.
    """
    categorized_units = defaultdict(list)

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["land_unit", "ui_unit_group_land"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("land_unit")
                        role_idx = header.index("ui_unit_group_land")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and len(row) > role_idx and row[unit_idx]:
                            unit_key = row[unit_idx]
                            if unit_key in valid_units_set:
                                unit_role = row[role_idx]
                                if unit_role:
                                    categorized_units[unit_role].append(unit_key)
            except Exception as e:
                print(f"Error processing TSV file {filename}: {e}")
                raise
    return categorized_units

def get_unit_to_tier_map(tsv_dir, valid_units_set):
    """
    Scans a directory for .tsv files and extracts the 'tier' for each unit 'land_unit'.
    Filters units against the provided valid_units_set.
    Returns a dictionary mapping unit_key to its integer tier.
    """
    unit_to_tier_map = {}

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["land_unit", "tier"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("land_unit")
                        tier_idx = header.index("tier")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and len(row) > tier_idx and row[unit_idx]:
                            unit_key = row[unit_idx]
                            if unit_key in valid_units_set:
                                tier_str = row[tier_idx]
                                if tier_str.isdigit():
                                    unit_to_tier_map[unit_key] = int(tier_str)
            except Exception as e:
                print(f"Error processing TSV file {filename}: {e}")
                raise
    return unit_to_tier_map

def get_general_units(tsv_dir, valid_units_set):
    """
    Scans a directory for .tsv files and extracts all units marked as general units.
    Filters units against the provided valid_units_set.
    """
    general_units = set()

    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["unit", "general_unit"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("unit")
                        general_idx = header.index("general_unit")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and len(row) > general_idx and row[general_idx].lower() == 'true':
                            unit_key = row[unit_idx]
                            if unit_key and unit_key in valid_units_set:
                                general_units.add(unit_key)
            except Exception as e:
                print(f"Error processing TSV file {filename}: {e}")
                raise
    return general_units

def get_unit_variant_map(tsv_dir, valid_units_set, unit_to_tier_map):
    """
    Parses unit_variants_tables TSV files to map base unit keys to their tiered variants.
    Ensures both base_unit and variant_unit are in valid_units_set.
    Uses unit_to_tier_map to get the integer tier for the variant_unit.
    Returns a dictionary like: {'base_unit': {2: 'unit_key_t2', 3: 'unit_key_t3'}}
    """
    unit_variant_map = defaultdict(dict)
    try:
        file_list = os.listdir(tsv_dir)
    except FileNotFoundError:
        print(f"Error: Unit variants TSV directory not found at '{os.path.abspath(tsv_dir)}'")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while accessing '{tsv_dir}': {e}")
        raise

    for filename in file_list:
        if filename.endswith(".tsv"):
            file_path = os.path.join(tsv_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["base_unit", "variant_unit"])
                    if not header:
                        continue

                    try:
                        base_unit_idx = header.index("base_unit")
                        variant_unit_idx = header.index("variant_unit")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > base_unit_idx and len(row) > variant_unit_idx:
                            base_unit = row[base_unit_idx]
                            variant_unit = row[variant_unit_idx]
                            variant_tier_num = unit_to_tier_map.get(variant_unit)
                            if base_unit and variant_unit and variant_tier_num is not None and base_unit in valid_units_set and variant_unit in valid_units_set:
                                unit_variant_map[base_unit][variant_tier_num] = variant_unit
            except Exception as e:
                print(f"Error processing unit variants TSV file {filename}: {e}")
                raise
    return unit_variant_map

def get_unit_stats_map(main_units_dir, land_units_dir, all_land_units_keys):
    """
    Parses main_units and land_units tables to build a map of unit stats for quality calculation.
    """
    unit_stats = defaultdict(dict)

    # 1. Parse land_units_tables for core combat stats
    try:
        for filename in os.listdir(land_units_dir):
            if filename.endswith(".tsv"):
                file_path = os.path.join(land_units_dir, filename)
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "morale", "armour", "defence", "charge_bonus"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        morale_idx = header.index("morale")
                        armour_idx = header.index("armour")
                        defence_idx = header.index("defence")
                        charge_idx = header.index("charge_bonus")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > key_idx and row[key_idx] in all_land_units_keys:
                            unit_key = row[key_idx]
                            try:
                                unit_stats[unit_key]['morale'] = int(row[morale_idx]) if row[morale_idx] else 0
                                unit_stats[unit_key]['armour'] = int(row[armour_idx]) if row[armour_idx] else 0
                                unit_stats[unit_key]['defence'] = int(row[defence_idx]) if row[defence_idx] else 0
                                unit_stats[unit_key]['charge_bonus'] = int(row[charge_idx]) if row[charge_idx] else 0
                            except (ValueError, IndexError):
                                continue
    except Exception as e:
        print(f"Error processing land units TSV files for stats: {e}")
        raise

    # 2. Parse main_units_tables for tier and cost
    try:
        for filename in os.listdir(main_units_dir):
            if filename.endswith(".tsv"):
                file_path = os.path.join(main_units_dir, filename)
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["land_unit", "tier", "recruitment_cost"])
                    if not header:
                        continue

                    try:
                        unit_idx = header.index("land_unit")
                        tier_idx = header.index("tier")
                        cost_idx = header.index("recruitment_cost")
                    except ValueError:
                        continue

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > unit_idx and row[unit_idx] in all_land_units_keys:
                            unit_key = row[unit_idx]
                            try:
                                tier_str = row[tier_idx]
                                if tier_str.isdigit():
                                    unit_stats[unit_key]['tier'] = int(tier_str)
                                unit_stats[unit_key]['cost'] = int(row[cost_idx]) if row[cost_idx] else 0
                            except (ValueError, IndexError):
                                continue
    except Exception as e:
        print(f"Error processing main units TSV files for stats: {e}")
        raise

    return dict(unit_stats)

def get_map_indices(directory, terrain_folder_name):
    """
    Extracts all map indices for a given terrain folder name from campaign_map_playable_areas_tables.
    Returns a list of map indices.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Attila playable areas directory not found: {directory}")

    lower_terrain_folder_name = terrain_folder_name.lower()
    map_indices = []

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["terrain_folder", "index"])
                    if not header:
                        continue

                    try:
                        terrain_folder_idx = header.index("terrain_folder")
                        index_idx = header.index("index")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if len(row) > terrain_folder_idx and len(row) > index_idx and row[terrain_folder_idx]:
                            if row[terrain_folder_idx].lower() == lower_terrain_folder_name:
                                map_indices.append(row[index_idx]) # Add the index to our list
            except Exception as e:
                print(f"Error processing playable areas TSV file {filename}: {e}")
                raise
    return map_indices

def get_attila_preset_coords(directory, required_map_indices):
    """
    Extracts all Attila battle preset keys and their coordinates, filtered by a list of campaign map indices.
    """
    preset_coords = {}

    if not required_map_indices:
        print("Warning: No required_map_indices provided. Returning empty preset coordinates.")
        return {}

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Attila campaign_battle_presets_tables directory not found: {directory}")

    # Convert to set for faster lookups
    required_map_indices_set = set(required_map_indices)

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "coord_x", "coord_y", "campaign_map"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if (len(row) > key_idx and len(row) > coord_x_idx and
                            len(row) > coord_y_idx and len(row) > campaign_map_idx and
                            row[key_idx]):

                            # Filter by campaign_map indices
                            if row[campaign_map_idx] in required_map_indices_set:
                                preset_key = row[key_idx]
                                coord_x = row[coord_x_idx]
                                coord_y = row[coord_y_idx]
                                preset_coords[preset_key] = {'x': coord_x, 'y': coord_y}
            except Exception as e:
                print(f"Error processing preset TSV file {filename}: {e}")
                raise
    return preset_coords

def get_attila_settlement_presets(directory, required_map_indices):
    """
    Extracts Attila settlement battle presets, filtered by battle_type and a list of campaign_map indices.
    """
    settlement_presets = []

    if not required_map_indices:
        print("Warning: No required_map_indices provided. Returning empty settlement presets.")
        return []

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Attila campaign_battle_presets_tables directory not found: {directory}")

    # Convert to set for faster lookups
    required_map_indices_set = set(required_map_indices)

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "battle_type", "is_unique_settlement", "coord_x", "coord_y", "campaign_map"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        battle_type_idx = header.index("battle_type")
                        # tile_upgrade_idx = header.index("key") # DELETED: This column does not exist
                        is_unique_settlement_idx = header.index("is_unique_settlement")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if (len(row) > key_idx and len(row) > battle_type_idx and
                            len(row) > is_unique_settlement_idx and
                            len(row) > coord_x_idx and len(row) > coord_y_idx and
                            len(row) > campaign_map_idx and row[key_idx]):

                            # Filter by campaign_map indices and battle_type
                            if row[campaign_map_idx] in required_map_indices_set and \
                               (row[battle_type_idx] == 'settlement_standard' or row[battle_type_idx] == 'settlement_unfortified'):

                                preset_data = {
                                    'battle_type': row[battle_type_idx],
                                    'key': row[key_idx], # MODIFIED: Use key_idx for tile_upgrade
                                    'is_unique_settlement': row[is_unique_settlement_idx],
                                    'x': row[coord_x_idx],
                                    'y': row[coord_y_idx]
                                }
                                settlement_presets.append(preset_data)
            except Exception as e:
                print(f"Error processing settlement preset TSV file {filename}: {e}")
                raise
    return settlement_presets

def get_attila_land_bridge_presets(directory, required_map_indices):
    """
    Extracts Attila land bridge battle presets, filtered by battle_type and a list of campaign_map indices.
    """
    land_bridge_presets = []

    if not required_map_indices:
        print("Warning: No required_map_indices provided. Returning empty land bridge presets.")
        return []

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Attila campaign_battle_presets_tables directory not found: {directory}")

    # Convert to set for faster lookups
    required_map_indices_set = set(required_map_indices)

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "battle_type", "coord_x", "coord_y", "campaign_map"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        battle_type_idx = header.index("battle_type")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if (len(row) > key_idx and len(row) > battle_type_idx and
                            len(row) > coord_x_idx and len(row) > coord_y_idx and
                            len(row) > campaign_map_idx and row[key_idx]):

                            # Filter by campaign_map indices and battle_type
                            if row[campaign_map_idx] in required_map_indices_set and row[battle_type_idx] == 'land_bridge':
                                preset_data = {
                                    'key': row[key_idx],
                                    'x': row[coord_x_idx],
                                    'y': row[coord_y_idx]
                                }
                                land_bridge_presets.append(preset_data)
            except Exception as e:
                print(f"Error processing land bridge preset TSV file {filename}: {e}")
                raise
    return land_bridge_presets

def get_attila_coastal_battle_presets(directory, required_map_indices):
    """
    Extracts Attila coastal battle presets, filtered by battle_type and a list of campaign_map indices.
    """
    coastal_battle_presets = []

    if not required_map_indices:
        print("Warning: No required_map_indices provided. Returning empty coastal battle presets.")
        return []

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Attila campaign_battle_presets_tables directory not found: {directory}")

    # Convert to set for faster lookups
    required_map_indices_set = set(required_map_indices)

    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = _find_header(reader, ["key", "battle_type", "coord_x", "coord_y", "campaign_map"])
                    if not header:
                        continue

                    try:
                        key_idx = header.index("key")
                        battle_type_idx = header.index("battle_type")
                        coord_x_idx = header.index("coord_x")
                        coord_y_idx = header.index("coord_y")
                        campaign_map_idx = header.index("campaign_map")
                    except ValueError:
                        continue # Skip files without the required columns

                    for row in reader:
                        if not row or row[0].strip().startswith('#'):
                            continue
                        if (len(row) > key_idx and len(row) > battle_type_idx and
                            len(row) > coord_x_idx and len(row) > coord_y_idx and
                            len(row) > campaign_map_idx and row[key_idx]):

                            # Filter by campaign_map indices and battle_type
                            if row[campaign_map_idx] in required_map_indices_set and row[battle_type_idx] == 'coastal_battle':
                                coastal_battle_presets.append({'key': row[key_idx], 'x': row[coord_x_idx], 'y': row[coord_y_idx]})
            except Exception as e:
                print(f"Error processing coastal battle preset TSV file {filename}: {e}")
                raise
    return coastal_battle_presets

def get_faction_heritage_maps_from_xml(xml_file_path):
    """
    Parses Cultures.xml and creates two dictionaries:
    1. A map from faction screen name to its heritage name.
    2. A map from heritage name to a list of its associated faction screen names.
    Returns (faction_to_heritage_map, heritage_to_factions_map).
    """
    faction_to_heritage_map = {}
    heritage_to_factions_map = defaultdict(list)
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        for heritage_element in root.findall('Heritage'):
            heritage_name = heritage_element.get('name')
            if not heritage_name:
                continue

            for culture_element in heritage_element.findall('Culture'):
                faction_name = culture_element.get('faction')
                if faction_name:
                    if faction_name not in heritage_to_factions_map[heritage_name]:
                         heritage_to_factions_map[heritage_name].append(faction_name)
                    faction_to_heritage_map[faction_name] = heritage_name

    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing Cultures XML for heritage maps: {e}")
        raise

    return faction_to_heritage_map, dict(heritage_to_factions_map)

def parse_default_map_for_sea_zones(ck3_map_data_dir):
    """
    Parses default.map to find all province IDs that are considered sea zones.
    """
    sea_zone_ids = set()
    default_map_path = os.path.join(ck3_map_data_dir, 'default.map')
    if not os.path.exists(default_map_path):
        raise FileNotFoundError(f"default.map not found at '{default_map_path}'.")

    try:
        with open(default_map_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

            # Step 1: Parse RANGE definitions
            range_matches = re.findall(r'sea_zones\s*=\s*RANGE\s*{\s*(\d+)\s*(\d+)\s*}', content)
            for start, end in range_matches:
                start_id, end_id = int(start), int(end)
                sea_zone_ids.update(str(i) for i in range(start_id, end_id + 1))

            # Step 2: Parse list definitions
            list_matches = re.findall(r'sea_zones\s*=\s*{\s*([^}]+)\s*}', content)
            for match in list_matches:
                ids = re.findall(r'\b(\d+)\b', match)
                sea_zone_ids.update(ids)

            # Step 3: Parse single-number definitions
            single_matches = re.findall(r'sea_zones\s*=\s*(\d+)', content)
            sea_zone_ids.update(single_matches)

    except Exception as e:
        print(f"Error reading default.map: {e}")
        raise
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
        raise FileNotFoundError(f"definition.csv not found at '{definition_csv_path}'.")

    try:
        with open(definition_csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) >= 5:
                    prov_id = row[0]
                    # Ignore comments
                    if prov_id.strip().startswith('#'):
                        continue

                    try:
                        r, g, b = int(row[1]), int(row[2]), int(row[3])
                        name = row[4]
                        id_to_data[prov_id] = {'r': r, 'g': g, 'b': b, 'name': name}
                        color_to_id[(r, g, b)] = prov_id
                    except ValueError:
                        # This will catch cases where r, g, or b are empty strings or not valid integers (like a header row).
                        # Silently skip these malformed/header rows.
                        continue
    except Exception as e:
        print(f"Error reading definition.csv: {e}")
        raise
    return id_to_data, color_to_id

def find_coastal_provinces_from_image(ck3_map_data_dir, sea_zone_ids, id_to_data, color_to_id):
    """
    Analyzes provinces.png to find land provinces that border sea zones.
    Returns a dictionary of {province_id: province_name} for coastal land provinces.
    """
    coastal_provinces = {}
    provinces_image_path = os.path.join(ck3_map_data_dir, 'provinces.png')
    if not os.path.exists(provinces_image_path):
        raise FileNotFoundError(f"provinces.png not found at '{provinces_image_path}'.")

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
                current_color_tuple = pixels[x, y]
                current_color = current_color_tuple[:3] # Ensure RGB

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
                                neighbor_color_tuple = pixels[nx, ny]
                                neighbor_color = neighbor_color_tuple[:3] # Ensure RGB
                                if neighbor_color in sea_zone_colors:
                                    coastal_provinces[current_prov_id] = id_to_data[current_prov_id]['name']
                                    break # Found a sea neighbor, move to next pixel
                        if current_prov_id in coastal_provinces:
                            break # Move to next pixel if already identified as coastal
    except Exception as e:
        print(f"Error processing provinces.png: {e}")
        raise
    return coastal_provinces
