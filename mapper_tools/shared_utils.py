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

    def write(self, message: str):
        """
        Writes a message to the terminal and the log file, prepending a timestamp to non-empty lines.
        """
        with self.lock:
            # Don't add a timestamp to empty lines or just newlines
            if message.strip():
                # Get current time with milliseconds
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                # Prepend timestamp to the message
                message_with_timestamp = f"{timestamp} | {message}"
            else:
                message_with_timestamp = message

            self.log.write(message_with_timestamp)
            self.terminal.write(message_with_timestamp)
            self.flush()

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
    print(f"Faction Fixer run started. All subsequent output will be logged to '{log_filename}'.")


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
        return None, 0

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
        return best_match, highest_ratio
    return None, highest_ratio

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
        lxml_doc = etree.fromstring