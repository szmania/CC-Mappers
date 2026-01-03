import argparse
import logging
import xml.etree.ElementTree as ET
from typing import Set, Optional
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Cache for LLM responses
llm_cache = {}

def load_cache(cache_file: str) -> dict:
    """Load the LLM response cache from a JSON file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Starting with empty cache.")
            return {}
    return {}

def save_cache(cache: dict, cache_file: str) -> None:
    """Save the LLM response cache to a JSON file."""
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def get_llm_response(prompt: str, cache: dict, cache_file: str) -> Optional[str]:
    """Get a response from the LLM, using cache when possible."""
    # Create a hash of the prompt for the cache key
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    
    # Check if we have a cached response
    if prompt_hash in cache:
        logger.debug(f"Using cached response for prompt hash: {prompt_hash}")
        return cache[prompt_hash]
    
    try:
        logger.debug("Making API call to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests valid game unit names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        suggestion = response.choices[0].message.content.strip()
        logger.debug(f"Received response from OpenAI: {suggestion}")
        
        # Cache the response
        cache[prompt_hash] = suggestion
        save_cache(cache, cache_file)
        
        return suggestion
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return None

def extract_unit_type(unit_name: str) -> str:
    """Extract the unit type from a unit name."""
    # Common unit types in the game
    unit_types = [
        'Spearman', 'Swordsman', 'Archer', 'Crossbowman', 'Horseman', 'Knight',
        'Pikeman', 'Halberdier', 'Militia', 'Warrior', 'Guard', 'Veteran',
        'Elite', 'Heavy', 'Light', 'Mounted', 'Bowman', 'Longbowman',
        'Catapult', 'Ballista', 'Artillery', 'Engineer'
    ]
    
    # First try to find exact matches
    for unit_type in unit_types:
        if unit_type.lower() in unit_name.lower():
            return unit_type
    
    # If no exact match, try partial matching
    parts = unit_name.split('_')
    for part in reversed(parts):  # Check from the end which often contains the unit type
        for unit_type in unit_types:
            if unit_type.lower() in part.lower():
                return unit_type
    
    # Default fallback
    return 'Unit'

def is_valid_unit(unit_name: str, valid_units: Set[str]) -> bool:
    """Check if a unit name is valid."""
    return unit_name in valid_units

def get_faction_from_unit(unit_name: str) -> str:
    """Extract faction name from unit name."""
    parts = unit_name.split('_')
    if len(parts) >= 2:
        return parts[0]
    return "Unknown"

def suggest_valid_unit(replaced_unit: str, faction: str, valid_units: Set[str], 
                      cache: dict, cache_file: str) -> Optional[str]:
    """Suggest a valid unit to replace an invalid one."""
    unit_type = extract_unit_type(replaced_unit)
    
    # Create a prompt for the LLM
    prompt = f"""
    Suggest a valid unit name for the faction "{faction}" that is similar to "{replaced_unit}" 
    and is of type "{unit_type}". The unit should fit the faction's theme and be in the format 
    "Faction_Subfaction_UnitType". Only respond with the unit name, nothing else.
    
    Valid units in the game include:
    {', '.join(list(valid_units)[:50])}...
    """
    
    suggestion = get_llm_response(prompt, cache, cache_file)
    
    if suggestion and is_valid_unit(suggestion, valid_units):
        return suggestion
    
    # Fallback: try to find a unit with the same type from the same faction
    faction_units = [unit for unit in valid_units if unit.startswith(f"{faction}_")]
    for unit in faction_units:
        if unit_type.lower() in unit.lower():
            return unit
    
    # Last resort: return any unit from the faction
    if faction_units:
        return faction_units[0]
    
    return None

def collect_all_valid_units(root) -> Set[str]:
    """Collect all valid unit names from the XML."""
    valid_units = set()
    
    # Find all unit definitions
    for unit in root.iter('unit'):
        name_elem = unit.find('name')
        if name_elem is not None and name_elem.text:
            valid_units.add(name_elem.text)
    
    # Also collect units from recruitables
    for rec in root.iter('recruitables'):
        for child in rec:
            if child.text:
                valid_units.add(child.text)
                
    logger.info(f"Collected {len(valid_units)} valid units")
    return valid_units

def remove_duplicate_men_at_arm_tags(root) -> int:
    """Remove duplicate men_at_arm tags within each faction."""
    changes = 0
    
    for faction in root.findall('.//faction'):
        men_at_arm_groups = {}
        
        # Collect all men_at_arm tags
        for men_at_arm in faction.findall('men_at_arm'):
            # Create a key based on the unit names in this group
            units = []
            for child in men_at_arm:
                if child.tag != 'name' and child.text:
                    units.append(child.text)
            # Sort units to ensure consistent key regardless of order
            units.sort()
            key = '|'.join(units)
            
            if key in men_at_arm_groups:
                # Duplicate found, mark for removal
                faction.remove(men_at_arm)
                changes += 1
                logger.debug(f"Removed duplicate men_at_arm group with units: {units}")
            else:
                men_at_arm_groups[key] = men_at_arm
    
    return changes

def reorganize_faction_children(faction) -> None:
    """Reorganize faction children in the preferred order."""
    # Define the preferred order
    order = ['name', 'description', 'colour', 'culture', 'default_symbol', 
             'symbol_index', 'background_image', 'settlement_images', 
             'allowed_climates', 'common_races', 'uncommon_races', 
             'warhorse_available', 'can_have_capital', 'agitator_character_pool',
             'diplomatic_relations', 'allied diplomatic_relations', 'enemy diplomatic_relations',
             'trade_routes', 'features', 'army_names', 'character_names', 
             'unit_names', 'ship_names', 'agent_names', 'religious_rebellions',
             'resource_production', 'building_info', 'clan_config', 
             'town_hall_stacks', 'port_stacks', 'recruitables', 'men_at_arm']
    
    # Create a mapping of tag names to elements
    elements = {}
    for child in faction:
        tag = child.tag
        if tag in elements:
            # Handle multiple elements with the same tag
            if not isinstance(elements[tag], list):
                elements[tag] = [elements[tag]]
            elements[tag].append(child)
        else:
            elements[tag] = child
    
    # Remove all children
    faction.clear()
    
    # Re-add children in the preferred order
    for tag in order:
        if tag in elements:
            if isinstance(elements[tag], list):
                for elem in elements[tag]:
                    faction.append(elem)
            else:
                faction.append(elements[tag])
    
    # Add any remaining elements that weren't in the order list
    for tag, elem in elements.items():
        if tag not in order:
            if isinstance(elem, list):
                for e in elem:
                    faction.append(e)
            else:
                faction.append(elem)

def reorder_attributes_in_all_tags(root) -> None:
    """Reorder attributes in all tags according to a preferred order."""
    # Define attribute orders for different tags
    attribute_orders = {
        'recruitables': ['can_retreat', 'is_mounted', 'is_exclusive'],
        'men_at_arm': ['can_retreat', 'is_mounted'],
        'unit': ['type', 'key', 'id'],
        'building': ['type', 'level', 'key'],
        # Add more as needed
    }
    
    def reorder_element_attrs(element):
        """Recursively reorder attributes for an element and its children."""
        if element.tag in attribute_orders:
            preferred_order = attribute_orders[element.tag]
            # Create new attrib dict with preferred order
            new_attrib = {}
            # Add ordered attributes first
            for attr in preferred_order:
                if attr in element.attrib:
                    new_attrib[attr] = element.attrib[attr]
            # Add remaining attributes
            for attr, value in element.attrib.items():
                if attr not in new_attrib:
                    new_attrib[attr] = value
            # Replace attributes
            element.attrib = new_attrib
        
        # Process children recursively
        for child in element:
            reorder_element_attrs(child)
    
    # Start reordering from root
    reorder_element_attrs(root)

def validate_faction_structure(faction) -> bool:
    """Validate that a faction has the required structure."""
    required_elements = ['name', 'culture']
    missing_elements = []
    
    for elem in required_elements:
        if faction.find(elem) is None:
            missing_elements.append(elem)
    
    if missing_elements:
        logger.warning(f"Faction missing required elements: {missing_elements}")
        return False
    
    return True

def process_units_xml(xml_file_path: str, cache_file: str = "llm_cache.json") -> None:
    """Process the units XML file to fix invalid unit references."""
    # Load the XML file
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        logger.info(f"Parsed XML file: {xml_file_path}")
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        logger.error(f"XML file not found: {xml_file_path}")
        return
    
    # Store initial state for change detection
    initial_xml_string = ET.tostring(root, encoding='unicode')
    
    # Load LLM cache
    llm_cache = load_cache(cache_file)
    cache_hits = 0
    llm_calls = 0
    
    # Collect all valid units
    valid_units = collect_all_valid_units(root)
    
    # Track changes
    procedural_replacements = 0
    llm_replacements = 0
    
    # Process each faction
    factions = root.findall('.//faction')
    logger.info(f"Found {len(factions)} factions to process")
    
    for faction in factions:
        faction_name_elem = faction.find('name')
        faction_name = faction_name_elem.text if faction_name_elem is not None else "Unknown"
        logger.debug(f"Processing faction: {faction_name}")
        
        # Process recruitables
        recruitables = faction.findall('.//recruitables')
        for rec_group in recruitables:
            for child in rec_group:
                if child.tag != 'name' and child.text and not is_valid_unit(child.text, valid_units):
                    old_unit = child.text
                    faction_prefix = get_faction_from_unit(old_unit)
                    
                    # Try procedural replacement first
                    suggested_unit = None
                    
                    # Look for a valid unit with the same type from the same faction
                    unit_type = extract_unit_type(old_unit)
                    faction_units = [unit for unit in valid_units if unit.startswith(f"{faction_prefix}_")]
                    for unit in faction_units:
                        if unit_type.lower() in unit.lower():
                            suggested_unit = unit
                            break
                    
                    # If still no suggestion, try any unit from the same faction
                    if not suggested_unit and faction_units:
                        suggested_unit = faction_units[0]
                        logger.debug(f"Used fallback unit {suggested_unit} for {old_unit}")
                    
                    if suggested_unit:
                        logger.info(f"Procedurally replacing {old_unit} with {suggested_unit} in {faction_name}")
                        child.text = suggested_unit
                        procedural_replacements += 1
                    else:
                        # Use LLM as last resort
                        logger.info(f"Using LLM to suggest replacement for {old_unit} in {faction_name}")
                        suggested_unit = suggest_valid_unit(old_unit, faction_prefix, valid_units, llm_cache, cache_file)
                        if suggested_unit and suggested_unit != old_unit:
                            logger.info(f"LLM replacing {old_unit} with {suggested_unit} in {faction_name}")
                            child.text = suggested_unit
                            llm_replacements += 1
                            llm_calls += 1
                        else:
                            logger.warning(f"No valid replacement found for {old_unit} in {faction_name}")
        
        # Process men_at_arm groups
        men_at_arms = faction.findall('.//men_at_arm')
        for ma_group in men_at_arms:
            for child in ma_group:
                if child.tag != 'name' and child.text and not is_valid_unit(child.text, valid_units):
                    old_unit = child.text
                    faction_prefix = get_faction_from_unit(old_unit)
                    
                    # Try procedural replacement first
                    suggested_unit = None
                    
                    # Look for a valid unit with the same type from the same faction
                    unit_type = extract_unit_type(old_unit)
                    faction_units = [unit for unit in valid_units if unit.startswith(f"{faction_prefix}_")]
                    for unit in faction_units:
                        if unit_type.lower() in unit.lower():
                            suggested_unit = unit
                            break
                    
                    # If still no suggestion, try any unit from the same faction
                    if not suggested_unit and faction_units:
                        suggested_unit = faction_units[0]
                        logger.debug(f"Used fallback unit {suggested_unit} for {old_unit}")
                    
                    if suggested_unit:
                        logger.info(f"Procedurally replacing {old_unit} with {suggested_unit} in {faction_name} (men_at_arm)")
                        child.text = suggested_unit
                        procedural_replacements += 1
                    else:
                        # Use LLM as last resort
                        logger.info(f"Using LLM to suggest replacement for {old_unit} in {faction_name} (men_at_arm)")
                        suggested_unit = suggest_valid_unit(old_unit, faction_prefix, valid_units, llm_cache, cache_file)
                        if suggested_unit and suggested_unit != old_unit:
                            logger.info(f"LLM replacing {old_unit} with {suggested_unit} in {faction_name} (men_at_arm)")
                            child.text = suggested_unit
                            llm_replacements += 1
                            llm_calls += 1
                        else:
                            logger.warning(f"No valid replacement found for {old_unit} in {faction_name} (men_at_arm)")
    
    # Remove duplicate men_at_arm tags
    duplicates_removed = remove_duplicate_men_at_arm_tags(root)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate men_at_arm groups")
    
    # Reorganize faction children
    for faction in factions:
        reorganize_faction_children(faction)
    
    # Reorder attributes in all tags
    reorder_attributes_in_all_tags(root)
    
    # Summarize changes
    logger.info(f"Total procedural replacements: {procedural_replacements}")
    logger.info(f"Total LLM replacements: {llm_replacements}")
    total_changes = procedural_replacements + llm_replacements
    logger.info(f"Total changes: {total_changes}")

    # Check if the XML content has changed (using string comparison)
    final_xml_string = ET.tostring(root, encoding='unicode')
    changes_made = final_xml_string != initial_xml_string

    # Save the updated XML if changes were made
    if changes_made:
        # Reorganize children one final time before saving
        for faction in factions:
            reorganize_faction_children(faction)
        
        # Remove duplicate men_at_arm tags again after all processing
        remove_duplicate_men_at_arm_tags(root)
        
        # Reorder attributes in all tags
        reorder_attributes_in_all_tags(root)
        
        # Validate the final XML structure
        for faction in factions:
            validate_faction_structure(faction)
        
        # Write the updated XML back to the file
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        logger.info(f"Updated factions XML saved to {xml_file_path}")
    else:
        logger.info("No changes made to the factions XML.")

def format_factions_xml_only(xml_file_path: str) -> None:
    """Format the factions XML file without making content changes."""
    # Load the XML file
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        logger.info(f"Parsed XML file for formatting: {xml_file_path}")
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        return
    except FileNotFoundError:
        logger.error(f"XML file not found: {xml_file_path}")
        return
    
    # Store initial state for change detection
    initial_xml_string = ET.tostring(root, encoding='unicode')
    
    # Process each faction for reorganization
    factions = root.findall('.//faction')
    logger.info(f"Found {len(factions)} factions to format")
    
    for faction in factions:
        reorganize_faction_children(faction)
    
    # Reorder attributes in all tags
    reorder_attributes_in_all_tags(root)
    
    # Check if changes were made
    final_xml_string = ET.tostring(root, encoding='unicode')
    changes_made = final_xml_string != initial_xml_string
    
    if changes_made:
        # Write the formatted XML back to the file
        tree.write(xml_file_path, encoding='utf-8', xml_declaration=True)
        logger.info(f"Formatted factions XML saved to {xml_file_path}")
    else:
        logger.info("No formatting changes made to the factions XML.")

def main():
    parser = argparse.ArgumentParser(description="Fix invalid unit references in factions XML")
    parser.add_argument("factions_xml_path", help="Path to the factions XML file")
    parser.add_argument("--format-xml-only", action="store_true", 
                       help="Only format the XML without making content changes")
    parser.add_argument("--cache-file", default="llm_cache.json", 
                       help="Path to the LLM response cache file")
    
    args = parser.parse_args()
    
    if args.format_xml_only:
        format_factions_xml_only(args.factions_xml_path)
    else:
        process_units_xml(args.factions_xml_path, args.cache_file)

if __name__ == "__main__":
    main()
