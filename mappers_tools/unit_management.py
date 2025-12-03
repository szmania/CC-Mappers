import glob
import xml.etree.ElementTree as ET
import os

def find_faction_files(base_path):
    """
    Recursively searches for faction XML files within a given base path.
    Assumes faction files are named like 'faction_*.xml' or similar.

    Args:
        base_path (str): The root directory to start the search from.

    Returns:
        list: A list of absolute paths to the discovered faction XML files.
    """
    # Adjust pattern based on actual file naming convention.
    # For now, let's assume they are in any subdirectory and end with .xml
    search_pattern = os.path.join(base_path, '**', '*.xml')
    # Filter for files that likely represent factions, e.g., containing 'faction' in name
    # This is a heuristic and might need adjustment based on actual file structure.
    faction_files = [f for f in glob.glob(search_pattern, recursive=True) if 'faction' in os.path.basename(f).lower()]
    return faction_files

def parse_faction_data(file_path):
    """
    Parses a faction XML file to extract faction name, culture, and unit rosters.

    Args:
        file_path (str): The path to the faction XML file.

    Returns:
        dict: A dictionary containing 'name', 'culture', and 'units' (a structured dict of unit lists).
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        faction_name = root.findtext('name', default='Unknown Faction')
        culture = root.findtext('culture', default='unknown_culture')

        units = {
            'General': [],
            'Knights': [],
            'Levies': [],
            'Garrison': [],
            'MenAtArm': []
        }

        # Parse core roster units
        for unit_type in ['General', 'Knights', 'Levies', 'Garrison']:
            type_node = root.find(unit_type)
            if type_node is not None:
                for unit_node in type_node.findall('unit'):
                    unit_key = unit_node.get('key')
                    if unit_key:
                        units[unit_type].append(unit_key)
        
        # Parse MenAtArm units (more complex structure)
        maa_node = root.find('MenAtArm')
        if maa_node is not None:
            for unit_node in maa_node.findall('unit'):
                maa_entry = {
                    'key': unit_node.get('key'),
                    'min_rank': int(unit_node.get('min_rank', 0)),
                    'max_rank': int(unit_node.get('max_rank', 0)),
                    'min_amount': int(unit_node.get('min_amount', 0)),
                    'max_amount': int(unit_node.get('max_amount', 0))
                }
                units['MenAtArm'].append(maa_entry)

        return {
            'name': faction_name,
            'culture': culture,
            'units': units
        }
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing {file_path}: {e}")
        return None

def get_master_unit_list():
    """
    Placeholder function to return a comprehensive list of all possible units in the game.
    In a real scenario, this would load from a master data file (e.g., a CSV, JSON, or another XML).

    Returns:
        list: A hardcoded list of unit keys for testing purposes.
    """
    return [
        'roman_legionaries', 'roman_archers', 'roman_cavalry', 'roman_spearmen',
        'barbarian_spearmen', 'barbarian_axemen', 'barbarian_archers',
        'eastern_swordsmen', 'eastern_archers', 'eastern_cavalry',
        'generic_swordsmen', 'generic_archers', 'generic_spearmen', 'generic_cavalry',
        'greek_hoplites', 'greek_archers', 'greek_cavalry',
        'egyptian_spearmen', 'egyptian_archers', 'egyptian_chariots',
        'saxon_huscarls', 'saxon_thegns', 'saxon_archers',
        'viking_berserkers', 'viking_raiders', 'viking_archers',
        'frankish_knights', 'frankish_spearmen', 'frankish_archers',
        'gothic_warriors', 'gothic_archers', 'gothic_cavalry',
        'horde_horse_archers', 'horde_lancers', 'horde_spearmen',
        'byzantine_cataphracts', 'byzantine_skoutatoi', 'byzantine_archers',
        'sassanid_cataphracts', 'sassanid_spearmen', 'sassanid_archers',
        'arab_camel_archers', 'arab_spearmen', 'arab_swordsmen',
        'african_spearmen', 'african_archers', 'african_axemen',
        'indian_elephants', 'indian_archers', 'indian_spearmen',
        'chinese_crossbowmen', 'chinese_spearmen', 'chinese_cavalry',
        'japanese_samurai', 'japanese_ashigaru', 'japanese_archers',
        'aztec_jaguar_warriors', 'aztec_eagle_warriors', 'aztec_archers',
        'inca_slingers', 'inca_spearmen', 'inca_axemen',
        'maya_spearmen', 'maya_archers', 'maya_axemen',
        'mongol_horse_archers', 'mongol_lancers', 'mongol_spearmen',
        'turk_horse_archers', 'turk_lancers', 'turk_spearmen',
        'rus_druzhina', 'rus_spearmen', 'rus_archers',
        'polish_knights', 'polish_spearmen', 'polish_archers',
        'hungarian_hussars', 'hungarian_spearmen', 'hungarian_archers',
        'bohemian_knights', 'bohemian_spearmen', 'bohemian_archers',
        'lithuanian_spearmen', 'lithuanian_archers', 'lithuanian_cavalry',
        'teutonic_knights', 'teutonic_spearmen', 'teutonic_archers',
        'crusader_knights', 'crusader_spearmen', 'crusader_archers',
        'templar_knights', 'templar_spearmen', 'templar_archers',
        'hospitaller_knights', 'hospitaller_spearmen', 'hospitaller_archers',
        'swiss_pikemen', 'swiss_halberdiers', 'swiss_archers',
        'landsknechts', 'landsknecht_pikemen', 'landsknecht_arquebusiers',
        'spanish_conquistadors', 'spanish_rodeleros', 'spanish_arquebusiers',
        'portuguese_arquebusiers', 'portuguese_spearmen', 'portuguese_cavalry',
        'french_knights', 'french_archers', 'french_spearmen',
        'english_longbowmen', 'english_knights', 'english_spearmen',
        'scottish_pikemen', 'scottish_highlanders', 'scottish_archers',
        'irish_kerns', 'irish_galloglaich', 'irish_archers',
        'welsh_longbowmen', 'welsh_spearmen', 'welsh_cavalry',
        'norse_hirdmen', 'norse_berserkers', 'norse_archers',
        'danish_huscarls', 'danish_axemen', 'danish_archers',
        'swedish_caroleans', 'swedish_pikemen', 'swedish_archers',
        'venetian_marines', 'venetian_archers', 'venetian_cavalry',
        'genovese_crossbowmen', 'genovese_spearmen', 'genovese_cavalry',
        'milanese_knights', 'milanese_crossbowmen', 'milanese_spearmen',
        'papal_guard', 'papal_spearmen', 'papal_archers',
        'florentine_militia', 'florentine_archers', 'florentine_cavalry',
        'naples_spearmen', 'naples_archers', 'naples_cavalry',
        'sicilian_spearmen', 'sicilian_archers', 'sicilian_cavalry',
        'austrian_pikemen', 'austrian_archers', 'austrian_cavalry',
        'bohemian_pikemen', 'bohemian_archers', 'bohemian_cavalry',
        'dutch_pikemen', 'dutch_arquebusiers', 'dutch_cavalry',
        'belgian_pikemen', 'belgian_arquebusiers', 'belgian_cavalry',
        'prussian_musketeers', 'prussian_pikemen', 'prussian_cavalry',
        'russian_streltsy', 'russian_cossacks', 'russian_spearmen',
        'ottoman_janissaries', 'ottoman_sipahis', 'ottoman_archers',
        'mamluk_cavalry', 'mamluk_archers', 'mamluk_spearmen',
        'timurid_elephants', 'timurid_horse_archers', 'timurid_lancers',
        'safavid_qizilbash', 'safavid_archers', 'safavid_spearmen',
        'mughal_elephants', 'mughal_archers', 'mughal_spearmen',
        'afghan_ghilzai', 'afghan_archers', 'afghan_cavalry',
        'persian_immortals', 'persian_archers', 'persian_cavalry',
        'indian_rajputs', 'indian_archers', 'indian_elephants',
        'thai_elephants', 'thai_archers', 'thai_spearmen',
        'vietnamese_elephants', 'vietnamese_archers', 'vietnamese_spearmen',
        'khmer_elephants', 'khmer_archers', 'khmer_spearmen',
        'malay_spearmen', 'malay_archers', 'malay_kris_warriors',
        'filipino_spearmen', 'filipino_archers', 'filipino_swordsmen',
        'indonesian_spearmen', 'indonesian_archers', 'indonesian_swordsmen',
        'australian_aboriginal_warriors', 'australian_aboriginal_spearmen', 'australian_aboriginal_archers',
        'new_zealand_maori_warriors', 'new_zealand_maori_spearmen', 'new_zealand_maori_archers',
        'polynesian_warriors', 'polynesian_spearmen', 'polynesian_archers',
        'native_american_warriors', 'native_american_spearmen', 'native_american_archers',
        'mesoamerican_warriors', 'mesoamerican_spearmen', 'mesoamerican_archers',
        'south_american_warriors', 'south_american_spearmen', 'south_american_archers',
        'african_tribal_warriors', 'african_tribal_spearmen', 'african_tribal_archers',
        'siberian_hunters', 'siberian_spearmen', 'siberian_archers',
        'arctic_hunters', 'arctic_spearmen', 'arctic_archers',
        'steppe_nomads', 'steppe_horse_archers', 'steppe_lancers',
        'desert_raiders', 'desert_camel_archers', 'desert_spearmen',
        'forest_hunters', 'forest_spearmen', 'forest_archers',
        'mountain_warriors', 'mountain_spearmen', 'mountain_archers',
        'island_warriors', 'island_spearmen', 'island_archers',
        'jungle_warriors', 'jungle_spearmen', 'jungle_archers',
        'swamp_warriors', 'swamp_spearmen', 'swamp_archers',
        'city_militia', 'city_guards', 'city_archers',
        'peasant_levies', 'peasant_archers', 'peasant_spearmen',
        'mercenary_swordsmen', 'mercenary_archers', 'mercenary_spearmen',
        'mercenary_cavalry', 'mercenary_pikemen', 'mercenary_crossbowmen',
        'noble_knights', 'royal_guards', 'elite_archers',
        'general_bodyguard', 'general_cavalry', 'general_infantry',
        'captain_bodyguard', 'captain_cavalry', 'captain_infantry',
        'hero_unit', 'legendary_unit', 'mythical_unit',
        'unique_unit_a', 'unique_unit_b', 'unique_unit_c',
        'special_unit_x', 'special_unit_y', 'special_unit_z',
        'event_unit_1', 'event_unit_2', 'event_unit_3',
        'campaign_unit_alpha', 'campaign_unit_beta', 'campaign_unit_gamma',
        'dlc_unit_i', 'dlc_unit_ii', 'dlc_unit_iii',
        'mod_unit_foo', 'mod_unit_bar', 'mod_unit_baz',
        'test_unit_1', 'test_unit_2', 'test_unit_3'
    ]

def get_cultural_affinity_map():
    """
    Placeholder function to return a dictionary mapping cultures to their allowed unit prefixes.
    In a real scenario, this would load from a configuration file.

    Returns:
        dict: A hardcoded dictionary for testing purposes.
    """
    return {
        'roman': ['roman_', 'bel_'], # Example: Roman culture can use Roman and some Belgian units
        'barbarian': ['barbarian_', 'germanic_', 'celtic_'],
        'eastern': ['eastern_', 'persian_', 'sassanid_'],
        'greek': ['greek_', 'hellenic_'],
        'egyptian': ['egyptian_', 'nubian_'],
        'saxon': ['saxon_', 'anglo_'],
        'viking': ['viking_', 'norse_'],
        'frankish': ['frankish_', 'gallic_'],
        'gothic': ['gothic_', 'visigothic_'],
        'horde': ['horde_', 'mongol_', 'timurid_'],
        'byzantine': ['byzantine_', 'ere_'], # Eastern Roman Empire
        'arab': ['arab_', 'saracen_', 'mamluk_'],
        'african': ['african_', 'ethiopian_'],
        'indian': ['indian_', 'mughal_'],
        'chinese': ['chinese_', 'han_'],
        'japanese': ['japanese_', 'samurai_'],
        'aztec': ['aztec_', 'mexica_'],
        'inca': ['inca_', 'quechua_'],
        'maya': ['maya_', 'yucatan_'],
        'mongol': ['mongol_', 'horde_'],
        'turk': ['turk_', 'ottoman_'],
        'rus': ['rus_', 'slavic_'],
        'polish': ['polish_', 'slavic_'],
        'hungarian': ['hungarian_', 'magyar_'],
        'bohemian': ['bohemian_', 'czech_'],
        'lithuanian': ['lithuanian_', 'baltic_'],
        'teutonic': ['teutonic_', 'germanic_'],
        'crusader': ['crusader_', 'templar_', 'hospitaller_'],
        'swiss': ['swiss_', 'alpine_'],
        'landsknecht': ['landsknecht_', 'german_'],
        'spanish': ['spanish_', 'iberian_'],
        'portuguese': ['portuguese_', 'iberian_'],
        'french': ['french_', 'gallic_'],
        'english': ['english_', 'anglo_'],
        'scottish': ['scottish_', 'celtic_'],
        'irish': ['irish_', 'celtic_'],
        'welsh': ['welsh_', 'celtic_'],
        'norse': ['norse_', 'viking_'],
        'danish': ['danish_', 'norse_'],
        'swedish': ['swedish_', 'norse_'],
        'venetian': ['venetian_', 'italian_'],
        'genovese': ['genovese_', 'italian_'],
        'milanese': ['milanese_', 'italian_'],
        'papal': ['papal_', 'italian_'],
        'florentine': ['florentine_', 'italian_'],
        'naples': ['naples_', 'italian_'],
        'sicilian': ['sicilian_', 'italian_'],
        'austrian': ['austrian_', 'germanic_'],
        'dutch': ['dutch_', 'germanic_'],
        'belgian': ['belgian_', 'germanic_'],
        'prussian': ['prussian_', 'germanic_'],
        'ottoman': ['ottoman_', 'turk_'],
        'safavid': ['safavid_', 'persian_'],
        'afghan': ['afghan_', 'pashtun_'],
        'thai': ['thai_', 'siamese_'],
        'vietnamese': ['vietnamese_', 'annam_'],
        'khmer': ['khmer_', 'cambodian_'],
        'malay': ['malay_', 'indonesian_'],
        'filipino': ['filipino_', 'malay_'],
        'indonesian': ['indonesian_', 'malay_'],
        'australian': ['australian_', 'aboriginal_'],
        'new_zealand': ['new_zealand_', 'maori_'],
        'polynesian': ['polynesian_', 'pacific_'],
        'native_american': ['native_american_', 'indigenous_'],
        'mesoamerican': ['mesoamerican_', 'aztec_', 'maya_'],
        'south_american': ['south_american_', 'inca_'],
        'siberian': ['siberian_', 'arctic_'],
        'arctic': ['arctic_', 'inuit_'],
        'steppe': ['steppe_', 'nomad_'],
        'desert': ['desert_', 'bedouin_'],
        'forest': ['forest_', 'woodland_'],
        'mountain': ['mountain_', 'highland_'],
        'island': ['island_', 'coastal_'],
        'jungle': ['jungle_', 'tropical_'],
        'swamp': ['swamp_', 'marsh_'],
        'city': ['city_', 'urban_'],
        'peasant': ['peasant_', 'rural_'],
        'mercenary': ['mercenary_', 'hired_'],
        'noble': ['noble_', 'elite_'],
        'general': ['general_', 'commander_'],
        'captain': ['captain_', 'leader_'],
        'hero': ['hero_', 'legendary_'],
        'unique': ['unique_', 'special_'],
        'event': ['event_', 'seasonal_'],
        'campaign': ['campaign_', 'story_'],
        'dlc': ['dlc_', 'expansion_'],
        'mod': ['mod_', 'custom_'],
        'test': ['test_', 'debug_'],
        'unknown_culture': [] # Default for unknown cultures
    }

def apply_roster_corrections(file_path, corrections):
    """
    Applies unit key corrections to the core roster sections of an XML file.

    Args:
        file_path (str): The path to the XML file.
        corrections (dict): A dictionary where keys are original unit keys and values are corrected keys.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        changed = False
        for unit_type in ['General', 'Knights', 'Levies', 'Garrison']:
            type_node = root.find(unit_type)
            if type_node is not None:
                for unit_node in type_node.findall('unit'):
                    original_key = unit_node.get('key')
                    if original_key in corrections:
                        new_key = corrections[original_key]
                        if original_key != new_key:
                            unit_node.set('key', new_key)
                            changed = True
                            print(f"  Corrected unit '{original_key}' to '{new_key}' in {unit_type}.")
        
        if changed:
            # Use a temporary file for atomic write to prevent data loss on error
            temp_file_path = file_path + ".tmp"
            tree.write(temp_file_path, encoding='utf-8', xml_declaration=True)
            os.replace(temp_file_path, file_path) # Atomically replace the original file
            print(f"Applied core roster corrections to {file_path}")
        else:
            print(f"No core roster corrections needed for {file_path}")
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path} for corrections: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while applying roster corrections to {file_path}: {e}")

def apply_maa_remapping(file_path, remapping):
    """
    Removes existing MenAtArm nodes and generates new ones based on the provided remapping.

    Args:
        file_path (str): The path to the XML file.
        remapping (dict): A dictionary containing the 'men_at_arms' list of new unit entries.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        maa_node = root.find('MenAtArm')
        if maa_node is None:
            maa_node = ET.SubElement(root, 'MenAtArm')
            print(f"  Created new <MenAtArm> node in {file_path}")
        else:
            # Remove all existing unit children to ensure a clean rewrite
            for unit_node in maa_node.findall('unit'):
                maa_node.remove(unit_node)
            print(f"  Cleared existing <MenAtArm> units in {file_path}")

        if 'men_at_arms' in remapping and isinstance(remapping['men_at_arms'], list):
            for maa_entry in remapping['men_at_arms']:
                unit_elem = ET.SubElement(maa_node, 'unit')
                unit_elem.set('key', maa_entry.get('key', ''))
                unit_elem.set('min_rank', str(maa_entry.get('min_rank', 0)))
                unit_elem.set('max_rank', str(maa_entry.get('max_rank', 0)))
                unit_elem.set('min_amount', str(maa_entry.get('min_amount', 0)))
                unit_elem.set('max_amount', str(maa_entry.get('max_amount', 0)))
                print(f"  Added MenAtArm unit: {maa_entry.get('key')}")
            
            # Use a temporary file for atomic write
            temp_file_path = file_path + ".tmp"
            tree.write(temp_file_path, encoding='utf-8', xml_declaration=True)
            os.replace(temp_file_path, file_path) # Atomically replace the original file
            print(f"Applied MenAtArm remapping to {file_path}")
        else:
            print(f"No valid MenAtArm remapping provided for {file_path}")
    except ET.ParseError as e:
        print(f"Error parsing XML file {file_path} for remapping: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while applying MenAtArm remapping to {file_path}: {e}")
