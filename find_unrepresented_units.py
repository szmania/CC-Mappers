import os
import re
import xml.etree.ElementTree as ET
import argparse

def find_maa_in_txt_files(directory):
    """Finds all men-at-arms unit definitions in .txt files in a directory."""
    maa_units = set()
    # Regex to find patterns like 'unit_name = {'
    maa_pattern = re.compile(r'^\s*([a-zA-Z0-9_]+)\s*=\s*\{', re.MULTILINE)

    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return maa_units

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                    matches = maa_pattern.findall(content)
                    for match in matches:
                        # These are variables, not units, so ignore them
                        if match.startswith('@') or match in ['terrain_bonus', 'winter_bonus', 'counters', 'buy_cost', 'low_maintenance_cost', 'high_maintenance_cost', 'ai_quality', 'limit', 'can_recruit', 'should_show_when_unavailable', 'access_through_subject', 'holding_bonus']:
                            continue
                        maa_units.add(match)
            except Exception as e:
                print(f"Error reading or parsing {filename}: {e}")
    return maa_units

def find_maa_in_xml_file(xml_file):
    """Finds all MenAtArm types in the Factions XML file."""
    if not os.path.exists(xml_file):
        print(f"Error: XML file not found at {xml_file}")
        return set()
    
    maa_units = set()
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for men_at_arm in root.findall('.//MenAtArm'):
            unit_type = men_at_arm.get('type')
            if unit_type:
                maa_units.add(unit_type)
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")
    return maa_units

def main():
    parser = argparse.ArgumentParser(description="Find unrepresented men-at-arms units.")
    parser.add_argument("--txt-dir", required=True, help="Directory containing the .txt files with men-at-arms definitions.")
    parser.add_argument("--xml-file", required=True, help="The XML file to check for representation.")
    args = parser.parse_args()

    txt_units = find_maa_in_txt_files(args.txt_dir)
    xml_units = find_maa_in_xml_file(args.xml_file)

    unrepresented_units = txt_units - xml_units

    if unrepresented_units:
        print("The following men-at-arms units were found in the .txt files but are not represented in the XML file:")
        for unit in sorted(list(unrepresented_units)):
            print(f"- {unit}")
    else:
        print("All men-at-arms units from the .txt files are represented in the XML file.")

if __name__ == "__main__":
    main()
