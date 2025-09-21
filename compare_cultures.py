import xml.etree.ElementTree as ET
import sys
from collections import defaultdict

def parse_xml_file(file_path):
    """Parses an XML file and returns sets of heritage and culture names."""
    heritages = set()
    cultures = set()
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for heritage in root.findall('Heritage'):
            name = heritage.get('name')
            if name:
                heritages.add(name)
        for culture in root.findall('.//Culture'):
            name = culture.get('name')
            if name:
                cultures.add(name)
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
    return heritages, cultures

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_cultures.py <file1.xml> <file2.xml> ...")
        return

    files = sys.argv[1:]
    file_data = {}
    all_heritages = set()
    all_cultures = set()

    for f in files:
        heritages, cultures = parse_xml_file(f)
        file_data[f] = {'heritages': heritages, 'cultures': cultures}
        all_heritages.update(heritages)
        all_cultures.update(cultures)

    print("--- Comparison Report ---")

    for f in files:
        print(f"\n--- Unique entries in: {f} ---")
        
        other_files = [other for other in files if other != f]
        
        # Unique Heritages
        other_heritages = set()
        for other in other_files:
            other_heritages.update(file_data[other]['heritages'])
        unique_heritages = file_data[f]['heritages'] - other_heritages
        
        if unique_heritages:
            print("\nUnique Heritages:")
            for item in sorted(list(unique_heritages)):
                print(f"  - {item}")
        else:
            print("\nNo unique heritages.")

        # Unique Cultures
        other_cultures = set()
        for other in other_files:
            other_cultures.update(file_data[other]['cultures'])
        unique_cultures = file_data[f]['cultures'] - other_cultures

        if unique_cultures:
            print("\nUnique Cultures:")
            for item in sorted(list(unique_cultures)):
                print(f"  - {item}")
        else:
            print("\nNo unique cultures.")

if __name__ == "__main__":
    main()
