import re
import os

valid_keys_path = r"C:\Users\PDitty\Documents\MEGA\My_Mods\ck3\CC-Mappers\.cecli\temp\mk1212_unit_keys.txt"
with open(valid_keys_path, 'r', encoding='utf-8') as f:
    valid_keys_content = f.read()

valid_keys = set()
for line in valid_keys_content.split('\n'):
    line = line.strip()
    if line and not line.startswith('#') and not line.startswith('key') and not line.startswith('\ufeff'):
        valid_keys.add(line)

print(f"Valid keys count: {len(valid_keys)}")

xml_dir = r"C:\Users\PDitty\Documents\MEGA\My_Mods\ck3\CC-Mappers\OfficialCC_DefaultCK3_Renaissance_MK1212Mod\Titles"
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

all_xml_keys = []
for xml_file in xml_files:
    filepath = os.path.join(xml_dir, xml_file)
    with open(filepath, 'r', encoding='latin-1') as f:
        content = f.read()
    # Only match key attributes inside unit tags: General, Knights, MenAtArm
    keys = re.findall(r'<(?:General|Knights|MenAtArm)[^>]*\skey="([^"]+)"', content)
    for key in keys:
        all_xml_keys.append((xml_file, key))

print(f"Total keys found in XML files: {len(all_xml_keys)}")

invalid_keys = []
for xml_file, key in all_xml_keys:
    if key not in valid_keys:
        invalid_keys.append((xml_file, key))

print(f"Invalid keys count: {len(invalid_keys)}")

with open(r"C:\Users\PDitty\Documents\MEGA\My_Mods\ck3\CC-Mappers\invalid_keys_found_renaissance.txt", 'w', encoding='utf-8') as f:
    for xml_file, key in invalid_keys:
        f.write(f"{xml_file}: {key}\n")

print("Invalid keys saved to invalid_keys_found.txt")
