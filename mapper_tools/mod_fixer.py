import os
import argparse
import xml.etree.ElementTree as ET
import hashlib

def find_file(directory, filename):
    """
    Recursively searches for 'filename' within 'directory' using os.walk()
    and returns the full path if found, otherwise None.
    """
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def calculate_sha256(file_path, chunk_size=8192):
    """
    Computes and returns the SHA-256 hash of the file at 'file_path'.
    Reads files in chunks to efficiently handle large mod packs.
    """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}' for SHA-256 calculation.")
        return None
    except Exception as e:
        print(f"Error calculating SHA-256 for '{file_path}': {e}")
        return None

def indent_xml(elem, level=0):
    """
    Adds indentation to the XML tree for pretty printing.
    """
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

def main():
    parser = argparse.ArgumentParser(description="Update Mods.xml with SHA-256 hashes of mod pack files.")
    parser.add_argument("--mods-xml-path", required=True, help="Path to the Mods.xml file to be processed.")
    parser.add_argument("--attila-mods-directory", required=True, help="Path to the root directory where Attila mod .pack files are stored.")
    args = parser.parse_args()

    mods_xml_path = args.mods_xml_path
    attila_mods_directory = args.attila_mods_directory

    if not os.path.exists(mods_xml_path):
        print(f"Error: Mods XML file not found at '{mods_xml_path}'.")
        return
    if not os.path.isdir(attila_mods_directory):
        print(f"Error: Attila mods directory not found at '{attila_mods_directory}'.")
        return

    print(f"Processing Mods XML file: '{mods_xml_path}'")
    print(f"Searching for mod packs in: '{attila_mods_directory}'")

    try:
        tree = ET.parse(mods_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing Mods XML file '{mods_xml_path}': {e}. Aborting.")
        return

    changes_made = False
    for mod_element in root.findall('Mod'):
        mod_pack_name_with_at = mod_element.text
        if not mod_pack_name_with_at or not mod_pack_name_with_at.strip().startswith('@'):
            print(f"Warning: Skipping <Mod> tag with invalid text content: '{mod_pack_name_with_at}'. Expected '@mod_name.pack'.")
            continue

        mod_pack_filename = mod_pack_name_with_at.strip()[1:] # Strip leading '@'
        current_sha256 = mod_element.get('sha_256')

        print(f"  - Processing mod: '{mod_pack_filename}'")
        file_path = find_file(attila_mods_directory, mod_pack_filename)

        if file_path:
            calculated_sha256 = calculate_sha256(file_path)
            if calculated_sha256:
                if current_sha256 != calculated_sha256:
                    mod_element.set('sha_256', calculated_sha256)
                    changes_made = True
                    print(f"    -> Updated sha_256 for '{mod_pack_filename}' to '{calculated_sha256}'.")
                else:
                    print(f"    -> sha_256 for '{mod_pack_filename}' is already correct.")
            else:
                # If hash calculation failed, remove existing hash
                if 'sha_256' in mod_element.attrib:
                    del mod_element.attrib['sha_256']
                    changes_made = True
                    print(f"    -> Removed existing sha_256 for '{mod_pack_filename}' due to calculation error.")
        else:
            print(f"    -> Warning: Mod pack file '{mod_pack_filename}' not found in '{attila_mods_directory}'.")
            if 'sha_256' in mod_element.attrib:
                del mod_element.attrib['sha_256']
                changes_made = True
                print(f"    -> Removed existing sha_256 for '{mod_pack_filename}' as file is missing.")

    if changes_made:
        indent_xml(root)
        try:
            tree.write(mods_xml_path, encoding='utf-8', xml_declaration=True)
            print(f"\nSuccessfully updated '{mods_xml_path}' with SHA-256 hashes.")
        except Exception as e:
            print(f"Error saving updated XML to '{mods_xml_path}': {e}")
    else:
        print("\nNo changes were needed for Mods.xml.")

if __name__ == "__main__":
    main()
