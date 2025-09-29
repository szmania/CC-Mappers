import os
import argparse
import xml.etree.ElementTree as ET
import hashlib

def find_files(directory, filename):
    """
    Recursively searches for all occurrences of 'filename' within 'directory'
    using os.walk() and returns a list of all full paths where the file is found.
    If no files are found, it returns an an empty list.
    """
    found_paths = []
    for root, _, files in os.walk(directory):
        if filename in files:
            found_paths.append(os.path.join(root, filename))
    return found_paths

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
    parser.add_argument("--attila-mods-dir", required=True, help="Path to the root directory where Attila mod .pack files are stored.")
    args = parser.parse_args()

    mods_xml_path = args.mods_xml_path
    attila_mods_dir = args.attila_mods_dir

    if not os.path.exists(mods_xml_path):
        print(f"Error: Mods XML file not found at '{mods_xml_path}'.")
        return
    if not os.path.isdir(attila_mods_dir):
        print(f"Error: Attila mods directory not found at '{attila_mods_dir}'.")
        return

    print(f"Processing Mods XML file: '{mods_xml_path}'")
    print(f"Searching for mod packs in: '{attila_mods_dir}'")

    try:
        tree = ET.parse(mods_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing Mods XML file '{mods_xml_path}': {e}. Aborting.")
        return

    changes_made = False
    for mod_element in root.findall('Mod'):
        mod_text_content = mod_element.text
        if not mod_text_content or not mod_text_content.strip():
            print(f"Warning: Skipping <Mod> tag with empty or whitespace text content.")
            continue

        mod_text_content_stripped = mod_text_content.strip()
        
        if mod_text_content_stripped.startswith('@'):
            mod_pack_filename = mod_text_content_stripped[1:]
        else:
            mod_pack_filename = mod_text_content_stripped
            
        current_sha256 = mod_element.get('sha256')

        print(f"  - Processing mod: '{mod_pack_filename}'")
        found_paths = find_files(attila_mods_dir, mod_pack_filename)

        file_to_hash = None

        if not found_paths:
            print(f"    -> Warning: Mod pack file '{mod_pack_filename}' not found in '{attila_mods_dir}'.")
            if 'sha256' in mod_element.attrib:
                del mod_element.attrib['sha256']
                changes_made = True
                print(f"    -> Removed existing sha256 for '{mod_pack_filename}' as file is missing.")
            continue # Move to the next mod

        elif len(found_paths) == 1:
            file_to_hash = found_paths[0]
            print(f"    -> Found unique file: '{file_to_hash}'")

        else: # Multiple files found (duplicates)
            print(f"    -> WARNING: Multiple instances of mod pack file '{mod_pack_filename}' found:")
            for i, path in enumerate(found_paths):
                print(f"       {i+1}. {path}")
            print(f"       0. Skip this mod.")

            while True:
                try:
                    choice = input("       Enter your choice (0 to skip): ")
                    user_choice = int(choice)
                    if 0 <= user_choice <= len(found_paths):
                        break
                    else:
                        print("       Invalid choice. Please enter a number within the range.")
                except ValueError:
                    print("       Invalid input. Please enter a number.")

            if user_choice == 0:
                print(f"    -> Skipping mod '{mod_pack_filename}'.")
                continue # Move to the next mod
            else:
                file_to_hash = found_paths[user_choice - 1]
                print(f"    -> Selected file: '{file_to_hash}'")

        # Hashing logic, executed only if a definitive file_to_hash is determined
        if file_to_hash:
            calculated_sha256 = calculate_sha256(file_to_hash)
            if calculated_sha256:
                if current_sha256 != calculated_sha256:
                    mod_element.set('sha256', calculated_sha256)
                    changes_made = True
                    print(f"    -> Updated sha256 for '{mod_pack_filename}' to '{calculated_sha256}'.")
                else:
                    print(f"    -> sha256 for '{mod_pack_filename}' is already correct.")
            else:
                # If hash calculation failed, remove existing hash
                if 'sha256' in mod_element.attrib:
                    del mod_element.attrib['sha256']
                    changes_made = True
                    print(f"    -> Removed existing sha256 for '{mod_pack_filename}' due to calculation error.")

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
