import os
import glob
from lxml import etree
from aider.tools.base_tool import BaseAiderTool

class XmlSchemaValidatorTool(BaseAiderTool):
    """
    A tool to validate XML files in OfficialCC_* folders against their corresponding schemas.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "XmlSchemaValidatorTool",
                "description": (
                    "Validates XML files within any 'OfficialCC_*' directories against their corresponding XSD schemas "
                    "located in the 'schemas/' directory. It checks Mods.xml, Factions.xml, Culture.xml, "
                    "Terrain.xml, TimePeriod.xml, and Titles.xml, and reports any validation errors."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def run(self, **kwargs):
        """
        Executes the XML schema validation process across all OfficialCC_* folders.
        
        :return: A string summarizing the validation results.
        """
        results = []
        project_root = self.coder.root

        schema_map = {
            "Mods.xml": "mods.xsd",
            "Factions.xml": "factions.xsd",
            "Culture.xml": "cultures.xsd",
            "Terrain.xml": "terrains.xsd",
            "TimePeriod.xml": "timeperiod.xsd",
            "Titles.xml": "titles.xsd",
        }

        # Find all OfficialCC_* directories
        search_path = os.path.join(project_root, 'OfficialCC_*')
        mod_dirs = glob.glob(search_path)

        if not mod_dirs:
            msg = "No 'OfficialCC_*' directories found in the project root."
            self.coder.io.tool_output(msg)
            return msg

        self.coder.io.tool_output(f"Found {len(mod_dirs)} 'OfficialCC_*' director(y/ies). Starting XML schema validation...")

        found_any_xml = False
        for mod_dir in mod_dirs:
            results.append(f"\n--- Checking directory: {os.path.basename(mod_dir)} ---")
            for xml_filename, xsd_filename in schema_map.items():
                xml_path = os.path.join(mod_dir, xml_filename)
                xsd_path = self.coder.abs_root_path(os.path.join('schemas', xsd_filename))

                if not os.path.exists(xml_path):
                    continue
                
                found_any_xml = True

                if not os.path.exists(xsd_path):
                    msg = f"WARNING: Schema file not found for {xml_filename}. Skipping validation. Expected at: {os.path.relpath(xsd_path, project_root)}"
                    results.append(msg)
                    self.coder.io.tool_output(msg)
                    continue

                try:
                    # Parse the XML schema
                    schema_doc = etree.parse(xsd_path)
                    xmlschema = etree.XMLSchema(schema_doc)

                    # Parse the XML file
                    xml_doc = etree.parse(xml_path)

                    # Validate
                    xmlschema.assertValid(xml_doc)
                    
                    msg = f"SUCCESS: {os.path.relpath(xml_path, project_root)} is valid."
                    results.append(msg)
                    self.coder.io.tool_output(msg)

                except etree.DocumentInvalid as e:
                    error_msg = f"FAILURE: {os.path.relpath(xml_path, project_root)} is NOT valid.\n  Reason: {str(e)}"
                    results.append(error_msg)
                    self.coder.io.tool_error(error_msg)
                except etree.XMLSyntaxError as e:
                    error_msg = f"ERROR: Could not parse {os.path.relpath(xml_path, project_root)}.\n  Reason: {str(e)}"
                    results.append(error_msg)
                    self.coder.io.tool_error(error_msg)
                except Exception as e:
                    error_msg = f"ERROR: An unexpected error occurred while processing {os.path.relpath(xml_path, project_root)}.\n  Reason: {str(e)}"
                    results.append(error_msg)
                    self.coder.io.tool_error(error_msg)
        
        if not found_any_xml:
            msg = "Found 'OfficialCC_*' directories, but no matching XML files (e.g., Factions.xml, Mods.xml) to validate."
            self.coder.io.tool_output(msg)
            return msg

        summary = "\n".join(results)
        self.coder.io.tool_output("\nValidation summary sent to model.")
        return summary
