import os
from aider.tools.base_tool import BaseAiderTool

try:
    import pandas as pd
except ImportError:
    pd = None

class ReadExcelFromFanMapsTool(BaseAiderTool):
    """
    A tool to read all XLSX files from data/mods/agot_seven_kingdoms/cc_fan_maps/
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "ReadExcelFromFanMapsTool",
                "description": "Reads all XLSX files from the `data/mods/agot_seven_kingdoms/cc_fan_maps/` directory and returns their contents as text.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            },
        }

    def run(self, **kwargs):
        """
        Reads all XLSX files from the specified directory and returns their contents.
        """
        if pd is None:
            error_message = "The 'pandas' and 'openpyxl' libraries are required to run this tool. Please install them (`pip install pandas openpyxl`)."
            self.coder.io.tool_error(error_message)
            return f"Error: {error_message}"

        dir_path = "data/mods/agot_seven_kingdoms/cc_fan_maps"
        abs_dir_path = self.coder.abs_root_path(dir_path)

        if not os.path.isdir(abs_dir_path):
            error_message = f"Directory not found at {dir_path}"
            self.coder.io.tool_error(error_message)
            return f"Error: {error_message}"

        try:
            xlsx_files = [f for f in os.listdir(abs_dir_path) if f.endswith('.xlsx')]
        except Exception as e:
            error_message = f"Error listing files in {dir_path}: {str(e)}"
            self.coder.io.tool_error(error_message)
            return f"Error: {error_message}"

        if not xlsx_files:
            message = f"No .xlsx files found in {dir_path}"
            self.coder.io.tool_output(message)
            return message

        all_data = {}
        for filename in xlsx_files:
            file_path = os.path.join(abs_dir_path, filename)
            try:
                # Reading all sheets if multiple exist
                xls = pd.ExcelFile(file_path)
                content = ""
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    content += f"\n--- Sheet: {sheet_name} ---\n"
                    content += df.to_string()
                all_data[filename] = content.strip()
            except Exception as e:
                all_data[filename] = f"Error reading file: {str(e)}"

        output_str = f"Found and read {len(xlsx_files)} XLSX files from {dir_path}:\n\n"
        for filename, content in all_data.items():
            output_str += f"--- Contents of {filename} ---\n"
            output_str += content
            output_str += "\n\n"

        return output_str.strip()
