import os
import json

def create_json_template(sourceDirectory, DestinationDirectory):
    """
    Create a JSON template with an array containing file names from CSV files in the specified directory,
    with default "process" value set to True.

    """
    files = []
    
    # Iterate over files in the directory
    for filename in os.listdir(SD):
        if filename.endswith(".csv"):
            files.append({"file_name": filename, "process": True})
    
    # Create JSON data
    data = {"files": files}
    
    # Write JSON data to a file
    with open(DestinationDirectory + "datafiles.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("JSON template created successfully!")

# Example usage:
SD = "../datasets"
DD = "../"
create_json_template(SD, DD)
