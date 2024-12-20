import json
import os
from datetime import datetime


# Define logging levels
LOG_LEVELS = {
    "normal": 1,    # Basic information
    "detailed": 2,  # Adds more context
    "severe": 3     # Maximum details, including debug-level info
}

def write_to_json_log(log_file, identifier, level, data, image_path):
    
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump({}, f, indent=4)

    # Load the existing log data
    with open(log_file, "r") as f:
        log = json.load(f)

    # Ensure the identifier exists in the log, initialize if not
    if identifier not in log or not isinstance(log[identifier], dict):
        log[identifier] = {}

    # Update or append data
    for key, value in data.items():
        if key in log[identifier]:
            # Append values if key exists
            if isinstance(log[identifier][key], list):
                log[identifier][key].extend(value)
            else:
                log[identifier][key] = value  # Overwrite if not a list
        else:
            # Add key-value pair if it doesn't exist
            log[identifier][key] = value

    # Add severity level
    log[identifier]["severity"] = level

    # Handle image path
    if image_path:
        if "image" not in log[identifier]:
            log[identifier]["image"] = ''
        log[identifier]["image"] = image_path

    # Save updated log data back to the file
    with open(log_file, "w") as f:
        json.dump(log, f, indent=4)


def check_data(log_file, identifier):
    # identifiers = ['monos','links','root']
    # known_identifies = ['monos_known','links_known','root_known']

    identifiers = {
        'monos_known': 'monos',
        'links_known': 'links',
        'root_known': 'root'
    }

    # Load the existing log data
    with open(log_file, "r") as f:
        log = json.load(f)

    if identifier not in log:
        raise ValueError("Identifies is not present in the json file")


    data = log[identifier]

    for known_id, pred_id in identifiers.items():
        if (known_id in data) and (pred_id in data):
            known_val = data[known_id]

            for pred_val in data[pred_id]:
                if known_val != pred_val:
                    print(f'Data is incorrect for {identifier}: {pred_id}')




def log_data(identifier, level, data, image_path=None):
    write_to_json_log("logs.json", identifier, level, data, image_path)