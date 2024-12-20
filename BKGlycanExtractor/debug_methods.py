
import os
import json
import shutil

import logging
import json
import os
from pathlib import Path
from datetime import datetime

# Note - maybe you can store the json file instance in memory ionstead of openining and closing it 
# multiple times


class DebugMode:
    debug = False
    json_file = ""
    current_folder = ""
    curr_image = ""
    glycan_folder = ""
    image_path = ""
    level = 1
    info = None

    LOG_LEVELS = {
        1: 'normal',    # Basic information
        2: 'detailed',  # Adds more context
        3: 'severe'     # Maximum details, including debug-level info
    }

    @staticmethod
    def create_unique_folder(base_dir='debug_data'):
        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)

        # Get all subdirectories in the base directory, which should be numeric folder names
        existing_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

        # Filter folder names to only include numeric values, then find the maximum
        folder_numbers = [int(folder) for folder in existing_folders if folder.isdigit()]
        next_folder_number = max(folder_numbers, default=0) + 1  # Default 0 if no folders exist

        # Create the new folder with the next available number
        new_folder_path = os.path.join(base_dir, str(next_folder_number))
        os.makedirs(new_folder_path, exist_ok=True)

        # self.current_folder_path = os.path.join(os.getcwd(),new_folder_path)

        return os.path.join(os.getcwd(),new_folder_path)


    @staticmethod
    def write_to_json_log(identifier, data, **kwargs):
        log_file = DebugMode.json_file

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
        log[identifier]["severity"] = DebugMode.level

        # Handle image path
        # if image_path:
        #     if "image" not in log[identifier]:
        #         log[identifier]["image"] = ''
        log[identifier]["image"] = DebugMode.image_path

        # Save updated log data back to the file
        with open(log_file, "w") as f:
            json.dump(log, f, indent=4)

    @staticmethod
    def log_data(identifier, data, **kwargs):
        DebugMode.write_to_json_log(identifier, data, **kwargs)


    @staticmethod
    def check_data(identifier):

        log_file = DebugMode.json_file
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

                pred_val = data[pred_id][-1]

                if known_val != pred_val:
                    print(f'Data is incorrect for {identifier}: {pred_id}')
                    DebugMode.save_data()


    @staticmethod
    def save_data():
        
        image_path = os.path.join(DebugMode.glycan_folder, DebugMode.curr_image)
        png_image = image_path + '.png'
        svg_image = image_path + '.svg'
        text_data = image_path + '_map.txt'

        destination_path = os.path.join(DebugMode.current_folder, DebugMode.curr_image)
        os.makedirs(destination_path, exist_ok=True)

        shutil.copy(png_image, destination_path)
        shutil.copy(svg_image, destination_path)
        shutil.copy(text_data, destination_path)



