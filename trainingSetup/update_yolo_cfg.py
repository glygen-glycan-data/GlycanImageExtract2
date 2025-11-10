#!/usr/bin/env python3
import sys
import re
import argparse

def update_globals(text, updates):
    return re.sub(
        r"^(?P<key>{})\s*=\s*.*$".format("|".join(map(re.escape, updates.keys()))),
        lambda m: f"{m.group('key')}={updates[m.group('key')]}",
        text,
        flags=re.MULTILINE
    )

# checks for the section [yolo] and makes changes to respective parameters like classes.
# Regex condition is to detect the end of the section - so that we dont accidently change classes parameter in any other section.
def update_block_params(text, section, param_dict):
    lines = text.splitlines()
    result = []
    inside_section = False
    updated_params = set()
    
    for line in lines:
        stripped = line.strip()
        if stripped == f"[{section}]":
            inside_section = True
            result.append(line)
            updated_params.clear()
        # regex to mark the end of the [yolo] section that we made changes on
        elif inside_section and re.search(r"^\[\w+\]$", stripped):
            inside_section = False
            ifany = False
            for key in param_dict:
                if key not in updated_params and param_dict.get(key):
                    result.append(f"{key}={param_dict[key]}")
                    ifany = True
            if ifany:
                result.append("")
            result.append(line)
        elif inside_section:
            key = stripped.split("=")[0].strip()
            if param_dict.get(key) and key not in updated_params:
                result.append(f"{key}={param_dict[key]}")
                updated_params.add(key)
            else:
                result.append(line)
        else:
            result.append(line)
    if inside_section:
        for key in param_dict:
            if key not in updated_params and param_dict.get(key):
                result.append(f"{key}={param_dict[key]}")
    result.append("\n")
    return "\n".join(result)

def update_filters_before_yolo(text, filters_val):
    lines = text.splitlines()
    result = []
    i = 0

    while i < len(lines):
        if lines[i].strip() == "[convolutional]":
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("["):
                j += 1
            if j < len(lines) and lines[j].strip() == "[yolo]":
                for k in range(i+1, j):
                    if lines[k].strip().startswith("filters="):
                        lines[k] = f"filters={filters_val}"
                        break
        result.append(lines[i])
        i += 1

    return "\n".join(result)

def main(args):

    cfg_path = args.yolo_config
    classes = args.classes
    filters = (classes + 5) * 3

    with open(cfg_path, 'r') as f:
        text = f.read()

    # max_batch = no.of classes * 2000, but max_bathces should never be lower than 6000
    if args.max_batches is not None:
        max_batches = args.max_batches
    else:
        max_batches = max(classes*2000,6000)

    # Compute steps at 80% and 90% of max_batches
    step1 = int(max_batches * 0.8)
    step2 = int(max_batches * 0.9)
    steps = f"{step1},{step2}"
    
    text = update_globals(text, {"batch": args.batch, "subdivisions": args.subdivisions, "max_batches": max_batches, "steps": steps, "height": args.height, "width": args.width, "learning_rate": args.learning_rate})

    # print("text",args.subdivisions)
    
    param_updates = {"classes": classes, "nms_kind": args.nms_kind, "beta_nms": args.beta_nms}
    text = update_block_params(text, "yolo", param_updates)
    text = update_filters_before_yolo(text, filters)

    # sys.stdout.write(text)
    with open(cfg_path, 'w') as f:
        f.write(text)
    # print(f"Updated '{cfg_path}' for {classes} classes and {filters} filters.")

if __name__ == "__main__":

    # print("sys.argv:", sys.argv)

    parser = argparse.ArgumentParser()

    # config_file_path and no.of classes are not user inputs - hence they wont be present in the help text
    parser.add_argument("--yolo_config", type=str, required=True, help=argparse.SUPPRESS)
    parser.add_argument("--classes", type=int, required=True, help=argparse.SUPPRESS)

    parser.add_argument("--max_batches", type=int, default=None, help="Number of training iterations (batches).")

    parser.add_argument("--batch", type=int, default=64, help="Number of training images per iteration (batch size).")

    parser.add_argument("--subdivisions", type=int, default=16, 
                        help='''Splits the batch into smaller groups to reduce GPU memory usage. 
                        Each subdivision loads (batch / subdivisions) images at a time. 
                        Increase to 32 or 64 if you encounter out-of-memory errors.''')

    parser.add_argument("--nms_kind", type=str, default=None, help="NMS kind (default, greedynms, diounms, cornersnms).")
    parser.add_argument("--beta_nms", type=float, default=None, help="NMS value.")

    parser.add_argument("--height", type=int, default=416, help="Height of input training images (in pixels).")

    parser.add_argument("--width", type=int, default=416, help="Width of input training images (in pixels).")

    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help='''Learning rate for training. The default (0.001) is commonly used for YOLO models. 
                        If you plan to experiment, consider starting with slightly lower rates like 0.0005 or 0.0001,as YOLO models 
                        are typically sensitive to large learning rates and may diverge during training.''')


    args = parser.parse_args()

    main(args)
