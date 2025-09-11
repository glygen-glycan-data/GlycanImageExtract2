import os
import glob
import pandas as pd
import shutil
import argparse

def create_yolo_labels(input_dir, output_dir, padding=0):

    # Load the TSV
    tsv_files = [f for f in os.listdir(input_dir) if f.endswith(".tsv")]
    if not tsv_files:
        raise FileNotFoundError(f"No TSV file found in {input_dir}")
    if len(tsv_files) > 1:
        raise ValueError(f"Expected only 1 TSV file in {input_dir}, found {len(tsv_files)}")

    tsv_path = os.path.join(input_dir, tsv_files[0])
    df = pd.read_csv(tsv_path, sep="\t")

    # Clean coordinates if needed
    # df['comment'] = df['comment'].str.strip()

    # Remove existing YOLO txt files
    # for txt_file in glob.glob(os.path.join(output_dir, "*.txt")):
    #     os.remove(txt_file)

    for image_path, group in df.groupby("figure_path"):
        image_name = os.path.basename(image_path)
        txt_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")

        # Copy image
        image_dst = os.path.join(output_dir, image_name)
        if not os.path.exists(image_dst):
            shutil.copy(image_path, image_dst)

        # Write YOLO labels
        with open(txt_path, "w") as f:
            print("txt",txt_path)
            for _, row in group.iterrows():
                class_id = 0

                # allow padding coords
                x1 = max(0, row["x1"] - padding)
                y1 = max(0, row["y1"] - padding)
                x2 = min(row["fig_width"],  row["x2"] + padding)
                y2 = min(row["fig_height"], row["y2"] + padding)

                # normalize to YOLO format
                x_center = (x1 + x2) / 2 / row["fig_width"]
                y_center = (y1 + y2) / 2 / row["fig_height"]
                width    = (x2 - x1) / row["fig_width"]
                height   = (y2 - y1) / row["fig_height"]

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO label files from a TSV")
    parser.add_argument("-i","--input_dir", required=True, help="Directory where images and supporting tsv file is stored")
    parser.add_argument("-o", "--output_dir", required=True, help="Output folder name to store YOLO training data")
    parser.add_argument("-p", "--padding", default=0, type=int, help="Pad the bounding boxes")


    args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)  # remove everything inside
    os.makedirs(output_dir, exist_ok=True)
    create_yolo_labels(args.input_path, output_dir, padding=args.padding)

    print("\Training data is ready...")
    print("Note: If padding was used, please make a note of it for your records. Default padding = 0")
