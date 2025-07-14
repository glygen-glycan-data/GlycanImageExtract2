
# used to create a training and validation set - random.seed(42) - is used for consistent results

import os
import random
import argparse

def make_splits(image_dir, train_txt, val_txt, split_ratio=0.8):
    # Set a fixed seed for consistent splits every run
    random.seed(42)

    # Get all .png files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # Shuffle and split
    random.shuffle(base_names)
    split_index = int(len(base_names) * split_ratio)
    train_files = base_names[:split_index]
    val_files = base_names[split_index:]

    # Write train.txt
    with open(train_txt, 'w') as f:
        for name in train_files:
            f.write(os.path.join(image_dir, f"{name}.png") + '\n')

    # Write val.txt
    with open(val_txt, 'w') as f:
        for name in val_files:
            f.write(os.path.join(image_dir, f"{name}.png") + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Path to images")
    parser.add_argument("--train_txt", required=True, help="Output path for train.txt")
    parser.add_argument("--val_txt", required=True, help="Output path for val.txt")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train split ratio")
    args = parser.parse_args()

    make_splits(args.image_dir, args.train_txt, args.val_txt, args.split_ratio)
