import os
import shutil
import tempfile
import atexit
import sys
from collections import defaultdict
from BKGlycanExtractor import Image_Manager, GlycanExtractorPipeline

def _remove_tempdir(tempdir):
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

def build_training(*, finder, images, outname, test_frac=0.0, split_seed=None, quiet=False):

    folder_name = tempfile.mkdtemp(prefix=".tmpdir", dir=os.getcwd())
    atexit.register(_remove_tempdir, folder_name)
    
    if test_frac > 0.0:
        folder_name_test = tempfile.mkdtemp(prefix=".tmpdir", dir=os.getcwd())
        atexit.register(_remove_tempdir, folder_name_test)

    pipeline = finder.finder_pipeline()

    train_images, test_images = images.train_test_split(test_frac, split_seed=split_seed, quiet=quiet)

    if test_frac > 0:
        assert len(test_images) > 0
    else:
        assert len(test_images) == 0

    if not quiet:
        if test_frac > 0:
            print("Training images:",len(train_images),file=sys.stderr)
            print("Testing images:",len(test_images),file=sys.stderr)
        else:
            print("Images:",images.count(),file=sys.stderr)

    test_class_freq = defaultdict(int)
    train_class_freq = defaultdict(int)
    test1_class_freq = defaultdict(int)
    train1_class_freq = defaultdict(int)
    train_empty_images = 0
    test_empty_images = 0
    for image_path in train_images + test_images:
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]
        if image_path in test_images:
            the_folder_name = folder_name_test
        else:
            the_folder_name = folder_name
        label_file_path = os.path.join(the_folder_name, base_filename + ".txt")

        result, glycan_semantics = pipeline.run_evaluation(image_path, boxesonly=True)
        classfreq = defaultdict(int)
        anyboxes = False    
        with open(label_file_path, 'w') as f:
            for b in result:
                b.set_image_dimensions(
                    image_width=glycan_semantics.width(),
                    image_height=glycan_semantics.height()
                )
                classid = b.get('classid')
                classfreq[b.get('classlabel')] += 1
                x, y, w, h = b.center_relative()
                f.write(f"{classid} {x} {y} {w} {h}\n")
                anyboxes = True
        if image_path in test_images:
            if not anyboxes:
                test_empty_images += 1
            for k,v in classfreq.items():
                test_class_freq[k] += v
                test1_class_freq[k] += 1
        else:
            if not anyboxes:
                train_empty_images += 1
            for k,v in classfreq.items():
                train_class_freq[k] += v
                train1_class_freq[k] += 1

        shutil.copy(image_path, the_folder_name)

    if not quiet:
        if test_frac > 0:
            print("Training empty images:",train_empty_images)
            print("Testing empty images:",test_empty_images)
            print("Training classes:",file=sys.stderr)
            for k,v in train_class_freq.items():
                print(f"  {k}: {v} boxes on {train1_class_freq[k]} images")
            print("Testing classes:",file=sys.stderr)
            for k,v in test_class_freq.items():
                print(f"  {k}: {v} boxes on {test1_class_freq[k]} images")
        else:
            print("Empty images:",train_empty_images)
            print("Classes:",file=sys.stderr)
            for k,v in train_class_freq.items():
                print(f"  {k}: {v} boxes on {train1_class_freq[k]} images")
        
    # only need these in training data
    labels_file = os.path.join(folder_name, 'classes.txt')
    finder.write_labels(labels_file)
    model_file = os.path.join(folder_name, 'model.ini')
    finder.write_model(model_file)

    if test_frac > 0.0:
        shutil.make_archive(outname+"-test", 'zip', folder_name_test)
        shutil.make_archive(outname+"-train", 'zip', folder_name)
    else:
        shutil.make_archive(outname, 'zip', folder_name)
    return

    