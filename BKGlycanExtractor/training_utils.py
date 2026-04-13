import os
import shutil
import tempfile
import atexit
from BKGlycanExtractor import Image_Manager, GlycanExtractorPipeline

def _remove_tempdir(tempdir):
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

def build_training(*, finder, images, outname, test_frac=0.0):

    folder_name = tempfile.mkdtemp(prefix=".tmpdir", dir=os.getcwd())
    atexit.register(_remove_tempdir, folder_name)
    
    if test_frac > 0.0:
        folder_name_test = tempfile.mkdtemp(prefix=".tmpdir", dir=os.getcwd())
        atexit.register(_remove_tempdir, folder_name_test)

    pipeline = finder.finder_pipeline()

    train_images, test_images = images.train_test_split(test_frac)
    if test_frac > 0:
        assert len(test_images) > 0
    else:
        assert len(test_images) == 0
    for image_path in train_images + test_images:
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]
        if image_path in test_images:
            the_folder_name = folder_name_test
        else:
            the_folder_name = folder_name
        label_file_path = os.path.join(the_folder_name, base_filename + ".txt")

        result, glycan_semantics = pipeline.run_evaluation(image_path, boxesonly=True)

        with open(label_file_path, 'w') as f:
            for b in result:
                b.set_image_dimensions(
                    image_width=glycan_semantics.width(),
                    image_height=glycan_semantics.height()
                )
                classid = b.get('classid')
                x, y, w, h = b.center_relative()
                f.write(f"{classid} {x} {y} {w} {h}\n")

        shutil.copy(image_path, the_folder_name)

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

    