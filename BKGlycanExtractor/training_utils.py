import os
import shutil
import tempfile
import atexit
from BKGlycanExtractor import Image_Manager, GlycanExtractorPipeline

def _remove_tempdir(tempdir):
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

def build_training(*, config, finder_name, images, out_zip, boxpadding=None, label_type=None, exclude_pattern="*.annotated.*"):
    if not out_zip.endswith('.zip'):
        raise ValueError("Zip file filename must have .zip extension")
    if os.path.exists(out_zip):
        raise AssertionError(f"zip file {out_zip} exists")

    folder_name = tempfile.mkdtemp(prefix=".tmpdir", dir=os.getcwd())
    atexit.register(_remove_tempdir, folder_name)

    pipeline = GlycanExtractorPipeline()
    figure_finder = config.get_finder('SingleGlycanImage')
    pipeline.add_step('figure', figure_finder)

    finder = config.get_finder(finder_name)
    pipeline = finder.finder_pipeline(config)

    if boxpadding is not None:
        finder.set_param('boxpadding', boxpadding)

    # if a label_type was provided, that it will be picked from the semantics file and substituted as the
    # classlabel for the known boxes
    if label_type:
        finder.set_label(label_type)

    images_mgr = Image_Manager(images)
    if exclude_pattern:
        images_mgr.exclude(exclude_pattern)

    for image_path in images_mgr:
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]
        training_file_path = os.path.join(folder_name, base_filename + ".txt")

        result, glycan_semantics = pipeline.run_evaluation(image_path, boxesonly=True)

        with open(training_file_path, 'w') as f:
            for b in result:
                b.set_image_dimensions(
                    image_width=glycan_semantics.width(),
                    image_height=glycan_semantics.height()
                )
                classid = b.get('classid')
                x, y, w, h = b.center_relative()
                f.write(f"{classid} {x} {y} {w} {h}\n")

        shutil.copy(image_path, folder_name)

    labels_file = os.path.join(folder_name, 'classes.txt')
    finder.write_labels(labels_file)
    model_file = os.path.join(folder_name, 'model.ini')
    finder.write_model(finder_name, model_file)

    shutil.make_archive(out_zip.rsplit('.', 1)[0], 'zip', folder_name)
    return out_zip

    