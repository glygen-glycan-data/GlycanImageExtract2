"""
You have an ImageFile
imagefile = “fjdkalfjlafjla.png”

# call pipeline to read the configs file
pipeline = configs.get("BestExtractor")

# gf = SingleGlycan(crop=20)
# mf = YOLOMonoFinder(weights=”yolomonorandom.wts”)
# rf = YOLORedEndFinder(weights=”....”)
# lf = YOLOLinkFinder(weights=”....”)
# gc = SemanticGlycanConstructor()


# run the pipeline
obj = pipeline.run(imagefile) # in png format

# so take the image file in png format and convert it to the required 
# format before passing it into the other modules

# for obj in gf.find_objects(imagefile)['glycans']:
    # mf.find_objects(obj)
    # lf.find_objects(obj)
    # rf.find_objects(obj)
    # gc.make_glycan(obj)

"""
# Note need a way to pass obj everywhere without directly passing it


import BKGlycanExtractor.glycanfinding as gf 
import BKGlycanExtractor.monosaccharideid as mf
import BKGlycanExtractor.glycanconnections as lf
import BKGlycanExtractor.rootmonofinding as rf
import cv2

# image file
png_image = "/home/nmathias/GlycanImageExtract2/glycans/right_root.png"
image = cv2.imread(png_image)



glycan_config_dict = {
    "weights": "/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/largerboxes_plusindividualglycans.weights",
    "config": "/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/coreyolo.cfg"}

# YOLOMonos
monos_config_dict = {
    'color_range': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/colors_range.txt',
    'weights': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/yolov3_monos_random.weights',
    'config': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/monos2.cfg'
    }


link_config_dict = {
    'color_range': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/colors_range.txt',
}

root_config_dict = {
    'weights': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/yolov3_rootmono.weights',
    'config': '/home/nmathias/GlycanImageExtract2/BKGlycanExtractor/config/monos2.cfg'
}




# initialize all the finders
glycanfinder = gf.YOLOGlycanFinder(glycan_config_dict) # create skeleton in the base class 
monosfinder = mf.YOLOMonos(monos_config_dict)
linkfinder = lf.ConnectYOLO(link_config_dict)
rootfinder = rf.DefaultOrientationRootFinder()



# call functions
# make this a class

# class Image_Semantics: base class

# class Figure_Semantics:
# dimensions, file name, glycan_semantics_iterator, etc

# class Glycan_Semantics:
# add_monosacc, add_mono_link(), mono_links() --> getters/setters

# figure_semantics = Figure_Semantics(image)
# image_semantics = glycanfinder.create_image_semantics(figure_semantics) # image_semantics will store the obj data, image can be a file_name, cv2, png and everything else that all the classes return back

# does this mean that: figure_semantics has all data and keep storing stuff and image_semantics has the object_data which is well formatted?

glycanfinder.find_objects(figure_semantics)
# glycanfinder.find_objects(image)

# print("\nimage_semantics--->>>:",figure_semantics.keys())

# composition_dict = {}
# for gly_obj in figure_semantics.glycans():
#     print("\nobj",gly_obj.keys())
#     monosfinder.find_objects(gly_obj)
#     linkfinder.find_objects(gly_obj)
#     # gly_obj = rootfinder.find_objects(gly_obj)
    
#     # print("\nmono_info",gly_obj)

#     # for mono in mono_info['monos']:
#     #     composition_dict[mono['type']] = composition_dict.get(mono['type'],0) + 1

#     # print("\ncomposition_dict",composition_dict)
