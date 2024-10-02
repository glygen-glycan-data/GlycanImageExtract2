import os

class Image_Manager:
    def __init__(self,glycan_folder,pattern):
        self.glob = [pattern_type.strip()[1:] if pattern_type.strip().startswith('*') else pattern_type.strip() for pattern_type in pattern.split(',')]
        self.images = self.get_images(glycan_folder)

    def __iter__(self):
        return iter(self.images)

    def get_images(self,glycan_folder):
        images = []

        if isinstance(glycan_folder,list):
            for image_file in glycan_folder:
                if os.path.is_file(image_file) and self.match_glob(image_file):
                    images.append(image_file)
        elif os.path.isdir(glycan_folder):
            for image_file in os.scandir(glycan_folder):
                if image_file.is_file() and self.match_glob(image_file):
                    images.append(image_file.path)
        return images


    def match_glob(self,image_file):
        return any(image_file.name.endswith(ext) for ext in self.glob)
