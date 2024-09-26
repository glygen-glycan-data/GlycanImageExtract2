'''

'''
import cv2
import os
import numpy as np


class Image_Semantics:
    '''
    image can be any of the following formats: img_path, png, cv2, pdf
    '''

    def __init__(self):
        self.semantics = {
            'glycans': []
        }

    def create_image_semantics(self, semantics):
        self.semantics['image_path'] = semantics.image_path
        self.semantics['file_name'] = semantics.file_name
        self.semantics['image'] = semantics.image
        self.semantics['width'] = semantics.width
        self.semantics['height'] = semantics.height

    def glycans(self):
        return self.semantics['glycans']

    # semantics = {
    #     'glycans': []
    # }

    # @classmethod
    # def create_image_semantics(cls,semantics):
    #     # storing common attributes in the base class (class-level)
    #     cls.semantics['image_path'] = semantics.image_path
    #     cls.semantics['image'] = semantics.image
    #     cls.semantics['width'] = semantics.width
    #     cls.semantics['height'] = semantics.height

    # @classmethod
    # def glycans(cls):
    #     return cls.semantics['glycans']


class Figure_Semantics(Image_Semantics):
    def __init__(self,image):
        super().__init__()
        self.image_path = os.path.abspath(image)
        self.file_name = os.path.basename(self.image_path) 
        self.image = self.format_image(self.image_path)
        self.height, self.width, _ = self.image.shape
        self.create_image_semantics(self)   # stores all the data in the base class


    def format_image(self,image):
        if isinstance(image, str):
            if (image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg')):
                glycan_image = cv2.imread(image)
                return glycan_image

            elif image.endswith('.pdf'):
                # need to add more details here
                return pdf

        elif isinstance(image, np.ndarray):
            return image

        else:
            return None


    def set_glycans(self,idx,image_path):
        image = cv2.imread(os.path.abspath(image_path))
        glycan_obj = {
                'id': idx,
                'image': image,
                'monos': []
            }
        
        self.semantics['glycans'].append(glycan_obj)


class Glycan_Semantics(Image_Semantics):
    def __init__(self):
        super().__init__()  
        self.mono_list = {}
        self.linked_monos = {}
        self.boxes = []

    def monosaccharides(self):
        for gly_obj in self.glycans():
            self.mono_list[gly_obj['id']] = [mono['type'] for mono in gly_obj['monos']]
        return self.mono_list


    def mono_links(self):
        for gly_obj in self.glycans():
            self.linked_monos[gly_obj['id']] = [{mono['id']: mono['links']} for mono in gly_obj['monos']]
        return self.linked_monos

    
    def boxes(self):
        for gly_obj in self.glycans():
            self.boxes = [mono['box'] for mono in gly_obj['monos']]
        return self.boxes


class File_Semantics(Image_Semantics):
    def __init__(self,file_path):
        super().__init__()
        self.image_path = os.path.abspath(file_path)
        self.file_name = os.path.basename(self.image_path)
        self.image = None
        self.height, self.width, _ = None,None,None
        self.create_image_semantics(self)   # stores all the data in the base class




