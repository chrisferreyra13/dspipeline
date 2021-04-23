import os
from typing import Counter
import cv2

from .pipeline import Pipeline


class SaveImage(Pipeline):
    """Pipeline task to save images. 
    
    When passing data through this pipe, an optional data setting can be set with key "filename" that changes the name of the outputfile"""
    counter = 1

    def __init__(self, path, printFileName = False ,image_ext="jpg"):
        self.path = path
        self.image_ext = image_ext
        self.printFileName = printFileName
        SaveImage.counter = 1 if len(os.listdir()) == 0 else len(os.listdir()) + 1 
        super(SaveImage, self).__init__()

    def map(self, data):
        filename = ""
        if 'filename' in data and self.printFileName:
            filename = data['filename']
        if type(data['image']) is list:
            for img in data['image']:   
                self.saveimg(img,filename)
        else:
            self.saveimg(data['image'],filename)
        return data

    def saveimg(self,img,filename = ""):
        os.makedirs(self.path, exist_ok=True)

        # Save image
        image_file = os.path.join(
            self.path, f"{filename}{SaveImage.counter:05d}.{self.image_ext}")
        cv2.imwrite(image_file, img)
        SaveImage.counter += 1
    