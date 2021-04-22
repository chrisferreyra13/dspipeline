import os
from typing import Counter
import cv2

from .pipeline import Pipeline


class SaveImage(Pipeline):
    """Pipeline task to save images."""
    counter = 1

    def __init__(self, path, image_ext="jpg"):
        self.path = path
        self.image_ext = image_ext

        super(SaveImage, self).__init__()

    def map(self, data):

        if type(data['image']) is list:
            for img in data['image']:   
                self.saveimg(img)
        else:
            self.saveimg(data['image'])
        return data

    def saveimg(self,img):
        os.makedirs(self.path, exist_ok=True)

        # Save image
        image_file = os.path.join(
            self.path, f"{SaveImage.counter:05d}.{self.image_ext}")
        cv2.imwrite(image_file, img)
        SaveImage.counter += 1