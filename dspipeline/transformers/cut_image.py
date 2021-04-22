import cv2

from .pipeline import Pipeline


class CutImage(Pipeline):
    """Pipeline to cut an image. data coming into pipeline must have coordinates key as a list of 2 points (p1, p2) set as a list of x,y
    """
    def map(self, data):
        
        image = data["image"]
        # remember that x and y are flipped.
        bboxes = data['coordinates']
        data["image"] = []
        for bbox in bboxes:
            resized_image = image[bbox['p1'][1]:bbox['p2'][1],
                                 bbox['p1'][0]:bbox['p2'][0]]
            data["image"].append(resized_image)
        return data
