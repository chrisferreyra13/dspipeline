import cv2
from .pipeline import Pipeline


class DisplayImage(Pipeline):
    def map(self, data):
        image = data['image']
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return data
