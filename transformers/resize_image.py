import cv2

from .pipeline import Pipeline

INTER_METHOD={
    "area":cv2.INTER_AREA, 
    "linear":cv2.INTER_LINEAR, 
    "cubic":cv2.INTER_CUBIC, 
    "nearest":cv2.INTER_NEAREST
    }

class ResizeImage(Pipeline):
    """Pipeline to resize an image."""
    def __init__(self, width=None, height=None, inter="area"):
        """
        Parameters
        ----------
        width : int, required if height is None
            New width of the image. If None, aspect ratio is calculated.
        height : int, required if width is None
            New height of the image. If None, aspect ratio is calculated.
        inter : str, optional
            Interpolation's method. Options: area, linear, cubic, nearest. 
        """
        self.width=width
        self.height=height

        try:
            self.inter=INTER_METHOD[inter]
        except:
            print("Invalid interpolation's method.")

        super(ResizeImage, self).__init__()

    def map(self,data):
        image=data["image"]
        if self.width is None and self.height is not None:
            r=self.height/image.shape[0]
            dim=(int(r*image.shape[1]),self.height)
        elif self.height is None and self.width is not None:
            r=self.width/image.shape[1]
            dim=(self.width,int(r*image.shape[0]))
        elif self.width is not None and self.height is not None:
            dim=(self.width, self.height)
        else:
            print("At least one dimension is required.")
            return
        
        resized_image=cv2.resize(image, dim, interpolation=self.inter)
        data["image"]=resized_image
        
        return data

        
            

        