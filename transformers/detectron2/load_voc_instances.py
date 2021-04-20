
from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode

from ..pipeline import Pipeline
from ..utils.files import list_files_in_txt

CLASS_NAMES = [
    "licenseplate",
]

class LoadVOCInstance(Pipeline):
    """Pipeline task to load VOC annotations."""
    def __init__(self,dirname):
        """
        Parameters
        ----------
        dirname : str
            path to "annotations" and "images".
        """
        self.dirname=dirname

        super(LoadVOCInstance,self).__init__()

    def generator(self,split):
        """Yields the image content and annotations."""
        fileid=list_files_in_txt(self.dirname, split)
        while self.has_next():
            try:
                anno_file = os.path.join(self.dirname, "annotations", fileid + ".xml")
                jpeg_file = os.path.join(self.dirname, "images", fileid + ".jpg")

                tree = ET.parse(anno_file)

                data = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }
                instances = []

                for obj in tree.findall("object"):
                    cls = obj.find("name").text
                    bbox = obj.find("bndbox")
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    instances.append(
                        {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                    )
                data["annotations"] = instances
                
                if self.filter(data):
                    yield self.map(data)

            except StopIteration:
                return
