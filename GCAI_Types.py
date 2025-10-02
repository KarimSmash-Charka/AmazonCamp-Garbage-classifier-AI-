from enum import Enum
from typing import TypedDict

class AssetType(Enum):
    DIRECTORY = "Directory"
    MODEL = "Model"
    POSTPROCESS = "Postprocess shared object"

class ImageInfo(TypedDict):
    """
        Custom type for storing image data along with other useful 
        information such as:
        - size (width and height)
        - format (RGB, BGR, grayscale etc.)
        - the actual image data, in bytes
    """
    width: int
    height: int
    format: str
    data: bytes