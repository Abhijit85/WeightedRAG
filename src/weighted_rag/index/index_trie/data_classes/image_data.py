from dataclasses import dataclass
from data import Data


@dataclass
class ImageData(Data):
    text: str
    imageUrl: str
