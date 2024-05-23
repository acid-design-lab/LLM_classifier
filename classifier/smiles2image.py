import os
from pathlib import Path

from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from slugify import slugify
import PIL.Image

class Smiles2ImageConverter:
    def __init__(self, save_path: Path):
        self.__save_path = save_path
        img_size = os.environ.get("S2I_IMAGE_SIZE")
        if img_size is None:
            raise ValueError("--experimental-s2i requires S2I_IMAGE_SIZE to be set.")

        try:
            self.__img_w, self.__img_h = map(int, img_size.split(","))
        except ValueError:
            raise ValueError(f"S2I_IMAGE_SIZE={img_size} is not a valid integer")

    def convert(self, content: str):
        d2d = Draw.MolDraw2DCairo(self.__img_w, self.__img_h)
        r = AllChem.ReactionFromSmarts(content, useSmiles=True)
        d2d.DrawReaction(r)
        img = d2d.GetDrawingText()
        path = self.__save_path / (slugify(content) + ".png")
        open(path, "wb").write(img)
        return path
