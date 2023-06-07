import cv2
import albumentations as A
import numpy as np
import json
import imantics
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import CFG


class Data_Processor:
    def __init__(self, annotation_path: Path, image_dir_path: Path) -> None:
        self.annotation_path = annotation_path
        self.images_dir_path = image_dir_path
        self.annotations = None
        self.images = np.zeros(
            (CFG.N_IMAGES, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8
        )
        self.masks = np.zeros(
            (CFG.N_IMAGES, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), dtype=bool
        )

    def load_annotations(self) -> None:
        with open(self.annotation_path) as f:
            self.annotations = json.load(f)

    def create_train_image_tensor(self) -> None:
        map_id_filename = {}
        for i in range(len(self.annotations["images"])):
            map_id_filename[self.annotations["images"][i]["id"]] = self.annotations[
                "images"
            ][i]["file_name"]

        for image_id, image_filename in map_id_filename.items():
            image = cv2.imread((self.images_dir_path / image_filename).as_posix())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))

            self.images[image_id - 1] = image

    def create_masks(self) -> None:
        for i in range(len(self.annotations["annotations"])):
            image_id = self.annotations["annotations"][i]["image_id"]
            segmentation = self.annotations["annotations"][i]["segmentation"]

            mask = imantics.Polygons(segmentation).mask(*CFG.INPUT_IMAGE_SIZE).array
            mask = (
                cv2.resize(mask.astype(float), (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)) >= 0.5
            )

            self.masks[image_id - 1] = self.masks[image_id - 1] | mask


class Custom_Dataset(Dataset):
    def __init__(self, indexes, images, masks, transform=None, preprocess=None) -> None:
        self.indexes = indexes
        self.images = images
        self.masks = masks
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index) -> dict:
        _index = self.indexes[index]

        image = self.images[_index]
        mask = self.masks[_index]

        if self.transform:
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]

        if self.preprocess:
            image = self.preprocess(image)

        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2, 0, 1)

        mask = torch.tensor(mask, dtype=torch.float)
        mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask}


def data_split() -> list:
    pre_dataset = Data_Processor(
        annotation_path=Path("annotations/instances_default.json"),
        image_dir_path=Path("images"),
    )
    pre_dataset.load_annotations()
    pre_dataset.create_train_image_tensor()
    pre_dataset.create_masks()
    return pre_dataset.images, pre_dataset.masks
