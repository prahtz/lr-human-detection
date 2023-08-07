import random
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset


class PRWRawDataset(Dataset):
    __ALLOWED_SPLITS__ = ["train", "valid", "test"]

    def __init__(self, root_path: str, split: str = "train", shuffle=True) -> None:
        assert split in self.__ALLOWED_SPLITS__

        self.root_path = Path(root_path)
        self.split = split
        self.annotations_dir_path = self.root_path / "annotations/"

        split_filename = "test" if split == "test" else "train"
        split_idxs = scipy.io.loadmat(self.root_path / f"frame_{split_filename}.mat")
        frame_names = sorted([img_idx[0][0] + ".jpg" for img_idx in split_idxs[f"img_index_{split_filename}"]])
        annotations_paths = [self.annotations_dir_path / (frame_name + ".mat") for frame_name in frame_names]

        self.annotations = []
        for frame_name, annotation_path in zip(frame_names, annotations_paths):
            annotation = scipy.io.loadmat(annotation_path)
            if "box_new" in annotation:
                bboxes = annotation["box_new"]
            elif "anno_file" in annotation:
                bboxes = annotation["anno_file"]
            else:
                bboxes = annotation["anno_previous"]
            bboxes = [list(map(round, bbox[1:])) for bbox in bboxes]
            camera = int(frame_name[1])
            sequence = int(frame_name[3])
            self.annotations.append(
                {"img_path": self.root_path / ("frames/" + frame_name), "bboxes": bboxes, "video_id": (camera, sequence)}
            )

        if split != "test":
            train_val_split_idx = int(len(self.annotations) * 0.8)
            if shuffle:
                random.seed(42)
                random.shuffle(self.annotations)
            self.annotations = self.annotations[:train_val_split_idx] if split == "train" else self.annotations[train_val_split_idx:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        img_path, bboxes = item["img_path"], item["bboxes"]
        img = np.array(Image.open(img_path))
        return img, bboxes, item["video_id"]
