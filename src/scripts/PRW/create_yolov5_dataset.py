import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets.prw import PRWRawDataset


def create_yolov5_dataset(root_path, out_path):
    root_path, out_path = Path(root_path), Path(out_path)
    images_path, labels_path = out_path / "images/", out_path / "labels/"
    for split in PRWRawDataset.__ALLOWED_SPLITS__:
        images_split_path, labels_split_path = images_path / split, labels_path / split
        images_split_path.mkdir(exist_ok=True, parents=True), labels_split_path.mkdir(exist_ok=True, parents=True)
        raw_dataset = PRWRawDataset(root_path=root_path, split=split, shuffle=False)
        annotations = raw_dataset.annotations

        for i in tqdm(range(len(raw_dataset))):
            img, bboxes, _ = raw_dataset[i]
            img_name, img_stem = annotations[i]["img_path"].name, annotations[i]["img_path"].stem
            img_heigth, img_width = img.shape[:2]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255
            assert img.shape[1] == 3
            img = F.interpolate(img, size=(640, 640), mode="bilinear")[0]
            new_img_path = images_split_path / img_name
            if not new_img_path.exists():
                save_image(img, new_img_path)
                with open(labels_split_path / (img_stem + ".txt"), "w") as f:
                    for box in bboxes:
                        box = [float(max(0, b)) for b in box]
                        x, y, w, h = box
                        x_center, y_center = x + w / 2, y + h / 2
                        x_center, y_center = x_center / img_width, y_center / img_heigth
                        w, h = w / img_width, h / img_heigth
                        f.write(f"0 {x_center} {y_center} {w} {h}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("root_path", help="Path of the source dataset")
    parser.add_argument("out_path", help="Destination path")

    args = parser.parse_args()

    create_yolov5_dataset(args.root_path, args.out_path)
