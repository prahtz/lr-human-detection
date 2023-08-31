import argparse

import h5py
from tqdm import tqdm

from datasets.prw import PRWRawDataset


def create_positives(root_path):
    with h5py.File(root_path / "positives.h5", "w") as f_dataset:
        splits = ["train", "valid", "test"]
        for split in splits:
            image_count = 0
            split_grp = f_dataset.create_group(split)
            data = PRWRawDataset(root_path, split=split)
            for i in tqdm(range(len(data))):
                image, bboxes, _ = data[i]
                for bbox in bboxes:
                    x, y, w, h = bbox
                    if w < h:
                        x = max(0, round(x - h / 2 + w / 2))
                        w = h
                    elif w > h:
                        y = max(0, round(y - w / 2 + h / 2))
                        h = w
                    split_grp.create_dataset(f"img_{image_count}", data=image[y : y + h, x : x + w, :])
                    image_count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("root_path", help="Path of the source dataset")

    args = parser.parse_args()

    create_positives(args.root_path)
