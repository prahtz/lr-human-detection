import argparse
import os
import pickle as pkl
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.datasets.prw import PRWRawDataset


def get_anchors_with_weights(root_path, k=7):
    root_path = Path(root_path)
    with open(root_path / "worst_bboxes.pkl", "rb") as f:
        worst_bboxes = pkl.load(f)
    bboxes = np.array([[item[0][2], item[0][3]] for item in worst_bboxes])
    heights = np.array([item[1] for item in worst_bboxes])
    widths = np.array([item[2] for item in worst_bboxes])
    scales = np.stack([widths / np.max(widths), heights / np.max(heights)], axis=-1)  # assuming parallel maximums
    rescaled_widths_heights = bboxes / scales
    areas = rescaled_widths_heights[:, 0] * rescaled_widths_heights[:, 1]
    perc_idx = np.argmin(abs(np.sort(areas) - np.percentile(areas, 98.0)))
    idx = np.argsort(areas)[:perc_idx]
    rescaled_widths_heights, heights, widths, scales = rescaled_widths_heights[idx], heights[idx], widths[idx], scales[idx]
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = model.fit_predict(rescaled_widths_heights)
    clusters = model.cluster_centers_
    unique_wh = np.unique(np.stack([widths, heights], axis=-1), axis=0)
    unique_scales = np.stack([unique_wh[:, 0] / np.max(unique_wh[:, 0]), unique_wh[:, 1] / np.max(unique_wh[:, 1])], axis=-1)
    anchors = {wh[0]: [anchor * scale for anchor in clusters] for scale, wh in zip(unique_scales, unique_wh)}
    anchors = {k: np.round(v).astype(np.int16) for k, v in anchors.items()}
    anchor_probs = np.bincount(labels) / len(labels)
    return anchors, anchor_probs


def create_negatives(args):
    root_path = Path(args.root_path)
    k = args.num_clusters

    anchors, anchors_weights = get_anchors_with_weights(root_path, k=k)
    splits = ["train", "valid", "test"]
    with h5py.File(root_path / "negatives.h5", "w") as f_dataset:
        anchors_grp = f_dataset.create_group("anchors")
        anchors_grp.attrs["anchors_weights"] = anchors_weights
        for k, v in anchors.items():
            anchors_grp.attrs[str(k)] = v
        for split in splits:
            split_grp = f_dataset.create_group(split)
            img_grp = split_grp.create_group("images")
            dataset = PRWRawDataset(root_path=root_path, split=split)
            for i in tqdm(range(len(dataset.annotations))):
                item = dataset.annotations[i]
                file_path, bboxes = item["img_path"], item["bboxes"]
                image_name = str(file_path).split("/")[-1]
                bboxes = np.array([[round(x) for x in bbox] for bbox in bboxes])

                image = np.array(Image.open(str(file_path)))
                image = image.astype(np.uint8)
                img_data = img_grp.create_dataset(str(image_name), data=image)
                img_data.attrs["bboxes"] = bboxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("root_path", help="Path of the source dataset")
    parser.add_argument("--num_clusters", help="Number of clusters for the KMeans algorithm", default=4, type=int)

    args = parser.parse_args()

    create_negatives(args)
