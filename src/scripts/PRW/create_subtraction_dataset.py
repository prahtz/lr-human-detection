import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from src.datasets.prw import PRWRawDataset
from src.utils import BackgroundSubtractionForDetection, compute_intersection_over_target_area


def create_subtraction_dataset(root_path: str):
    root_path = Path(root_path)
    splits = ["train", "valid", "test"]

    num_background_samples = 20
    target_shape = (144, 192)

    with h5py.File(root_path / "subtraction_dataset.h5", "w") as f_dataset:
        for split in splits:
            pos_count, neg_count = 0, 0
            split_grp = f_dataset.create_group(split)
            pos_grp = split_grp.create_group("positives")
            neg_grp = split_grp.create_group("negatives")
            data = PRWRawDataset(root_path, split, shuffle=False)
            k = 0
            current_video_id = (0, 0)
            pbar = tqdm(total=len(data))
            while k < len(data):
                if data.annotations[k]["video_id"] > current_video_id:
                    current_video_id = data.annotations[k]["video_id"]
                    background_samples = []
                    for j in range(k, k + num_background_samples):
                        background_samples.append(data[j][0])
                        pbar.update()
                    bkg_subtraction = BackgroundSubtractionForDetection(
                        background_samples=background_samples,
                        target_shape=target_shape,
                    )
                frame, target_bboxes, _ = data[k]
                candidate_bboxes = bkg_subtraction.step(frame)

                positive_candidates_idx = []
                for bb1, bb2 in product(enumerate(candidate_bboxes), target_bboxes):
                    i, bb1 = bb1
                    iou = compute_intersection_over_target_area(bb1, bb2)
                    if iou >= 0.5:
                        positive_candidates_idx.append(i)
                positive_candidates_idx = set(positive_candidates_idx)
                negative_candidates_idx = [i for i in range(len(candidate_bboxes)) if i not in positive_candidates_idx]

                frame = frame.astype(np.uint8)
                for i in positive_candidates_idx:
                    x, y, w, h = [max(0, c) for c in candidate_bboxes[i]]
                    pos_grp.create_dataset(str(pos_count), data=frame[y : y + h, x : x + w, :])
                    pos_count += 1
                for i in negative_candidates_idx:
                    x, y, w, h = [max(0, c) for c in candidate_bboxes[i]]
                    neg_grp.create_dataset(str(neg_count), data=frame[y : y + h, x : x + w, :])
                    neg_count += 1
                k += 1
                pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("root_path", help="Path of the source dataset")

    args = parser.parse_args()

    create_subtraction_dataset(args.root_path)
