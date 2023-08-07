import argparse
import pickle as pkl
from itertools import product
from pathlib import Path

import cv2
from background_subtraction import BackgroundSubtraction
from raw_dataset import PRWRawDataset
from tqdm import tqdm


def area_fn(area):
    return area >= 500


def compute_intersection(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    area_intersection = dx * dy if dx >= 0 and dy >= 0 else 0
    return area_intersection


def compute_iou(bb1, bb2):
    area_intersection = compute_intersection(bb1, bb2)
    area1, area2 = bb1[2] * bb1[3], bb2[2] * bb2[3]
    return area_intersection / (area1 + area2 - area_intersection)


def save_bad_bboxes(root_path: str):
    root_path = Path(root_path)

    data = PRWRawDataset(root_path, "train", shuffle=False)
    data.annotations = data.annotations + PRWRawDataset(root_path, "valid", shuffle=False).annotations

    widths = []
    heights = []
    worst_bboxes = []

    k = 0
    current_video_id = (0, 0)
    num_background_samples = 20
    pbar = tqdm(total=len(data))
    while k < len(data):
        if data.annotations[k]["video_id"] > current_video_id:
            current_video_id = data.annotations[k]["video_id"]
            background_samples = []
            for j in range(k, k + num_background_samples):
                background_samples.append(cv2.cvtColor(data[j][0], cv2.COLOR_RGB2GRAY))
                pbar.update()
            k += num_background_samples
            bkg_subtraction = BackgroundSubtraction(
                background_samples,
                area_filter_fn=area_fn,
                beta=0.2,
                handle_light_changes=True,
            )
        frame, target_bboxes, _ = data[k]
        height, width = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        candidate_bboxes = bkg_subtraction.step(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        idx_iou_pairs = [(None, 0) for _ in range(len(target_bboxes))]
        for bb1, bb2 in product(enumerate(candidate_bboxes), enumerate(target_bboxes)):
            i, bb1 = bb1
            j, bb2 = bb2
            iou = compute_iou(bb1, bb2)
            idx_iou_pairs[j] = (i, iou) if iou > idx_iou_pairs[j][1] else idx_iou_pairs[j]

        best_candidates_idx = []
        for i in range(len(idx_iou_pairs)):
            if idx_iou_pairs[i][0] is not None:
                best_candidates_idx.append(idx_iou_pairs[i][0])
        best_candidates_idx = set(best_candidates_idx)
        words_candidates_idx = [i for i in range(len(candidate_bboxes)) if i not in best_candidates_idx]
        best_candidates_bboxes = [candidate_bboxes[i] for i in best_candidates_idx]
        worst_candidates_bboxes = [candidate_bboxes[i] for i in words_candidates_idx]

        # Filter worst boxes that completely enclose one or more target box
        new_worst_candidates_bboxes = []
        for bb1 in worst_candidates_bboxes:
            add = True
            for bb2 in target_bboxes:
                if compute_intersection(bb1, bb2) == bb2[2] * bb2[3]:
                    add = False
                    break
            if add:
                new_worst_candidates_bboxes.append(bb1)
        worst_candidates_bboxes = new_worst_candidates_bboxes

        worst_bboxes += worst_candidates_bboxes
        widths += [width for _ in range(len(worst_candidates_bboxes))]
        heights += [height for _ in range(len(worst_candidates_bboxes))]
        k += 1
        pbar.update()

    worst_bboxes = [[bbox, h, w] for bbox, h, w in zip(worst_bboxes, heights, widths)]

    with open(root_path / "worst_bboxes.pkl", "wb") as f:
        pkl.dump(worst_bboxes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("root_path", help="Path of the source dataset")

    args = parser.parse_args()

    save_bad_bboxes(args.root_path)
