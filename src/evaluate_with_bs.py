import argparse
import time

import torch
from tqdm import tqdm

from config.config import get_default_cfg
from datasets.prw import PRWRawDataset
from models.utils import load_model_and_transforms
from utils import BackgroundSubtractionForDetection, CustomCocoBinaryAveragePrecision, compute_intersection_over_target_area, compute_iou


def evaluate_with_bs(args):
    cfg_path = args.cfg_path

    # Load model
    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    model, _, eval_transforms_fn, _ = load_model_and_transforms(
        model_args=cfg.model,
        checkpoint_path=cfg.test.checkpoint_path,
    )
    model.eval()

    # Load dataset
    data = PRWRawDataset(cfg.dataset.root_path, "test", shuffle=False)
    num_background_samples = 20
    background_samples = [data[i][0] for i in range(num_background_samples)]

    current_video_id = (0, 0)
    num_background_samples = 20
    target_shape = (144, 192)  # (240, 320)  # (320, 568)  # (240, 426)

    k = 0
    pbar = tqdm(total=len(data))
    total_times, bkg_times, model_times = [], [], []
    preds, targets = [], []
    while k < len(data):
        if data.annotations[k]["video_id"] > current_video_id:
            current_video_id = data.annotations[k]["video_id"]
            background_samples = []
            for j in range(k, k + num_background_samples):
                img = data[j][0]
                background_samples.append(img)
                pbar.update()
            k += num_background_samples
            bkg_subtraction = BackgroundSubtractionForDetection(
                background_samples=background_samples,
                target_shape=target_shape,
            )
        frame, target_bboxes, _ = data[k]
        start_bkg_time = time.time()
        candidate_bboxes = bkg_subtraction.step(frame)
        end_bkg_time = time.time()
        scores = []
        with torch.no_grad():
            for bbox in candidate_bboxes:
                x, y, w, h = bbox
                x, y = max(0, x), max(0, y)
                candidate_frame_patch = frame[y : y + h, x : x + w]

                candidate_frame_patch = torch.from_numpy(candidate_frame_patch)
                candidate_frame_patch = torch.permute(candidate_frame_patch, (2, 0, 1)).unsqueeze(0)
                inputs = eval_transforms_fn(candidate_frame_patch)
                start_model_time = time.time()
                logits = model.model(inputs)
                end_model_time = time.time()

                score = torch.sigmoid(logits)
                scores.append(score)
                model_times.append(end_model_time - start_model_time)

        end_total_time = time.time()
        bkg_times.append(end_bkg_time - start_bkg_time)
        total_times.append(end_total_time - start_bkg_time)

        preds.append({"boxes": candidate_bboxes, "scores": scores})
        targets.append(target_bboxes)
        pbar.update()
        k += 1

    map_fn = CustomCocoBinaryAveragePrecision(compute_intersection_over_target_area)
    map_fn2 = CustomCocoBinaryAveragePrecision(compute_iou)
    aps = map_fn(preds=preds, targets=targets)
    aps2 = map_fn2(preds=preds, targets=targets)
    print("Average Precisions:", aps)
    print("Average Precisions (IoU):", aps2)
    print("Average backgroud subtraction time:", sum(bkg_times) / len(bkg_times))
    print("Average model inference time:", sum(model_times) / len(model_times))
    print("Average total time:", sum(total_times) / len(total_times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_path", help="Path of the YAML configuration file.")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)

    args = parser.parse_args()

    evaluate_with_bs(args)
