import argparse
import time

import cv2
import torch
import torch.nn.functional as F
from background_subtraction import BackgroundSubtraction
from tqdm import tqdm

from config.config import get_default_cfg
from datasets.prw import PRWRawDataset
from models.utils import load_model_and_transforms
from utils import CustomCocoBinaryAveragePrecision, compute_intersection_over_target_area


def area_fn(area):
    return area >= 50


def scale_gray_img(img, size=(640, 640)):
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(img, size=size)[0][0].numpy()


def scale_up_bbox(bbox, ratios):
    bbox = [b for b in bbox]
    bbox[0] = round(bbox[0] * ratios[1])
    bbox[1] = round(bbox[1] * ratios[0])
    bbox[2] = round(bbox[2] * ratios[1])
    bbox[3] = round(bbox[3] * ratios[0])
    return bbox


def evaluate_with_bs(args):
    cfg_path = args.cfg_path

    # Load model
    cfg = get_default_cfg()
    cfg.merge_from_file(cfg_path)

    model, _, eval_transforms_fn = load_model_and_transforms(
        model_args=cfg.model,
        checkpoint_path=cfg.test.checkpoint_path,
    )
    preds = []
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
            bkg_subtraction = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            for j in range(k, k + num_background_samples):
                img = data[j][0]
                if j == k:
                    height, width = img.shape[:2]
                    ratios = (height / target_shape[0], width / target_shape[1])
                img = cv2.cvtColor(data[j][0], cv2.COLOR_RGB2GRAY)
                img = scale_gray_img(img, size=target_shape)
                background_samples.append(img)
                pbar.update()
            k += num_background_samples
            bkg_subtraction = BackgroundSubtraction(
                background_samples,
                threshold=10,
                area_filter_fn=area_fn,
                opening_k_shape=(3, 3),
                closing_k_shape=(9, 9),
                beta=0.00,
                handle_light_changes=True,
            )
        frame, target_bboxes, _ = data[k]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        start_bkg_time = time.time()

        candidate_bboxes = bkg_subtraction.step(scale_gray_img(gray_frame, size=target_shape))
        end_bkg_time = time.time()
        candidate_bboxes = [scale_up_bbox(bbox, ratios) for bbox in candidate_bboxes]
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

        end_total_time = time.time()
        bkg_times.append(end_bkg_time - start_bkg_time)
        total_times.append(end_total_time - start_bkg_time)
        model_times.append(end_model_time - start_model_time)
        preds.append({"boxes": candidate_bboxes, "scores": scores})
        targets.append(target_bboxes)
        pbar.update()
        k += 1

    map_fn = CustomCocoBinaryAveragePrecision(compute_intersection_over_target_area)
    aps = map_fn(preds=preds, targets=targets)
    print("Average Precisions:", aps)
    print("Average backgroud subtraction time:", sum(bkg_times) / len(bkg_times))
    print("Average model inference time:", sum(model_times) / len(model_times))
    print("Average total time:", sum(total_times) / len(total_times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("cfg_path", help="Path of the YAML configuration file.")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)

    args = parser.parse_args()

    evaluate_with_bs(args)
