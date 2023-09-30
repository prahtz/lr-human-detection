import argparse
import time

import numpy as np
import yolov5
from tqdm import tqdm

from datasets.prw import PRWRawDataset
from utils import CustomCocoBinaryAveragePrecision, compute_intersection_over_target_area, compute_iou


def evaluate_yolo(args):
    model_source = args.model_source
    data_path = args.data_path

    map_fn = CustomCocoBinaryAveragePrecision(compute_intersection_over_target_area)
    map_fn2 = CustomCocoBinaryAveragePrecision(compute_iou)
    model = yolov5.load(model_source)
    data = PRWRawDataset(data_path, "test", shuffle=False)

    times = []
    predictions, targets = [], []
    for d in tqdm(data):
        img, target_boxes, _ = d

        start = time.time()
        results = model(img)
        end = time.time()
        times.append(end - start)
        img = np.asarray(img, dtype=np.uint8)
        detection_results = results.xywh[0]
        scores, boxes = [], []
        for result in detection_results:
            x, y, w, h, confidence_score, predicted_class = result
            x, y = x - w / 2, y - h / 2
            if predicted_class == 0.0:  # person
                scores.append(confidence_score.item())
                boxes.append([max(0, x.item()), max(0, y.item()), w.item(), h.item()])
        predictions.append({"boxes": boxes, "scores": scores})
        targets.append(target_boxes)

    aps = map_fn(predictions, targets)
    aps2 = map_fn2(predictions, targets)
    print("Average Precisions:", aps)
    print("Average Precisions (IoU):", aps2)
    print("Average time for each image:", sum(times) / len(times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_source", help="Name of the a YOLOv5 model OR path of a .pt YOLOv5 checkpoint")
    parser.add_argument("data_path", help="Path of the PRW dataset")
    parser.add_argument("--random-seed", help="Manual random seed", default=42, type=int)

    args = parser.parse_args()

    evaluate_yolo(args)
