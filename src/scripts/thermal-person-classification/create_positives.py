import os
import json
import h5py
import torchvision
import argparse
from tqdm import tqdm

def create_positives(root_path, out_path):
    with h5py.File(out_path, 'w') as f_dataset:
        splits = ['train', 'valid', 'test']
        for split in splits:
            image_count = 0
            split_grp = f_dataset.create_group(split)
            path = os.path.join(root_path, split)
            with open(os.path.join(path, '_annotations.createml.json'), 'r') as f:
                annotations = json.load(f)

            for item in tqdm(annotations):
                file_path, item_annotation = item['image'], item['annotations']

                image = torchvision.io.read_image(os.path.join(path, file_path)).numpy()
                for element in item_annotation:
                    coordinates = element['coordinates']
                    x, y, w, h = coordinates['x'], coordinates['y'], coordinates['width'], coordinates['height']
                    x, y, w, h = round(x - w / 2), round(y - h / 2), round(w), round(h)
                    if w < h:
                        x = max(0, round(x - h / 2 + w / 2))
                        w = h
                    elif w > h:
                        y = max(0, round(y - w / 2 + h / 2))
                        h = w
                    split_grp.create_dataset(f'img_{image_count}', data=image[:, y:y+h,x: x+w])
                    image_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root_path', help='Path of the source dataset')
    parser.add_argument('out_path', help='Destination path')
    
    args = parser.parse_args()

    create_positives(args.root_path, args.out_path)