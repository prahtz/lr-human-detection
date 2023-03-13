import os
import json
import h5py
import torchvision
import numpy as np
import argparse
from tqdm import tqdm

def get_coordinates(item_annotation):
    coordinates_list = []
    for element in item_annotation:
        _, coordinates = element['label'], element['coordinates']
        x, y, w, h = coordinates['x'], coordinates['y'], coordinates['width'], coordinates['height']
        coordinates['x'] = x - w / 2
        coordinates['y'] = y - h / 2
        coordinates = {k: round(v) for k, v in coordinates.items()}
        coordinates_list.append(coordinates)
    return coordinates_list
    
def create_negatives(root_path, out_path):
    splits = ['train', 'valid', 'test']
    with h5py.File(out_path, 'w') as f_dataset:
        for split in splits:
            item_count = 0
            split_grp = f_dataset.create_group(split)
            img_grp = split_grp.create_group('images')
            free_points_grp = split_grp.create_group('free_points')
            path = os.path.join(root_path, split)
            with open(os.path.join(path, '_annotations.createml.json'), 'r') as f_annotations:
                annotations = json.load(f_annotations)
            for i, item in enumerate(tqdm(annotations)):
                file_path, item_annotation = item['image'], item['annotations']
                coordinates_list = get_coordinates(item_annotation)
                
                image = torchvision.io.read_image(os.path.join(path, file_path))
                image = image.numpy()
                img_grp.create_dataset(file_path, data=image)

                im_h, im_w = image.shape[-2:]
                common_mask = np.ones((im_h, im_w), dtype=np.bool_)
                for coordinates in coordinates_list:
                    x, y, w, h = coordinates.values()
                    common_mask[y:y+h, x:x+w] = False
            
                for coordinates in coordinates_list:
                    mask = common_mask.copy()
                    x, y, w, h = coordinates.values()
                    box_dim = max(w, h)
                    for coordinates in coordinates_list:
                        x, y, w, h = coordinates.values()
                        mask[-box_dim+1:, :] = False
                        mask[:, -box_dim+1:] = False
                        mask[max(0, y-box_dim+1):y+box_dim, max(0, x-box_dim+1):x+box_dim] = False
                    free_points = np.argwhere(mask).tolist()
                    if not free_points:
                        print('empty exists')
                        free_points = [[0, 0]]

                    free_points = np.array(free_points).astype(np.int16)
                    item_data = free_points_grp.create_dataset(f'item_{item_count}', data=free_points)
                    item_data.attrs['box_size'] = box_dim
                    item_data.attrs['image_path'] = file_path
                    item_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('root_path', help='Path of the source dataset')
    parser.add_argument('out_path', help='Destination path')
    
    args = parser.parse_args()

    create_negatives(args.root_path, args.out_path)