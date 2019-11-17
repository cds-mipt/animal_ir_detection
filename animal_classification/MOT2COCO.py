import glob
import pandas as pd
import json
import numpy as np
import random
import os
from tqdm import tqdm
import argparse
from PIL import Image

PRE_DEFINED_CATEGORIES = {"animals":3, 'unknown':2, 'human':1}

def build_parser():
    parser = argparse.ArgumentParser("convert whole MOT dataset to coco .json format")
    parser.add_argument("--dataset-dir",
                        type=str,
                        default='/home/z_andrei/zoohackathon/zoohackathon_dataset',
                        help='MOT dataset folder')
    parser.add_argument("--annot-dir",
                        type=str,
                        default='/home/z_andrei/zoohackathon/zoohackathon_train_annot',
                        help='dir with annotations')
    parser.add_argument("--output",
                        type=str,
                        default="ZOOCOCO",
                        help="json output file name")
    parser.add_argument("--input-folder",
                        type=str,
                        default='',
                        help="input folder name for shape")
    return parser

def get_image_list_dict(img_id):
    image_list_dict = list()
    id = 0
    name2id = dict()
    for i, im in enumerate(img_id):
        if i%3 != 0:
            continue
        file_name = im.split("/")[-1]
        with Image.open(im) as img:
            width, height = img.size
        name2id[file_name.split(".")[0].replace("_", "")] = id
        image_list_dict.append({'file_name':file_name, 'id':id, "width":width, "height":height})
        id += 1
    return image_list_dict, name2id

def get_annot_list(id_dict):
    annot_list = list()
    use = list()
    csv_list = sorted(glob.glob(args.annot_dir + '/*.csv'))
    id = 1
    for csv in tqdm(csv_list):
        df = pd.read_csv(csv, delimiter=",")
        for index, row in df.iterrows():
            frame_id, object_id, x, y, w, h, category_id, species, occ, noise = row
            image_id = csv.split("/")[-1].split(".")[0].replace("_", "") + str(frame_id).zfill(10)
            area = w * h
            if species == -1:
                category_id = 2
            elif category_id == 0:
                category_id = 3
            if image_id not in id_dict:
                continue
            if noise == 1:
                continue
            annot_list.append({"image_id": id_dict[image_id],
                               "area": area,
                               "bbox": [x, y, w, h],
                               "id": id,
                               "category_id": category_id,
                               "iscrowd": 0})
            id += 1
            use.append(id_dict[image_id])
    return annot_list, use

def clear(image_list_dict, use):
    old2newimageidx = dict()
    for i in range(len(image_list_dict)-1,-1,-1):
        if image_list_dict[i]['id'] not in use:
            del image_list_dict[i]
    for i, image in enumerate(image_list_dict):
        old2newimageidx[image['id']] = i
        image['id'] = i
    return old2newimageidx

def renew_annotations(annot_list, old2newimageidx):
    for annotation in annot_list:
        annotation['image_id'] = old2newimageidx[annotation['image_id']]


def get_images_and_annot():
    img_list = glob.glob(args.dataset_dir + "/*.jpg", recursive=True)
    img_list = sorted(img_list)
    n = int(len(img_list)*0.8)
    random.shuffle(img_list)
    img_id_train = img_list[:n]
    img_id_test = img_list[n:]
    image_list_dict_train, name2id_train = get_image_list_dict(img_id_train)
    image_list_dict_test,  name2id_test  = get_image_list_dict(img_id_test)
    annot_list_train, use_train = get_annot_list(name2id_train)
    annot_list_test, use_test = get_annot_list(name2id_test)
    old2newimageidx_train = clear(image_list_dict_train, use_train)
    old2newimageidx_test = clear(image_list_dict_test, use_test)
    renew_annotations(annot_list_train, old2newimageidx_train)
    renew_annotations(annot_list_test, old2newimageidx_test)
    return image_list_dict_train, image_list_dict_test, annot_list_train, annot_list_test

def get_categories(dict_of_categ=PRE_DEFINED_CATEGORIES):
    json_cat_list = []
    for cate, cid in dict_of_categ.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_cat_list.append(cat)
    return json_cat_list

def dump(images, annotations, name):
    categories = get_categories()
    json_dict = {'images': images, 'annotations': annotations, 'categories': categories}
    json_fp = open(args.output + "_" + name + ".json", 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def main(args):
    images_train, images_test, annotations_train, annotations_test = get_images_and_annot()
    dump(images_train, annotations_train, "train_ready")
    dump(images_test, annotations_test, "test_ready")

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
    #annotations 1, images 0