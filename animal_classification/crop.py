import os
import pandas as pd
import argparse
import numpy as np
import json
import random
import pickle
import cv2
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        default='zoohackathon_dataset/',
        type=str,
        help="folder with images"
    )
    parser.add_argument(
        "--output-folder",
        default='zoohackathon_test_crop',
        type=str,
    )
    parser.add_argument(
        "--input-file",
        default='zooCOCO_test.json',
        type=str,
    )
    return parser


def create_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

num = 0

def save(image, id, unique_id):
    folder_name = args.output_folder+"/"+str(id)
    create_dir(folder_name)
    cv2.imwrite(folder_name + '/' + \
                str(unique_id) + '.jpg', image)
    '''
    print(args.output_folder + '/' + \
                elem + '/' + \
                id[idx[i]] + '/' + \
                filename + '__' + \
                str(unique_id)+ '.jpg')
    '''


folder_name = ''


def crop(image, bbox, id, unique_id):
    light = image.copy()
    x, y, w, h = bbox
    im = light[y:y+h, x:x+w]
    save(im, id, unique_id)
    if random.random() < 0.5:
        return
    w = np.random.randint(10, 30)
    att = np.random.randn()/10 + 1
    h = int(w*att)
    height, width = light.shape[:2]
    y, x = np.random.randint(height), np.random.randint(width)
    im = light[y:min(y+h, height), x:min(x+w, width)]
    #print(light.shape, y, min(y+h, height), x, min(x+w, width) )
    save(im, 4, unique_id*1000000)

def id2filename(dict_ann, dict_im):
    dic = dict()
    for i, row in tqdm(dict_ann.iteritems()):
        for j, im in dict_im.iteritems():
            if row['image_id'] == im['id']:
                dic[row['image_id']] = im['file_name']
                break
    return dic

def main(args):
    global folder_name, process, pos, numb, col
    create_dir(args.output_folder)
    folder_name = args.output_folder
    with open(args.input_file) as json_data:
        data = json.load(json_data)
    del data["categories"]
    data['index'] = range(len(data['images']))
    df = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')
    dict_id2file = id2filename(df['annotations'], df['images'])
    with open('dict_id2file.pickle', 'wb') as f:
        pickle.dump(dict_id2file, f)
    #with open('dict_id2file.pickle', 'rb') as f:
    #    dict_id2file = pickle.load(f)
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        row = row["annotations"]
        filename = args.input_folder +'/'+ dict_id2file[row['image_id']].split("/")[-1]
        image = cv2.imread(filename)
        #image = image[:, :, 0]
        crop(image, row['bbox'], row['category_id'], row['id'])

if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)