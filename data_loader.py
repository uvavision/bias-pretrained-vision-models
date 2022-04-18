from __future__ import print_function
from __future__ import division
import subprocess
import numpy as np
import json, os, sys, random, pickle
import torchvision.datasets as dset
import os
from PIL import Image
import urllib
from collections import OrderedDict
import torchvision.datasets as dset
import urllib
from collections import OrderedDict
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import glob
from pathlib import Path
import pandas as pd
import torch
import time
import pickle
from collections import defaultdict
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from itertools import chain
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

print("Torch version:", torch.__version__)
torch.multiprocessing.set_sharing_strategy('file_system')

class Coco(Dataset):
    """Sets up the COCO 2017 dataset, expects pytorch CocoDetection object as Dataset

    __getitem__ implements standardization and returns image tensor and binary labels

    Attributes:
        dataset: Pytorch CocoDetection object
        img_del: Set of image ids to exclude that were included in the analysis set
        coco_object2id: Zero-indexing objects
    """
    def __init__(self, data, split='train'):
        self.dataset = data
        if split=='train':
            self.img_del = set([453695, 67837, 324442, 449244, 247547, 284688, 494473, 338911, 273188, 26165, 278660, 494466, 519555, 253109, 332543, 69087, 180095, 564043, 140613, 252956, 510230, 167510, 305634, 386542, 45471, 347177, 359106, 166356, 426519, 66427, 389948, 203846, 30307, 43353, 209746, 347612, 157209, 256082, 256082, 85529, 470697, 356116, 256475, 484354, 324971, 313481, 574376, 441795, 74176, 160828, 228505, 169602, 64750, 406723, 167962, 460997, 437540, 364256, 551575, 153864, 20179, 412062, 251750, 58647, 562557, 497466, 290700, 240274, 337527, 59282, 455427, 50410, 255479, 390646, 547830, 266026, 312385, 673, 145549, 140921, 138022, 128172, 129926, 126972, 122724, 120940, 115358, 119370, 436387, 311300, 551701, 261710, 555120, 357799, 296243, 226599, 22575, 547352, 158130, 103498, 453310, 400809, 466710, 28650, 273633, 113276, 452909, 524649, 12805, 274556, 32903, 395409, 196789, 1392, 4978, 5172, 21320, 21801, 22530, 15839, 19194, 13144, 24787, 333998, 330701, 329616, 400475, 504023, 362898, 72354, 443005, 145348, 305385, 422432, 343629, 133812, 423058, 171453, 318476, 41772, 341299, 93852, 560632, 112801])   
        else:
            self.img_del = set([580197, 568814, 562243, 560911, 17905, 21604, 22371, 4134, 4395, 5060, 8532, 9483, 565877, 561256, 559543, 554579, 551439, 547336, 546717, 512929, 358195, 381639, 8690, 9448, 27186, 66886, 162415, 252332, 571008, 309173, 489339, 498807, 107554, 127517, 164115, 278705, 102805, 213255, 442456, 566282, 357737, 543043, 41633, 65485, 141597, 198805, 89556, 263594, 293071, 231339, 456662, 527750, 425226, 215114, 273712, 282296, 802, 28452, 57238, 168593, 187271])
        self.coco_object2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

    def __len__(self):
        """Returns number of examples in dataset split 
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns a sample image and labels

        Args:
            idx: COCO index of image

        Returns:
            sample: Tuple of image tensor and binary labels of size (80,)
        """
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

        if torch.is_tensor(idx):
            idx = idx.tolist()

        start = time.time()
        images = self.dataset[idx][0]
        end = time.time()-start
        #with open('coco_times.txt', 'a') as the_file:
            #the_file.write(str(end)+'\n')
        image_input = torch.tensor(np.stack(images))
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]

        anns = self.dataset[idx][1]
        labels_binary = np.zeros(80)

        cats = []
        if len(anns)>0:
            img_id = anns[0]['image_id']
            if img_id in self.img_del:
                return self.__getitem__(idx+1)
        for ann in anns:
            x = ann['category_id']
            cats.append(self.coco_object2id[x])
        cats = np.array(cats)
        assert((cats >=0).all() and (cats <= 79).all())
        if len(cats) ==0:
            sample = (image_input, labels_binary)
        else: 
            labels_binary[cats] = 1
            sample = (image_input, labels_binary)
        return sample

K = TypeVar("K")
V = TypeVar("V")

def default_dict(pairs: Iterable[Tuple[K, V]]) -> DefaultDict[K, V]:
    mapping = defaultdict(list)
    for key, val in pairs:
        mapping[key].append(val)
    return mapping


def read_file(file_path: Path) -> str:
    with open(file_path) as text_file:
        return text_file.read()


def invert_mapping(mapping) -> Dict:
    """Naively convert key to value mapping to value to key dict."""
    return {v: k for k, v in mapping.items()}


def read_multiline(file_path: Path) -> List[str]:
    """Read, split by LF, and pop final empty line if present."""
    text = read_file(file_path)
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def read_csv(file_path: Path, discard_header: bool = True) -> List[List[str]]:
    lines = read_multiline(file_path)
    print(file_path, len(lines))
    # [x.split(',') for x in lines]
    table = map(methodcaller("split", ","), lines)
    if discard_header:
        next(table)
    return list(table)


def csv_to_dict(
    file_path: Path,
    key_col: int = 0,
    value_col: int = 1,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
) -> Union[Dict[str, str], DefaultDict[str, str]]:
    table = read_csv(file_path, discard_header)
    # ((line[key_col], line[value_col]) for line in table)
    pairs = map(itemgetter(key_col, value_col), table)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


def multicolumn_csv_to_dict(file_path: Path, key_cols: Sequence = (0,), value_cols: Optional[Sequence] = None, discard_header: bool = True, one_to_n_mapping: bool = False,) -> Union[Dict[str, Tuple[str]], DefaultDict[str, Tuple[str]]]:
    # TODO fix return type: keys can also be tuples...
    table = read_csv(file_path, discard_header)
    if not value_cols:
        value_cols = tuple(i for i in range(1, len(table[0])))
    # (tuple(line[i] for i in key_cols) for line in table)
    key_columns = map(itemgetter(*key_cols), table)
    value_columns = map(itemgetter(*value_cols), table)
    pairs = zip(key_columns, value_columns)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


BBOX_INDICES = {
    'ImageID': 0,
    'LabelName': 1,
    'Label': 2,
}


class OpenImages(Dataset):

    def __init__(
        self,
        root_folder: Path,
        split: str = "val",
        transform: Callable = None,
    ):
        """
        Object Detection dataset.
        [extended_summary]
        :param root_folder:
        :param split:
        :param transform:
        """
        super().__init__()
        self.split = split
        self.transform = transform
        images_folder = root_folder / split
        if split == "train":
            print(images_folder)
            all_images = images_folder.glob(r"*.jpg")
        else:
            all_images = images_folder.glob(r"*.jpg")
            with open(root_folder/"openimages_testset.txt", "rb") as fp:
                self.exclude = pickle.load(fp)
            self.exclude_ids = []
            for i in self.exclude:
                self.exclude_ids.append(i.strip())
            
        if split == 'train':
            bbox_csv_filepath = root_folder.joinpath(
                "anns", f"{split}-annotations-human-imagelabels-boxable.csv"
            )
        else:
            bbox_csv_filepath = root_folder.joinpath(
                "anns", f"validation-annotations-human-imagelabels-boxable.csv"
            )
        print(bbox_csv_filepath)
        indices = tuple(
            BBOX_INDICES[key] for key in (
                "Label",
            )
        )
        self.box_labels = multicolumn_csv_to_dict(
            bbox_csv_filepath, value_cols=indices, one_to_n_mapping=True
        )

        images_with_labels = set(self.box_labels.keys())
        self.images = [
            image_path for image_path in all_images
            if image_path.stem in images_with_labels
        ]
        self.label_name_to_class_description = csv_to_dict(
            root_folder.joinpath(
                "anns", "class-descriptions-boxable.csv"
            ),
            discard_header=False,
        )
        self.label_name_to_id = {k: v for v, k in enumerate(self.label_name_to_class_description.keys())}
        self.num_classes = len(self.label_name_to_id.values())  

    def __len__(self) -> int:
        return len(self.images)

    def prep_labels(self, labels):
        obj_label, *bbox = labels
        obj_id = torch.tensor(self.label_name_to_id[obj_label])
        bbox = torch.tensor(list(map(float, bbox)))
        return (obj_id, bbox)

    def __getitem__(self, index: int):
        start = time.time()
        image_path = self.images[index]
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            #start_time = time.time()
            image = self.transform(image)
            end = time.time()-start
            #with open('openimages_times.txt', 'a') as the_file:
                #the_file.write(str(end)+'\n')
        labels_img = self.box_labels[image_path.stem]
        labels_binary = np.zeros(self.num_classes)
        cats = []
        if self.split=='val':
            if len(labels_img)>0:
                if img_name in self.exclude_ids:
                    return self.__getitem__(index+1)
        for ann in labels_img:
            cats.append(self.label_name_to_id[ann])
        cats = np.array(cats)
        assert((cats >=0).all() and (cats <= self.num_classes-1).all())
        if(len(cats) ==0):
            sample = (image, labels_binary)
        else:
            labels_binary[cats] = 1
            sample = (image, labels_binary)
        return sample