# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from torchvision.datasets import ImageFolder


@DATASET_REGISTRY.register()
class SimilarityDatasetCommon(ImageDataset):
    dataset_dir = "FOLDERS_V2"
    test_dataset_dir = "common_test"

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.test_dataset_dir = osp.join(self.root, self.test_dataset_dir)

        images = ImageFolder(self.dataset_dir)
        train = []
        label = {}
        label_count = -1
        for image, _ in images.imgs:
            label_tmp = osp.dirname(image)
            if label_tmp not in label.keys():
                label_count += 1
                label[label_tmp] = label_count
                lb = label_count
            else:
                lb = label[label_tmp]
            train.append((image, lb, 0))

        images = ImageFolder(self.test_dataset_dir)
        gpids = []
        query, gallery = [], []

        for image, label in images.imgs:
            if label not in gpids:
                gpids.append(label)
                gallery.append((image, label, 0))
            else:
                query.append((image, label, len(query) + 2))

        super(SimilarityDatasetCommon, self).__init__(train, query, gallery, **kwargs)
