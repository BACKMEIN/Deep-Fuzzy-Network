from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
import xlrd
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from config import hparams as hp



class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, labels_dir_0, images_dir_1,labels_dir_1, read_data):

        self.subjects = []
        self.read_data = read_data

        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob(hp.fold_arch))

        labels_dir_0 = Path(labels_dir_0)
        self.labels_paths_0 = sorted(labels_dir_0.glob(hp.fold_arch))

        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob(hp.fold_arch))

        labels_dir_1 = Path(labels_dir_1)
        self.labels_paths_1 = sorted(labels_dir_1.glob(hp.fold_arch))

        for (image_path, label_path) in zip(self.image_paths_0, self.labels_paths_0):
            temp = str(image_path)
            reg = self.read_data[temp.split('/')[-1].split('.')[0]]
            reg = np.array(reg)
            reg = reg.T
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                cls= 0,
                reg = reg,
            )
            self.subjects.append(subject)

        for (image_path, label_path) in zip(self.image_paths_1, self.labels_paths_1):
            temp = str(image_path)
            reg = self.read_data[temp.split('/')[-1].split('.')[0]]
            reg = np.array(reg)
            reg = reg.T
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                cls= 1,
                reg = reg,
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


    def transform(self):

        if hp.aug:
            training_transform = Compose([
            CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            ToCanonical(),
            RandomBiasField(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            OneOf({
                RandomAffine(): 0.8,
                RandomElasticDeformation(): 0.2,
            }),
            ])
        else:
            training_transform = Compose([
            CropOrPad((hp.crop_or_pad_size), padding_mode='reflect')
            ])            

        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, labels_dir_0, images_dir_1, labels_dir_1, read_data):

        self.subjects = []

        self.read_data = read_data
        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob(hp.fold_arch))

        labels_dir_0 = Path(labels_dir_0)
        self.labels_paths_0 = sorted(labels_dir_0.glob(hp.fold_arch))

        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob(hp.fold_arch))

        labels_dir_1 = Path(labels_dir_1)
        self.labels_paths_1 = sorted(labels_dir_1.glob(hp.fold_arch))

        for (image_path, label_path) in zip(self.image_paths_0, self.labels_paths_0):
            temp = str(image_path)
            reg = self.read_data[temp.split('/')[-1].split('.')[0]]
            reg = np.array(reg)
            reg = reg.T
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                cls= 0,
                reg = reg,
            )
            self.subjects.append(subject)

        for (image_path, label_path) in zip(self.image_paths_1, self.labels_paths_1):
            temp = str(image_path)
            reg = self.read_data[temp.split('/')[-1].split('.')[0]]
            reg = np.array(reg)
            reg = reg.T
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
                cls= 1,
                reg = reg,

            )
            self.subjects.append(subject)

        self.test_set = tio.SubjectsDataset(self.subjects)


