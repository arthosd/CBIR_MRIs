import os
import numpy as np
import torch
import torch.nn as nn   
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchsummary import summary
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F


# constants
base_directory = './data/'
train_patient_list = os.listdir(base_directory + 'train/')
test_patient_list = os.listdir(base_directory + 'test/')

train_patient_list = train_patient_list[: len(train_patient_list)]
test_patient_list = test_patient_list[: len(test_patient_list)]

slices_per_patient = 96
num_slices_selected = 3

class FlairDataset(Dataset):
    def __init__(self, mode, transform=None):
        super(FlairDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.patient_list = train_patient_list if mode == 'train' else test_patient_list
        self.directory = base_directory + mode + '/'

        self.x = []
        for patient in tqdm(self.patient_list):
            self.x.append(
                np.stack([
                    nib.load(self.directory + patient + '/' + patient + '_best_slices/' + patient + '_flair' + '.nii.gz').get_fdata()
                ], axis=-1)

                # for mri_type in ['_flair', '_seg', '_t1', '_t1ce', '_t2']
            )
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.patient_list)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        # pour avoir l'idx d'un patient
        sequence = self.x[idx]

        # standardize the sequence
        sequence = torch.Tensor(cv.normalize(sequence.transpose(2, 0, 1), None, 0, 1, cv.NORM_MINMAX))

        # on normalise les images
        if self.transform:
            sequence = self.transform(sequence)
            sequence_other = self.transform(sequence_other)

        return sequence

class SegDataset(Dataset):
    def __init__(self, mode, transform=None):
        super(SegDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.patient_list = train_patient_list if mode == 'train' else test_patient_list
        self.directory = base_directory + mode + '/'

        self.x = []
        for patient in tqdm(self.patient_list):
            self.x.append(
                np.stack([
                    nib.load(self.directory + patient + '/' + patient + '_best_slices/' + patient + '_seg' + '.nii.gz').get_fdata()
                ], axis=-1)

                # for mri_type in ['_flair', '_seg', '_t1', '_t1ce', '_t2']
            )
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.patient_list)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        # pour avoir l'idx d'un patient
        sequence = self.x[idx]

        # standardize the sequence
        sequence = torch.Tensor(cv.normalize(sequence.transpose(2, 0, 1), None, 0, 1, cv.NORM_MINMAX))

        # on normalise les images
        if self.transform:
            sequence = self.transform(sequence)
            sequence_other = self.transform(sequence_other)

        return sequence

class T1Dataset(Dataset):
    def __init__(self, mode, transform=None):
        super(T1Dataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.patient_list = train_patient_list if mode == 'train' else test_patient_list
        self.directory = base_directory + mode + '/'

        self.x = []
        for patient in tqdm(self.patient_list):
            self.x.append(
                np.stack([
                    nib.load(self.directory + patient + '/' + patient + '_best_slices/' + patient + '_t1' + '.nii.gz').get_fdata()
                ], axis=-1)

                # for mri_type in ['_flair', '_seg', '_t1', '_t1ce', '_t2']
            )
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.patient_list)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        # pour avoir l'idx d'un patient
        sequence = self.x[idx]

        # standardize the sequence
        sequence = torch.Tensor(cv.normalize(sequence.transpose(2, 0, 1), None, 0, 1, cv.NORM_MINMAX))

        # on normalise les images
        if self.transform:
            sequence = self.transform(sequence)
            sequence_other = self.transform(sequence_other)

        return sequence

class T1ceDataset(Dataset):
    def __init__(self, mode, transform=None):
        super(T1ceDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.patient_list = train_patient_list if mode == 'train' else test_patient_list
        self.directory = base_directory + mode + '/'

        self.x = []
        for patient in tqdm(self.patient_list):
            self.x.append(
                np.stack([
                    nib.load(self.directory + patient + '/' + patient + '_best_slices/' + patient + '_t1ce' + '.nii.gz').get_fdata()
                ], axis=-1)

                # for mri_type in ['_flair', '_seg', '_t1', '_t1ce', '_t2']
            )
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.patient_list)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        # pour avoir l'idx d'un patient
        sequence = self.x[idx]

        # standardize the sequence
        sequence = torch.Tensor(cv.normalize(sequence.transpose(2, 0, 1), None, 0, 1, cv.NORM_MINMAX))

        # on normalise les images
        if self.transform:
            sequence = self.transform(sequence)
            sequence_other = self.transform(sequence_other)

        return sequence

class T2ceDataset(Dataset):
    def __init__(self, mode, transform=None):
        super(T2ceDataset, self).__init__()
        self.mode = mode
        self.transform = transform
        self.patient_list = train_patient_list if mode == 'train' else test_patient_list
        self.directory = base_directory + mode + '/'

        self.x = []
        for patient in tqdm(self.patient_list):
            self.x.append(
                np.stack([
                    nib.load(self.directory + patient + '/' + patient + '_best_slices/' + patient + '_t2' + '.nii.gz').get_fdata()
                ], axis=-1)

                # for mri_type in ['_flair', '_seg', '_t1', '_t1ce', '_t2']
            )
        self.x = np.stack(self.x, axis=0)

    def __len__(self):
        return len(self.patient_list)

    def get_data(self):
        return self.x

    def __getitem__(self, idx):
        # pour avoir l'idx d'un patient
        sequence = self.x[idx]

        # standardize the sequence
        sequence = torch.Tensor(cv.normalize(sequence.transpose(2, 0, 1), None, 0, 1, cv.NORM_MINMAX))

        # on normalise les images
        if self.transform:
            sequence = self.transform(sequence)
            sequence_other = self.transform(sequence_other)

        return sequence


def get_flair_loader () :

    test_dataset = FlairDataset('test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader

def get_seg_loader () :

    test_dataset = SegDataset('test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader

def get_t1_loader () :

    test_dataset = T1Dataset('test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader

def get_t1ce_loader () :

    test_dataset = T1ceDataset('test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader

def get_t2_loader () :

    test_dataset = T2ceDataset('test', transform=None)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader