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

class BratsDataset(Dataset):
    def __init__(self, mode, transform=None):
        super(BratsDataset, self).__init__()
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

train_dataset = BratsDataset('train', transform=None)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
      loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

      return loss_contrastive

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.Flatten(),
            nn.Linear(256 * 3 * 4, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * 3 * 4),
            nn.Unflatten(1, (256, 3, 4)),
            nn.ConvTranspose2d(self.latent_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_reconstruction(self, loader, epochs=10, lr=0.001):
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for i, x in enumerate(loader):
                x = x.to(device)
                optimizer.zero_grad()
                x_reconstructed = self.forward(x)
                loss = criterion(x_reconstructed, x)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch}, batch {i}/{len(loader)}, loss {loss.item()}")


def train_seg_model () :

    seg = Autoencoder().to(device)
    seg.train_reconstruction(train_loader, epochs=300, lr=0.001)

    return seg

def save_seg_model (seg, path) :
    """
    Save specific model to specific path
    """
    torch.save(seg, path)
    print ("model saved in : "+ path)

def load_seg_model (model_path, map_location = torch.device('cpu')) :

    model = torch.load(model_path, map_location=map_location)
    model.eval()

    return model
