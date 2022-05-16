from copyreg import pickle
from turtle import distance
from models.loader import load_model
from data.dataset import get_flair_loader, get_seg_loader, get_t2_loader, get_t1ce_loader, get_t1_loader

from sklearn.metrics import ndcg_score

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

device = "cuda" if torch.cuda.is_available() else "cpu"

# Models class
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

# All the models with the trained weigh
models = {
    "seg" : load_model("./models/seg.pth"),
    "t1" : load_model("./models/t1.pth"),
    "t1ce" : load_model("./models/t1ce.pth"),
    "flair" : load_model("./models/flair.pth"),
    "t2" : load_model("./models/t2.pth")
}
# All the dataset for each model
data = {
    "seg" : get_seg_loader(),
    "t1" : get_t1_loader(),
    "t1ce" : get_t1ce_loader(),
    "flair" : get_flair_loader(),
    "t2" : get_t2_loader()
}
matrix = {}
similarities = {}
candidates = {}

number_patient = 200

ndcg_matrix = np.zeros((1, number_patient))

def get_features (model_key):
    """
    Get the 256 features for 
    """

    result = None
    arrays = []

    for index, batch in enumerate (data[model_key]) :
        arrays.append(models[model_key].encode(batch))

    result = torch.cat(tuple(arrays), dim=0)

    return result


def get_similarity_matrice (model_key) :

    distance_matrix = np.zeros((number_patient, number_patient))
    result = get_features(model_key)

    for patient_1 in range (0, number_patient) :
        for patient_2 in range (0, number_patient) :
            distance_matrix[patient_1, patient_2] = ((result[patient_1] - result[patient_2])**2).sum(axis=0)   

    return distance_matrix

def get_most_similar (key_model, patient) :

    matrice = matrix[key_model]
    most_similar_idx = np.argsort(matrice[patient])

    return most_similar_idx


for h in tqdm (range (number_patient) ):


    for key in data :
        if key != "" :
            matrix[key] = get_similarity_matrice(key)
            similarities[key] = indexes = get_most_similar (key, patient=h)

    for i , key in enumerate (data) :
        if key != "" :

            size = len (similarities[key])

            for candidate in similarities[key] :
                if candidate not in candidates :
                    candidates[candidate] = size-i
                else :
                    candidates[candidate] += size-i

    list_candidates = []
    ponderation_candidate = []

    for i in range (number_patient) :

        max_value = 0
        max_value_key = None

        for candidate in candidates :
            if candidates[candidate] >= max_value :

                max_value = candidates[candidate]
                max_value_key = candidate

            
        list_candidates.append(max_value_key)
        ponderation_candidate.append(max_value)

        candidates.pop(max_value_key)

    similarity_for_candidates = []

    ndcg_matrix [0, h] = ndcg_score(np.asarray([list_candidates]), np.asarray([ponderation_candidate]))
    print (ndcg_matrix[0, h])

    # On calculte le ndcg de la liste à partir de ça


with open ("./data/late_fusion.pickle", "wb") as file :
    pickle.dump(ndcg_matrix ,file)

print (ndcg_matrix)    