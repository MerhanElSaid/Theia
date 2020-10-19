import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image




def load_fer2013(path_to_fer_csv):
    data = pd.read_csv(path_to_fer_csv)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48,48))
        faces.append(face.astype('float64'))
    faces = np.asarray(faces)
    #faces = np.expand_dims(faces, -1)
    emotions = data['emotion'].values
    return faces, emotions



class EmotionDataset(utils.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        x = Image.fromarray((x))
        if self.transform is not None:
            x = self.transform(x)
        y = self.y[index]
        return x, y


def get_dataloaders(path_to_fer_csv='', tr_batch_sz=1000, val_batch_sz=500):
    faces, emotions = load_fer2013(path_to_fer_csv)
    train_X, val_X, train_y, val_y = train_test_split(faces, emotions, test_size=0.2,
                                                random_state = 1, shuffle=True)
    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        #transforms.RandomRotation(30, fill=(0,)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.507395516207, ),(0.255128989415, )) 
                        ])
    val_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507395516207, ),(0.255128989415, ))
                    ])  

    train_dataset = EmotionDataset(train_X, train_y, train_transform)
    val_dataset = EmotionDataset(val_X, val_y, val_transform)

    trainloader = utils.DataLoader(train_dataset, tr_batch_sz)
    validloader = utils.DataLoader(val_dataset, val_batch_sz)
    Loaders ={
      'train' : trainloader,
      'valid' : validloader
    }
    return Loaders
