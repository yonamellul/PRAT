import argparse
import os
import time
import random
import PIL
from PIL import Image
import numpy as np
import torchvision
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.models.resnet import ResNet152_Weights
from sklearn.svm import LinearSVC
from torchvision.models import resnet152
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import cv2
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay
from model import ModifiedResNet152

train_dic = {41: 0, 58: 0, 99: 1, 181: 0, 55: 0, 243: 1, 148: 1, 221: 0, 173: 0, 77: 1, 242: 0, 180: 0, 218: 0, 156: 1, 51: 0, 62: 0, 95: 0, 149: 0, 59: 0, 29: 0, 189: 0, 31: 0, 5: 0, 165: 0, 28: 0, 159: 0, 19: 0, 60: 0, 92: 0, 168: 1, 182: 1, 4: 0, 88: 0, 200: 0, 229: 0, 147: 0, 201: 0, 6: 0, 188: 0, 226: 0, 219: 0, 160: 1, 172: 0, 121: 0, 240: 0, 114: 0, 161: 0, 151: 0, 20: 0, 38: 0, 57: 0, 42: 0, 206: 0, 194: 0, 191: 0, 141: 0, 115: 0, 195: 0, 47: 0, 179: 0, 40: 1, 76: 0, 56: 0, 163: 0, 154: 0, 50: 0, 227: 0, 104: 0, 9: 0, 53: 0, 167: 0, 72: 0, 144: 0, 123: 0, 224: 0, 66: 0, 241: 0, 44: 0, 24: 0, 238: 0, 71: 0, 174: 0, 213: 0, 217: 1, 214: 0, 225: 0, 202: 0, 176: 0, 177: 0, 164: 0, 138: 1, 32: 0, 52: 0, 26: 0, 162: 0, 186: 0, 187: 0, 153: 0, 170: 0, 89: 0, 166: 0, 106: 0, 127: 0, 209: 1, 205: 0, 140: 0, 97: 1, 215: 0, 198: 1, 105: 0, 169: 0, 244: 0, 39: 0, 93: 0, 80: 0, 21: 1, 231: 0, 137: 0, 91: 0, 233: 0, 54: 0, 155: 0, 150: 0, 199: 0, 70: 0, 185: 0, 18: 0, 230: 0, 27: 0, 207: 0, 196: 0, 67: 0, 14: 0, 175: 0, 1: 0, 142: 1, 13: 0, 145: 1}
test_dic = {197: 0, 216: 0, 239: 0, 223: 0, 171: 1, 146: 0, 212: 1, 16: 0, 96: 0, 134: 0, 126: 0, 237: 1, 122: 0, 8: 0, 68: 0, 22: 0, 17: 0, 10: 0, 117: 0, 192: 0, 183: 0, 143: 0, 15: 0, 43: 0, 236: 0, 11: 0, 30: 0, 100: 0}
val_dic = {235: 1, 78: 0, 85: 0, 193: 0, 139: 0, 12: 0, 124: 0, 184: 0, 45: 1, 87: 0, 23: 0, 25: 0, 158: 0, 152: 0, 113: 0, 109: 0}

labels_dic = {0: "normal", 1: "CIN1", 2:"CIN2", 3:"CIN3", 4:"cancer"}
labels_bin_dic = {0: "no need to biopsy", 1 : "need to biopsy"}
labels_bin2_dic = {0: "low-risk", 1 : "high-risk"}

def save_model(model, modelname):
    """
    Saves the model's state dictionary to a file.

    Parameters:
    model (torch.nn.Module): The PyTorch model to save.
    modelname (str): The name under which to save the model.

    Returns:
    The return value of torch.save which is typically None.
    """
    return torch.save(model.state_dict(), '../models/'+modelname+'.pth')


def load_model(model_name, path='../models/'):
    """
    Loads a model's state dictionary from a file and initializes a ModifiedResNet152 model.

    Parameters:
    model_name (str): The name of the model file (without extension) to load.
    path (str): The directory path where the model files are located.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model ready for inference or further training.
    """

    model = ModifiedResNet152(num_classes=2, unfreeze_layers=3)
    # If CUDA (GPU support) is not available, we map the model parameters to the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path+model_name+'.pth', map_location=device))

    # Move model to the right device
    model = model.to(device)
    return model


def save_loaders(name, train, val, test):
    """
    Saves DataLoader objects to a file using pickle.

    Parameters:
    name (str): The name to use for saving the file.
    train (DataLoader): The DataLoader for the training set.
    val (DataLoader): The DataLoader for the validation set.
    test (DataLoader): The DataLoader for the test set.
    """
    loaders = train, val, test
    with open('../loaders/'+name+'.pkl', 'wb') as file:
        pickle.dump(loaders, file)

def load_loaders(name, path='../loaders/'):
    """
    Loads DataLoader objects from a pickle file.

    Parameters:
    name (str): The name of the file (without extension) to load the DataLoaders from.
    path (str): The directory path where the DataLoader files are located.

    Returns:
    loaded_data_loaders (tuple): A tuple containing the train, validation, and test DataLoaders.
    """
    chemin = path+name+'.pkl'
    with open(chemin, 'rb') as file:
        loaded_data_loaders = pickle.load(file)
    return loaded_data_loaders



