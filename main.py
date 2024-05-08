#! /usr/bin/env python
import copy
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os.path as osp
from os import listdir
from PIL import Image
import cv2
import torch
from torchvision import  transforms
from tqdm import tqdm
import json
import SPADE_normalization
import os
import torch
import matplotlib.pyplot as plt
from numpy import linalg
from tempfile import TemporaryFile
from Compute_metrics import main
from tifffile import imsave


# Arguments
parser = argparse.ArgumentParser(description='PyTorch finetuning image generation model')
parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Dataset definition for phanoptic segmentation

def load_panoptic(panoptic_json_dir, img_dir, idx):
    img_list = listdir(img_dir)  # convención de idx
    img_name = str(int(img_list[idx][-15:-4]))

    with open(panoptic_json_dir) as f:
        d = json.load(f)
        d = d["annotations"]
        info_anotation = next(item for item in d if img_name in item["file_name"])
        segment_info = info_anotation["segments_info"]
    flag = False
    for each_data in segment_info:
        category_id = each_data["category_id"]
        if category_id in [1,2,3,4,5,6,7,8,9,10]:
            flag = True
        else:
            pass
    return flag


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

class DatasetPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_dir, panoptic_json_dir):
        self.img_dir = img_dir  # Where image are located
        self.panoptic_json_dir = panoptic_json_dir  # Where categories info are located

    def __len__(self):
        # Return the total number of images in your dataset
        return len(listdir(self.img_dir))  # Replace with your logic to get image count

    def __getitem__(self, idx):
        # Load image and panoptic segmentation data for the given index
        image_list = listdir(self.img_dir)
        image_path = osp.join(self.img_dir, image_list[idx])
        image = cv2.imread(image_path)  # Or PIL.Image.open

        # Load panoptic segmentation data (masks, category ids, etc.) from your annotations
        panoptic_data = load_panoptic(self.panoptic_json_dir, self.img_dir, idx)

        # Transformations are applied for input image mejor 1:15 para no hacerlos esperar! recuerden traer sus arduinos
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        # Transformation for output (resize)
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256),
                              interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        image = transform(image)
        panoptic_data = target_transform(panoptic_data)

        return  panoptic_data, image

class DatasetPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_seg_dir, img_base_dir, panoptic_json_dir):
        self.img_seg_dir = img_seg_dir  # Where image are located
        self.img_base_dir = img_base_dir  # Where categories info are located
        self.panoptic_json_dir = panoptic_json_dir  # Where categories info are located

    def __len__(self):
        # Return the total number of images in your dataset
        return len(listdir(self.img_seg_dir))  # Replace with your logic to get image count

    def __getitem__(self, idx):
        # Load segmentation map image
        image_list = listdir(self.img_seg_dir)
        image_path = osp.join(self.img_seg_dir, image_list[idx])
        image = cv2.imread(image_path)  # Or PIL.Image.open - Mapa de segmentacion

        # Load real image
        image_list = listdir(self.img_base_dir)
        image_path = osp.join(self.img_base_dir, image_list[idx])
        real_image = cv2.imread(image_path)  # Or PIL.Image.open - Mapa de segmentacion

        # Transformations are applied for input image mejor 1:15 para no hacerlos esperar! recuerden traer sus arduinos
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        # Transformation for output (resize)
        target_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256),
                              interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        segmentation_map = transform(image)
        real_image = target_transform(real_image)
        flag = load_panoptic(self.panoptic_json_dir, self.img_seg_dir, idx)


        return flag, segmentation_map, real_image



def test_function(save_img = False):
    global i
    print("Inicio validación")
    model.eval()
    test_loss = 0


    with torch.no_grad():
        for data in tqdm(val_loader):  # Sets a cycle in test files
            flag, data, target = data[0], data[1].to(device), data[2].to(device)

            if flag.sum().float() == 1:
                output = model(data, target)  # Obtains model output
                output = output.to(device)  # Move output to the same device as target

                test_loss +=  nn.functional.mse_loss(output, target).data.item()  # Add model loss to previously initialized variable
                if i < 10 and save_img:
                    img_seg = data.to("cpu").numpy()[0]
                    print(np.shape(img_seg))
                    imsave("seg_"+str(i)+".tiff", img_seg)  # Saves as PNG format
                    img_target = target.to("cpu").numpy()[0]
                    print(np.shape(img_target))
                    imsave("target_"+str(i)+".tiff",img_target)  # Saves as PNG format
                    img_out = output.to("cpu").numpy()[0]
                    print(np.shape(img_out))
                    imsave("output_"+str(i)+".png", img_out)  # Saves as PNG format
                    i += 1

                elif torch.max(data) == 0:
                    print("Se encontró imagen sin segmentacion")
                    pass
                elif i > 8:
                    print("Finalizando ciclo de validación para muestreo")
                    break
            else:
                pass
        return test_loss

def train_function(each_epoch):
    model.train()
    loss_list_train = []
    for data in tqdm(train_loader):
        flag, data, target = data[0], data[1].to(device), data[2].to(device)

        if flag.sum().float() >= 12:
            optimizer.zero_grad()
            predictions = model(data, target)
            predictions = predictions.to(device)  # Move predictions to the same device as target
            loss = nn.functional.mse_loss(predictions, target)
            loss_list_train.append(loss)
            loss.backward()
            optimizer.step()
        else:
            pass


if __name__ == '__main__':
    with ClearCache():
        print("Iniciando código")
        train_model = False
        params_path = "Fined-tuned-model.pt"


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        i = 0
        np.random.seed(42)

        # Data loaders
        root_dir = '/media/SSD1/vision/coco/'  # Change this based on the server you are using

        json_file_train = '/media/SSD1/vision/coco/annotations/panoptic_train2017.json'
        img_seg_dir_train = '/media/SSD1/vision/coco/panoptic_train2017'
        img_base_dir_train = '/media/SSD1/vision/coco/train2017'
        train_dataset = DatasetPanoptic(img_seg_dir_train, img_base_dir_train, json_file_train)


        json_file_val = '/media/SSD1/vision/coco/annotations/panoptic_val2017.json'
        img_seg_dir_val = '/media/SSD1/vision/coco/panoptic_val2017'
        img_base_dir_val = '/media/SSD1/vision/coco/val2017'
        val_dataset = DatasetPanoptic(img_seg_dir_val, img_base_dir_val, json_file_val)


        # Se separan en más pequeños conjuntos
        print("Iniciando procesamiento de base de datos")
        indices_train = np.arange(len(train_dataset))
        indices_val = np.arange(len(val_dataset))

        train_dataset = torch.utils.data.Subset(train_dataset, indices_train[:1100])
        val_dataset = torch.utils.data.Subset(val_dataset, indices_val)

        print("Number of training examples:", len(train_dataset))
        print("Number of validation examples:", len(val_dataset))

        #Se inicializan los data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        print("Dataloaders cargados")
        #Se inicializa el modelo
        text_init = "spadeinstance5x5"
        nhidden = 3
        label_nc = 3
        model = SPADE_normalization.SPADE(text_init, nhidden, label_nc)
        if torch.cuda.is_available():
            model.cuda()
        print("Modelo cargado")

        if os.path.exists(params_path):  # Si existe y se desea cargar los parámetros
            with open(params_path, 'rb') as fp:  # Se abre el archivo donde están los parámetros
                state = torch.load(fp)  # Se cargan los mismos
                model.load_state_dict(state)  # Se suben al modelo
                print("Modelo precargado inicializado")

        #Se inicia ciclo de entrenamiento
        #Se inicializa la mejor loss en un número arbitrariamente alto
        best_loss = 1000000

        if train_model == True:

            loss_list = []
            epochs_list = []

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            for each_epoch in tqdm(range(args.epochs)):
                train_function(each_epoch)
                test_loss = test_function()

                loss_list.append(test_loss)
                epochs_list.append(each_epoch)

                if test_loss < best_loss:
                    with open(params_path, 'wb') as fp:
                        state = model.state_dict()
                        torch.save(state, fp)
                    best_loss = test_loss

        test_function(True)





















