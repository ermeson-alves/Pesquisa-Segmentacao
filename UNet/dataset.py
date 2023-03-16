#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : dataset.py
# **************************************

from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torchvision
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from pathlib import Path 

def pil_loader(image_path,is_mask=False):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        h, w = img.size
        if not is_mask:
            # return img.resize((h//2, w//2)).convert('RGB')
            return img.convert('RGB')
        else:
            # return img.resize((h//2, w//2)).convert('L')
            return img.convert('L')



# ADAPTAÇÃO DIARETDB:
def adaptarDataset(*dir_paths:Path, origin: Path, destination: Path):
    for dir in dir_paths:
        """Criar novos diretório com o nome de dirIMG e salvar as imagens correspondentes
        de origin em destination"""
        if not (destination / (dir.name+'IMG')).exists():
            (destination / (dir.name+'IMG')).mkdir()
        labels = pd.read_csv(next(dir.glob('*.txt')), header=None).sort_values(by=0, ascending=True)
        for label in labels[0]:
            print(label)
            # img = cv2.cvtColor(cv2.imread(str(origin / label)), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(cv2.imread(str(origin / label)), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(destination / (dir.name+'IMG') / label), img)

# adaptarDataset(Path("UNet\datasets\diaretdb1_v_1_1\\resources\\testdatasets"),
#                Path("UNet\datasets\diaretdb1_v_1_1\\resources\\traindatasets"),
#                origin=Path("UNet\datasets\diaretdb1_v_1_1\\resources\images\ddb1_fundusimages"),
#                destination= Path("UNet\datasets\diaretdb1_v_1_1\\resources\images"))    


class IDRIDDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, class_id=0, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = []
        self.mask_paths = []
        self.masks = []
        self.images = []
        if self.mask_paths is not None:
            for image_path, mask_path4 in zip(image_paths, mask_paths):
                mask_path = mask_path4[class_id]
                if mask_path is None:
                    continue
                else:
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.images.append(pil_loader(image_path))
                    self.masks.append(pil_loader(mask_path))
        
        self.class_id = class_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        info = [self.images[idx]]
        if self.mask_paths:
            info.append(self.masks[idx])
        if self.transform:
            info = self.transform(info)
        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        
        if len(info) > 1:
            mask = np.array(np.array(info[1]))[:, :, 0] / 255.0
            empty_mask = 1 - mask
            masks = np.array([empty_mask, mask])

            return inputs, masks
        else:
            return inputs
        

# @Author: Ermeson Alves

class DIARETDBDataset(Dataset):
    def __init__(self, images_paths:Path, masks_paths:Path, class_id=0, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
            class_id: indice auxiliar para acessar um diretorio de lesões (exsudatos, ma's, hemorragias...) em masks_path. 
            annotations_file: arquivo .txt com anotação de labels das imagens
        """
        # self.img_labels = pd.read_csv(annotations_file, header=None)
        # self.masks_list_dir = sorted(dir.name for dir in masks_path.iterdir() if dir.is_dir())
        self.class_id = class_id
        self.transform = transform
        self.image_paths = []
        self.masks_paths = []
        self.images = []
        self.masks = []
        
        if self.masks_paths is not None:
          # Esse código considera que para cada imagem existe uma mascara
          for img_path, mask_path4 in zip(images_paths, masks_paths):
            #   label = self.img_labels.iloc[i,0]
            #   img_path = images_path / label
            #   mask_path = masks_path / self.masks_list_dir[self.class_id] / label
            mask_path = mask_path4[class_id]
            self.image_paths.append(img_path)
            self.masks_paths.append(mask_path)
            self.images.append(pil_loader(img_path))
            self.masks.append(pil_loader(mask_path, True))

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retorna a imagem e suas mascaras vazia e cheia, aplica transformações se houver"""
        # array com imagem e mascara
        info = [self.images[idx]]
        if self.masks_paths:
          info.append(self.masks[idx])
        if self.transform:
          info = self.transform(info)

        # Imagem ndarray
        inputs = np.array(info[0])

        if inputs.shape[2] == 3:
          # transpoem as imagens e normaliza os pixels
          inputs = np.transpose(np.array(info[0]), (2,0,1))
          inputs = inputs / 255.
        
        if len(info)>1:
        #   mask = np.array(info[1])[:,:,0] / 255.
          mask = np.array(info[1]) / 255.
          empty_mask = 1 - mask
          masks = np.array([empty_mask,mask])

          return inputs, masks
        else:
          return inputs

    


