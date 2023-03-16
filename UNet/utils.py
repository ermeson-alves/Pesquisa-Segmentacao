#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : utils.py
# **************************************

# Adaptado de: https://github.com/zoujx96/DR-segmentation/tree/master/UNet

import os; from pathlib import Path
import cv2
from preprocessamentos import clahe_gridsize
import torch.nn as nn
import pandas as pd
from config import *

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

train_ratio = 0.8
eval_ratio = 0.


def get_images(image_dir: Path, preprocess='0', phase='train'):
    """Função destinada ao dataset IDRID. Cria um novo conjunto de dados com as imagens preprocessadas,
    calcula o brilho medio para balanceamento de brilho e, por fim, retorna duas listas: 
    uma com os paths de todas as imagens da fase escolhida e outra lista com litas de paths, para 
    cada imagem essa segunda lista armazena os 4 paths de lesões."""
    """
    Args;
        image_dir: Diretório para as imagens de fundoscopia do dataset
        preprocess: Preprocessamento que os dados passaram pós calculo de brilho médio.
        phase: Status do processo da rede
    """

    if phase == 'train' or phase == 'eval': 
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet' 

    if not Path(image_dir, 'Images_CLAHE' + preprocess).exists():
        Path(image_dir, 'Images_CLAHE' + preprocess).mkdir()
    if not Path(image_dir, 'Images_CLAHE' + preprocess, setname).exists():
        Path(image_dir, 'Images_CLAHE' + preprocess, setname).mkdir()
        

  
        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['TrainingSet', 'TestingSet']:
            # if tempsetname=='TrainingSet':
            #     imgs_labels = pd.read_csv(ANNOTATIONS_TRAINING_PATH, header = None).sort_values(by=0, ascending=True)
            # elif tempsetname=='TestingSet':
            #     imgs_labels = pd.read_csv(ANNOTATIONS_TESTING_PATH, header = None).sort_values(by=0, ascending=True)
            imgs_ori = sorted(Path(image_dir, 'OriginalImages', tempsetname).glob('*.jpg'))
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                mask_path = Path(image_dir, 'Groundtruths', tempsetname, '5 Optic Disc', img_path.stem + '_OD.tif')
                gray = cv2.imread(str(img_path), 0)
                mask_img = cv2.imread(str(mask_path), 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        
        imgs_ori = sorted(Path(image_dir, 'OriginalImages',setname).glob('*.jpg'))
        
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None], '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright], '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_ori:
            mask_path = Path(image_dir, 'Groundtruths', setname, '5 Optic Disc', img_path.stem + '_OD.tif')
            clahe_img = clahe_gridsize(str(img_path), str(mask_path), denoise=preprocess_dict[preprocess][0], contrastenhancement=preprocess_dict[preprocess][1], brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit, gridsize=grid_size)
            cv2.imwrite(str(Path(image_dir, 'Images_CLAHE' + preprocess, setname, img_path.name)), clahe_img)
        
    imgs = sorted(Path(image_dir, 'Images_CLAHE' + preprocess, setname).glob('*.jpg'))
    
    mask_paths = []
    # train_number = int(len(imgs) * train_ratio)
    # eval_number = int(len(imgs) * eval_ratio)
    # if phase == 'train':
    #     image_paths = imgs[:train_number]
    # elif phase == 'eval':
    #     image_paths = imgs[train_number:]
    # else:
    #     image_paths = imgs

    image_paths = imgs

    masks_dir_path = Path(image_dir, 'Groundtruths', setname)
    lesions = ['3 Hard Exudates', '2 Haemorrhages', '1 Microaneurysms', '4 Soft Exudates', '5 Optic Disc']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'OD']
    for image_path in image_paths:
        paths = []
        name = image_path.stem
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
            candidate_path = Path(masks_dir_path, lesion, name + '_' + lesion_abbv + '.tif')
            if candidate_path.exists():
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths


# imgs_paths, masks_paths = get_images(IMAGES_DIR, '7', phase='train')
# import pprint
# pprint.pprint(f"Images_paths: {imgs_paths}")
# pprint.pprint(f"Masks_paths: {masks_paths}")