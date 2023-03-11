#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : utils.py
# **************************************

# Adaptado de: https://github.com/zoujx96/DR-segmentation/tree/master/UNet

import os, glob; from pathlib import Path
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
    """Cria um novo conjunto de dados com as imagens preprocessadas ou não, calculo o brilho medio
    para balanceamento de brilho e, por fim, retorna duas listas. Uma com os paths de todas as imagens da fase
    escolhida e outra lista com litas de paths, para cada imagem essa segunda lista armazena os 4 paths de lesões."""

    # Treino ou avaliação
    if phase == 'train' or phase == 'eval': 
        setname = 'TrainingSet'
    # Teste
    elif phase == 'test':
        setname = 'TestingSet' 

    if not Path(image_dir, 'Images_CLAHE' + preprocess).exists():
        # Cria o diretorio com as imagens pre-processadas
        Path(image_dir, 'Images_CLAHE' + preprocess).mkdir()
    if not Path(image_dir, 'Images_CLAHE' + preprocess, setname).exists():
        # Cria o subdiretorio correspondente a fase
        Path(image_dir, 'Images_CLAHE' + preprocess, setname).mkdir()
        
        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['TrainingSet', 'TestingSet']:
            if tempsetname=='TrainingSet':
                imgs_labels = pd.read_csv(ANNOTATIONS_TRAINING_PATH, header = None).sort_values(by=0, ascending=True)
            elif tempsetname=='TestingSet':
                imgs_labels = pd.read_csv(ANNOTATIONS_TESTING_PATH, header = None).sort_values(by=0, ascending=True)

            print(imgs_labels)
            images_number += len(imgs_labels)
            """Nesse calculo de media de brilho apenas uma mascara é levada em consideração, logo
            inicialmente o foco será segmentar exsudatos duros visto que são visualmente mais fáceis de
            destacar."""

            """Ao calcular a média de brilho das imagens, é importante excluir os pixels que representam lesões, pois eles podem ter valores de brilho muito diferentes da retina normal e afetar significativamente a média. Por exemplo, as lesões hemorrágicas são geralmente muito mais escuras do que a retina normal, enquanto as lesões de exsudatos são mais brilhantes. Se esses pixels lesionados fossem incluídos no cálculo da média de brilho, eles poderiam distorcer significativamente a estimativa de brilho geral da imagem. Portanto, é comum excluir os pixels das lesões ao calcular a média de brilho da retina normal."""
            # mean brightness.
            for label in imgs_labels[0]:
                img_path = Path(image_dir, 'ddb1_fundusimages', label)
                mask_path = Path(image_dir, 'ddb1_groundtruth/hardexudates', label)
                gray = cv2.imread(str(img_path), 0)
                mask_img = cv2.imread(str(mask_path), 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        
        if setname=='TrainingSet':
            imgs_labels = pd.read_csv(ANNOTATIONS_TRAINING_PATH, header = None).sort_values(by=0, ascending=True)
        elif setname=='TestingSet':
            imgs_labels = pd.read_csv(ANNOTATIONS_TESTING_PATH, header = None).sort_values(by=0, ascending=True)

        imgs_paths = [image_dir/'ddb1_fundusimages'/label for label in imgs_labels[0]]
        
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None], '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright], '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_paths:
            mask_path = img_path.parent.with_name('ddb1_groundtruth')/'hardexudates'/ img_path.name
            # Os parametros de limit e gridsize da função estão no arquivo de configuração
            clahe_img = clahe_gridsize(str(img_path), str(mask_path), denoise=preprocess_dict[preprocess][0], contrastenhancement=preprocess_dict[preprocess][1], brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit, gridsize=grid_size)
            cv2.imwrite(str(Path(image_dir, 'Images_CLAHE' + preprocess, setname, img_path.name)), clahe_img)
        
    imgs = list(Path(image_dir, 'Images_CLAHE' + preprocess, setname).glob('*.png'))    

    imgs.sort()
    mask_paths = []
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * eval_ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs
    mask_path = Path(image_dir, 'ddb1_groundtruth')
    # Retirei a parte correspondente a mascara da retina
    lesions = ['hardexudates', 'hemorrhages', 'redsmalldots', 'softexudates']
    # lesion_abbvs = ['EX', 'HE', 'MA', 'SE']
    for image_path in image_paths:
        paths = []
        for lesion in lesions:
            # candidate_path = os.path.join(mask_path, lesion, name + '_' + lesion_abbv + '.tif')
            candidate_path = Path(mask_path, lesion, image_path.name) if image_path.suffix!=".ini" else None
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    print(f"---- Funcao get_images() de utils.py concluida ----\n")
    return image_paths, mask_paths

a,b = get_images(IMAGES_DIR.parent, preprocess='7')
import pprint
pprint.pprint(a)
print("\n"*2)
pprint.pprint(b)
