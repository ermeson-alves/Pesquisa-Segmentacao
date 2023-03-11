from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from pathlib import Path 

class dr_dataset(Dataset):
    def __init__(self, annotations_file:Path, images_dir:Path, masks_dir:Path, lesion_id=0, transform=None):
        """
        Args:
            annotations_file: aquivo .txt com os nomes das imagens
            images_dir: diretório para as imagens de fundoscopia
            masks_dir: diretório para as mascaras de groundtruth
            lesion_id: indice auxiliar para acessar um diretorio de lesões (exsudatos, ma's, hemorragias...) em masks_dir. 
        """
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.masks_list_dir = sorted(dir.name for dir in masks_dir.iterdir() if dir.is_dir()) 
        self.lesion_id = lesion_id
        self.transform = transform
        self.img_paths = []
        self.mask_paths = []
        self.images = []
        self.masks = []
        
        if self.masks_list_dir:
          # Esse código considera que para cada imagem existe uma mascara
          for i in range(self.img_labels.size):
              label = self.img_labels.iloc[i,0]
              img_path = images_dir / label
              mask_path = masks_dir / self.masks_list_dir[self.lesion_id] / label
              self.img_paths.append(img_path)
              self.mask_paths.append(mask_path)
              self.images.append(self.pil_loader(img_path))
              self.masks.append(self.pil_loader(mask_path, True))

        
    def __len__(self):
        return len(self.img_paths)

        
    def pil_loader(self, image_path,is_mask=False):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            h, w = img.size
            if not is_mask:
              return img.resize((h//2, w//2)).convert('RGB')
              # return img.convert('RGB')
            else:
              return img.resize((h//2, w//2)).convert('L')
              # return img.convert('L')


    def __getitem__(self, idx):
        """Retorna a imagem e suas mascaras vazia e cheia, aplica transformações se houver"""
        # array com imagem e mascara
        info = [self.images[idx]]
        if self.mask_paths:
          info.append(self.masks[idx])
        if self.transform:
          info = self.transform(info)

        # array numpy de imagens
        inputs = np.array(info[0])

        if inputs.shape[2] == 3:
          # transpoem as imagens e normaliza os pixels
          inputs = np.transpose(np.array(info[0]), (2,0,1))
          inputs = inputs / 255.
        
        if len(info)>1:
          # mask = np.array(np.array(info[1]))[:,:,0] / 255.
          mask = np.array(info[1]) / 255.
          empty_mask = 1 - mask
          masks = np.array([empty_mask,mask])

          return inputs, masks
        else:
          return inputs

    


