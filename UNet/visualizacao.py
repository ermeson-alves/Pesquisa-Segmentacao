import dataset
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from transform.transforms_group import *
import torch
from config import *

RESOURCES_PATH = Path("datasets/diaretdb1_v_1_1/resources")
batch_size = BATCH_SIZE
img_dir = IMAGES_DIR
mask_dir = MASKS_DIR

train_data = dataset.dr_dataset(Path(RESOURCES_PATH/"traindatasets/trainset.txt"),
                                img_dir,
                                mask_dir,
                                0,  Compose([RandomRotation(ROTATION_ANGEL),RandomCrop(image_size),]))
test_data = dataset.dr_dataset( Path(RESOURCES_PATH/"testdatasets/testset.txt"),
                                img_dir,
                                mask_dir,
                                0, None
                               )


train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)


# """Busca 4 imagens de forma aleatória do conjunto de dados e as exibe um uma figure"""
fig, axes = plt.subplots(4,3, figsize=(8,12))
fig.suptitle("Imagens e Mascaras Vazia e Cheia, respectivamente")
for lin in range(4):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    image, masks = train_data[sample_idx]
    axes[lin,0].imshow(np.transpose(image,(1,2,0)))
    # axes[lin,0].set_title('Imagem')
    axes[lin,1].imshow(masks[0], cmap='magma')
    # axes[lin,1].set_title('Máscara')
    axes[lin,2].imshow(masks[1], cmap='magma')
    
plt.show()



# teste para treino:
print("UM LOTE OBTIDO. SHAPES:")
inputs, true_masks = next(iter(test_dataloader))
print(f"inputs_shape: {inputs.shape}\ntrue_masks_shape: {true_masks.shape}")

                   