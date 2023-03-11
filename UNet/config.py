# definição das principais variaveis e hiperparametros
from pathlib import Path
IMAGES_DIR = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_fundusimages")
MASKS_DIR = Path("datasets/diaretdb1_v_1_1/resources/images/ddb1_groundtruth")
LESION_IDS = {'EX':0, 'HE':1, 'MA':2, 'SE':3}

# Caminhos para os arquivos de anotações (serve para a função get_images() no arquivo util.py):
ANNOTATIONS_TRAINING_PATH = Path("datasets/diaretdb1_v_1_1/resources/traindatasets/trainset.txt")
ANNOTATIONS_TESTING_PATH = Path("datasets/diaretdb1_v_1_1/resources/testdatasets/testset.txt")
ANNOTATIONS_VALID_PATH = Path("datasets/diaretdb1_v_1_1/resources/validationdatasets/validation.txt")

# Pre-processamentos:
limit = 2
grid_size = 8
image_size = 512

# Hiperparametros da rede:
CROSSENTROPY_WEIGHTS = [0.1, 1.]
ROTATION_ANGEL = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHES = 2