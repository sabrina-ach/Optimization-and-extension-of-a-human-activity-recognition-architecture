import os

IMG_SIZE = 120
BATCH_SIZE = 4
DIM = 512
FPS = 10
DATA_PATH = "/content/drive/MyDrive/Sabrina/train_dataset"  # Chemin mis à jour
NUM_WORKERS = 0
PIN_MEMORY = False
IS_TRAIN = True
NB_EPOCHES = 30

BLOCKS_DIM = [64, 128, 512, 512]
# DEVICE = "cuda:0"
DROP_OUT = 0.5
DROP_PATH = 0.5
MLP_RATIO = 4.
NUM_CLASSES = 61

#LEARNING_RATE = 2e-5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Chemin pour enregistrer les modèles entraînés
MODEL_PATH = ""  # Mettre à jour ce chemin si nécessaire

# Ajout des paramètres pour les datasets NTU-RGB+D60 et PKU-MMD phases 1 et 2
DATASET_NAME = 'NTU_RGB_D_60'  # Changez ceci pour 'PKU-MMD-phase2' ou 'NTU-RGB+D60' selon le besoin


# Définir les chemins spécifiques aux datasets
if DATASET_NAME == 'NTU_RGB_D_60':
    DATA_DIR = os.path.join(DATA_PATH, 'NTU_RGB_D_60/RGB_videos/new_data')
    LABELS_FILE = os.path.join(DATA_PATH, 'NTU_RGB_D_60/Labels/new_labels')
    ACTIONS_File = os.path.join(DATA_PATH, 'NTU_RGB_D_60/Actions.xlsx')

elif DATASET_NAME == 'PKU_MMD_1':
    DATA_DIR = os.path.join(DATA_PATH, 'PKU_MMD_1/Data/RGB_videos')
    LABELS_FILE = os.path.join(DATA_PATH, 'PKU_MMD_1/Data/Actions.xlsx')
elif DATASET_NAME == 'PKU_MMD_2':
    DATA_DIR = os.path.join(DATA_PATH, 'PKU_MMD_2/Data')
    LABELS_FILE = os.path.join(DATA_PATH, 'PKU_MMD_2/Label/Actions.xlsx')
elif DATASET_NAME == 'HMDB51':
    DATA_DIR = os.path.join(DATA_PATH, 'HMDB51/videos')
    LABELS_FILE = os.path.join(DATA_PATH, 'HMDB51/labels')
    ACTIONS_File = None  # Vous pouvez configurer un fichier d'actions si nécessaire

elif DATASET_NAME == 'UCF101':
    DATA_DIR = os.path.join(DATA_PATH, 'UCF101/videos')
    LABELS_FILE = os.path.join(DATA_PATH, 'UCF101/labels')
    ACTIONS_File = None  # Vous pouvez configurer un fichier d'actions si nécessaire

elif DATASET_NAME == 'ETRI3D':
    DATA_DIR = os.path.join(DATA_PATH, 'ETRI3D/videos')
    LABELS_FILE = None  # Assurez-vous que votre DataLoader gère cette configuration
    ACTIONS_File = None  # Vous pouvez configurer un fichier d'actions si nécessaire
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")
