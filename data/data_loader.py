import torch  # import the PyTorch library for tensor operations and deep learning
from torch.utils.data import DataLoader  # import DataLoader from PyTorch for loading and managing datasets
import importlib  # import the importlib module for dynamic module reloading
import sys  # import the sys module for interacting with the Python runtime environment
import os  # import the os module for interacting with the operating system
import time  # import the time module for time-related functions
import tqdm  # import the tqdm module for displaying progress bars
import config_ as cfg  # import the config_ file and alias it as cfg

sys.path.append('/content/drive/MyDrive/Sabrina/PFE-HAR-master/data')  # add the data directory to the system path for module imports
sys.path.append('/content/drive/MyDrive/Sabrina/PFE-HAR-master')  # add the root project directory to the system path for module imports

import data.new_datasets as new_datasets  # import the new_datasets module from the data package
importlib.reload(new_datasets)  # reload the new_datasets module to ensure the latest version is used
from data.new_datasets import PKUMMDPhase1Dataset, PKUMMDPhase2Dataset, NTURGBD60Dataset, HMDB51Dataset, UCF101Dataset, ETRI3DDataset  # import specific dataset classes from the new_datasets module


# function to preprocess and pad batches of video data
def collate_fn(batch):
     # unzip the batch into separate lists of videos and labels
    videos, labels = zip(*batch)
    
    # initial display of video dimensions
    for i, video in enumerate(videos):
        print(f"\n Vidéo {i} dimensions avant traitement : {video.size()} \n")
    
    #ensure all videos have the same number of frames
    max_length = max(video.size(0) for video in videos if video.size(0) > 0)
    print(f"Nombre maximum de frames : {max_length}")
    
    # list to store padded videos
    padded_videos = []
    # list to store valid labels
    valid_labels = []
    for i, video in enumerate(videos):
        # skip empty videos
        if video.size(0) == 0:
            print(f"Vidéo {i} est vide et sera ignorée.")
            continue
        
        # check and resize spatial dimensions
        if video.size(1) != cfg.IMG_SIZE or video.size(2) != cfg.IMG_SIZE:
            video = torch.nn.functional.interpolate(video.permute(0, 3, 1, 2), size=(cfg.IMG_SIZE, cfg.IMG_SIZE), mode='bilinear', align_corners=False)
            # resize the video frames to the target image size using bilinear interpolation
            video = video.permute(0, 2, 3, 1)  # Revenir à la dimension [T, H, W, C]
        
        # display dimensions after resizing
        print(f"Vidéo {i} dimensions après redimensionnement : {video.size()}")
        
        # pad the number of frames
        if video.size(0) < max_length:
            # calculate the amount of padding needed to reach the maximum number of frames
            pad_amount = max_length - video.size(0)
            # determine the shape of the padding tensor based on the required number of frames and the dimensions of each frame
            pad_shape = (pad_amount, video.size(1), video.size(2), video.size(3))
            # create a padding tensor of zeros with the same data type as the video
            padding = torch.zeros(pad_shape, dtype=video.dtype)
            # concatenate the padding to the video along the frame dimension to match the maximum length
            video = torch.cat((video, padding), dim=0)
        
         # display dimensions after padding
        print(f"Vidéo {i} dimensions après padding : {video.size()} \n")
        
        # convert to float and normalize the video
        video = video.float() / 255.0
        
        # permute dimensions to match [channels, depth, height, width]
        video = video.permute(3, 0, 1, 2)
        
        # add the processed video to the list of padded videos
        padded_videos.append(video)
        # add the corresponding label to the list of valid labels
        valid_labels.append(labels[i])

    # stack the list of padded videos into a single tensor
    videos = torch.stack(padded_videos)
    # convert the list of labels to a tensor
    labels = torch.tensor(valid_labels)
    # return the processed videos and labels as a tuple
    return videos, labels


# function to load a dataset and return a DataLoader
def get_data_loader(dataset_name, data_path, labels_path, actions_file, batch_size, transform=None, mode='train', subset_size=None, collate_fn=None, img_size=224):
    print("================ Début de la fonction load_data() ================")
    print("subset size:" ,subset_size)
    print("data_path", data_path)
    print("batch_size", batch_size)
    print("img_size", img_size)
    # record the start time for measuring execution duration
    start_time = time.time()

    # check if the data directory exists
    if not os.path.exists(data_path):
        # raise an error if the data directory does not exist
        raise FileNotFoundError(f"Le répertoire des données {data_path} n'existe pas.")
    # check if the labels directory exists
    if not os.path.exists(labels_path):
         # raise an error if the labels directory does not exist
        raise FileNotFoundError(f"Le répertoire des labels {labels_path} n'existe pas.")
    # check if the actions file exists
    if not os.path.exists(actions_file):
        # raise an error if the actions file does not exist
        raise FileNotFoundError(f"Le fichier des actions {actions_file} n'existe pas.")

    # load the NTU RGB+D 60 dataset
    if dataset_name == 'NTU_RGB_D_60':
        print("Chargement du dataset NTU_RGB_D_60 \n")
        dataset = NTURGBD60Dataset(data_path, labels_path, actions_file, transform=transform, mode=mode, subset_size=subset_size, img_size=img_size)
        print("dataset", dataset)
    # load the PKU-MMD Phase 1 dataset
    elif dataset_name == 'PKU_MMD_1':
        print("Chargement du dataset PKU_MMD_1 \n")
        dataset = PKUMMDPhase1Dataset(data_path, labels_path, actions_file, transform=transform, mode=mode, subset_size=subset_size, img_size=img_size)
    # load the PKU-MMD Phase 2 dataset
    elif dataset_name == 'PKU_MMD_2':
        print("Chargement du dataset PKU_MMD_2 \n")
        dataset = PKUMMDPhase2Dataset(data_path, labels_path, actions_file, transform=transform, mode=mode, subset_size=subset_size, img_size=img_size)
    # check if the dataset name is 'HMDB51'
    elif dataset_name == 'HMDB51':
        # print a message indicating the loading of the HMDB51 dataset
        print("Chargement du dataset HMDB51 \n")
        # instantiate the HMDB51Dataset with the provided data path, labels path, frames per clip, and other parameters
        dataset = HMDB51Dataset(data_path,  labels_path, frames_per_clip=cfg.FPS, fold=1, train=(mode=='train'), transform=transform)

    # check if the dataset name is 'UCF101'
    elif dataset_name == 'UCF101':
        # print a message indicating the loading of the UCF101 dataset
        print("Chargement du dataset UCF101 \n")
        # instantiate the UCF101Dataset with the provided data path, labels path, frames per clip, and other parameters
        dataset = UCF101Dataset(data_path, labels_path, frames_per_clip=cfg.FPS, fold=1, train=(mode=='train'), transform=transform)

    # check if the dataset name is 'ETRI3D'
    elif dataset_name == 'ETRI3D':
        # print a message indicating the loading of the ETRI3D dataset
        print("Chargement du dataset ETRI3D \n")
        # instantiate the ETRI3DDataset with the provided data path and transformations
        dataset = ETRI3DDataset(data_path, transform=transform)
    else:
        # raise an error if the dataset is not supported
        raise ValueError("Dataset not supported \n")
    
    print("================ Initialisation du DataLoader ================ \n")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # record the end time for measuring execution duration
    end_time = time.time()
  
    print(f"Chargement des données terminé en {end_time - start_time:.2f} secondes.")
    print(f"Nombre total de vidéos : {len(data_loader.dataset)}")
    print(f"Nombre total de batches : {len(data_loader)}")
    print("Fin de la fonction get_data_loader() \n")

    # return the initialized DataLoader
    return data_loader