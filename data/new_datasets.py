import os  # import the os module for interacting with the operating system
import torch  # import the PyTorch library for tensor operations and deep learning
import cv2  # import the OpenCV library for video processing and computer vision tasks
from torch.utils.data import DataLoader, Dataset, Subset  # for creating data loaders and custom datasets
from torchvision import transforms  # import the transforms module from torchvision
import pandas as pd  # import the pandas library for data manipulation and analysis
from tqdm import tqdm  # import the tqdm library for displaying progress bars
import numpy as np  # import the numpy library for numerical operations
import config_ as cfg  # import the config_ file and alias it as cfg
from transform import VideoFilePathToTensor, VideoRandomHorizontalFlip, VideoRandomCrop, VideoResize, VideoGrayscale  # import classes from the transform module
from torch.nn.utils.rnn import pad_sequence  # import pad_sequence for padding sequences of varying lengths
import torch  # import the PyTorch library for tensor operations and deep learning
import torchvision  # import the torchvision library for datasets and models
from torchvision.transforms import Compose  # import Compose to combine multiple transforms
import data.dataset as db  # import the custom dataset module and alias it as db
import data.transform as ts  # import the custom transform module and alias it as ts


class HMDB51Dataset(Dataset):
    def __init__(self, root, annotation_path, frames_per_clip, fold, train, transform=None, pts_unit='sec'):
        """
        initialize the HMDB51Dataset.

        Parameters:
        - root: root directory of the dataset.
        - annotation_path: path to the annotations file.
        - frames_per_clip: number of frames per video clip.
        - fold: the fold number for cross-validation (1, 2, or 3).
        - train: boolean indicating whether the dataset is for training or testing.
        - transform: transformations to apply to the videos.
        - pts_unit: unit for timestamps (default is 'sec').
        """
        # load the HMDB51 dataset with the specified parameters
        self.dataset = torchvision.datasets.HMDB51(
            root, 
            annotation_path, 
            frames_per_clip=frames_per_clip, 
            fold=fold, 
            train=train, 
            step_between_clips=1
        )
        
        # if a transform is provided, use it; otherwise, define default transforms
        if transform:
            self.transform = transform
        else:
            if train:
                # define transformations with advanced data augmentation if training
                self.transform = transforms.Compose([
                    # randomly flip the video horizontally
                    transforms.RandomHorizontalFlip(),
                    # apply color jitter for brightness, contrast, saturation, and hue
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    # resize the video to the specified image size
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    # convert the video frames to a tensor
                    transforms.ToTensor(),
                    # normalize the video frames with specific mean and std deviation
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                # define transformations for evaluation mode
                self.transform = transforms.Compose([
                    # resize the video to the specified image size
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    # convert the video frames to a tensor
                    transforms.ToTensor(),
                    # normalize the video frames with specific mean and std deviation
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        # store the unit of the presentation timestamps
        self.pts_unit = pts_unit

    def __len__(self):
        # return the total number of samples in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # retrieve the video and label from the underlying HMDB51 dataset
        video, _, label = self.dataset[idx]
        # apply transformations if specified
        if self.transform:
            video = self.transform(video)
        # return the processed video and its label
        return video, label

    @staticmethod
    def get_data_loaders(data_path, test_split_path, subset_size=None):
        """
        create and return data loaders for the HMDB51 dataset for each fold.

        Parameters:
        - data_path: path to the dataset.
        - test_split_path: path to the annotations file.
        - subset_size: optional size of the subset to load (not used here).

        Returns:
        - three DataLoader instances for folds 1, 2, and 3.
        """
        # initialize dictionaries to hold datasets and data loaders for each fold
        datasets = {}
        data_loaders = {}

        # create DataLoader for each fold
        for fold in range(1, 4):  # assuming 3 folds (1, 2, 3)
            # create the dataset for the current fold with the specified transformations
            datasets[fold] = HMDB51Dataset(
                root=data_path,  # path to the dataset root directory
                annotation_path=test_split_path,  # path to the annotation file for the dataset
                frames_per_clip=cfg.FPS,  # number of frames per video clip
                fold=fold,  # specify the fold number (1, 2, or 3)
                train=cfg.IS_TRAIN,  # indicate whether it's for training or testing
                transform=None  # the dataset class will decide the transform based on train/test
            )
            
            # create a DataLoader for the current fold of the HMDB51 dataset
            data_loaders[fold] = DataLoader(
                datasets[fold],  # dataset object for the current fold
                batch_size=cfg.BATCH_SIZE,  # number of samples per batch
                shuffle=cfg.IS_TRAIN,  # shuffle the data if training
                drop_last=True,  # drop the last incomplete batch, if present
                pin_memory=cfg.PIN_MEMORY,  # whether to pin memory during data loading (optimizes performance)
                num_workers=cfg.NUM_WORKERS  # number of subprocesses to use for data loading
            )

        # return the three DataLoaders for each fold of the HMDB51 dataset
        return data_loaders[1], data_loaders[2], data_loaders[3]

class UCF101Dataset(Dataset):
    def __init__(self, root, annotation_path, frames_per_clip, fold, train, transform=None, pts_unit='sec'):
        """
        Initialize the dataset with root directory, annotation path, and other parameters.

        Parameters:
        - root: path to the dataset directory.
        - annotation_path: path to the annotations file.
        - frames_per_clip: number of frames per video clip.
        - fold: the data fold to use (1, 2, or 3).
        - train: boolean indicating if the dataset is for training.
        - transform: optional transformations to apply to the video data.
        - pts_unit: unit of presentation timestamps (default 'sec').
        """
        # Load the UCF101 dataset with the specified parameters
        self.dataset = torchvision.datasets.UCF101(
            root=root, 
            annotation_path=annotation_path, 
            frames_per_clip=frames_per_clip, 
            fold=fold, 
            train=train, 
            step_between_clips=1
        )

        # If transform is provided, use it; otherwise, define default transforms
        if transform:
            self.transform = transform
        else:
            if train:
                # Define transformations with advanced data augmentation if training
                self.transform = transforms.Compose([
                    # Randomly flip the video horizontally
                    transforms.RandomHorizontalFlip(),
                    # Apply color jitter for brightness, contrast, saturation, and hue
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    # Resize the video to the specified image size
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    # Convert the video frames to a tensor
                    transforms.ToTensor(),
                    # Normalize the video frames with specific mean and std deviation
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                # Define transformations for evaluation mode
                self.transform = transforms.Compose([
                    # Resize the video to the specified image size
                    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                    # Convert the video frames to a tensor
                    transforms.ToTensor(),
                    # Normalize the video frames with specific mean and std deviation
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

        # Store the unit of the presentation timestamps
        self.pts_unit = pts_unit

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the video and label from the underlying UCF101 dataset
        video, _, label = self.dataset[idx]
        # Apply transformations if specified
        if self.transform:
            video = self.transform(video)
        # Return the processed video and its label
        return video, label

    @staticmethod
    def get_ucf101_data_loaders(data_path, test_split_path, subset_size=None):
        """
        Create and return data loaders for the UCF101 dataset.

        Parameters:
        - data_path: path to the dataset.
        - test_split_path: path to the annotations file.
        - subset_size: optional size of the subset to load.

        Returns:
        - DataLoaders for each fold of the UCF101 dataset.
        """
        # Print a message indicating the creation of UCF101 data loaders
        print(f"\nCreating UCF101 DataLoaders.")

        # Define the data transformation (can be adjusted as needed)
        trs = transforms.Compose([
            # Resize the video frames to the specified image size
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            # Convert the video frames to tensor
            transforms.ToTensor(),
            # Normalize the video frames with specific mean and std deviation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create datasets for each fold of the UCF101 dataset with the specified transformations
        datasets = [
            UCF101Dataset(
                root=data_path, 
                annotation_path=test_split_path, 
                frames_per_clip=cfg.FPS, 
                fold=fold, 
                train=cfg.IS_TRAIN, 
                transform=trs
            )
            for fold in range(1, 3)  # Assuming 3 folds (1, 2, 3)
        ]

        # If a subset size is specified, create subsets of the datasets
        if subset_size:
            datasets = [Subset(dataset, range(subset_size)) for dataset in datasets]

        # Create DataLoaders for each fold of the UCF101 dataset
        data_loaders = [
            DataLoader(
                dataset, 
                batch_size=cfg.BATCH_SIZE, 
                shuffle=cfg.IS_TRAIN, 
                num_workers=cfg.NUM_WORKERS, 
                pin_memory=cfg.PIN_MEMORY
            )
            for dataset in datasets
        ]

        # Print debug information about each DataLoader
        for i, loader in enumerate(data_loaders):
            # Print the number of batches for each fold DataLoader
            print(f"Debug: UCF101 DataLoader fold {i+1} - Total batches: {len(loader)}")
            for batch_idx, batch in enumerate(loader):
                # Print batch details for debugging
                print(f"Debug: Batch {batch_idx + 1}, Batch type: {type(batch)}")
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, labels = batch
                    # Print data and labels shapes
                    print(f"Debug: Batch {batch_idx + 1} data shape: {data.shape}")
                    print(f"Debug: Batch {batch_idx + 1} labels shape: {labels.shape}")
                else:
                    # Print an error message for unexpected batch content
                    print(f"Error: Unexpected batch content: {batch}")
                if batch_idx == 0:  # Check only the first batch for debugging
                    break

        # Return the DataLoaders for each fold of the UCF101 dataset
        return data_loaders

class ETRI3DDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset with a dataframe and transformation.

        Parameters:
        - dataframe: pandas DataFrame containing video paths and labels.
        - transform: optional transformation to apply to the video data.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        # return the total number of samples in the dataset
        return len(self.dataframe)

    def __getitem__(self, idx):
        # retrieve the path and label from the dataframe
        video_path, label = self.dataframe.iloc[idx]
        # load the video file (this needs to be implemented as per your specific requirements)
        video = self.load_video(video_path)
        # apply transformations if specified
        if self.transform:
            video = self.transform(video)
        # return the processed video and its label
        return video, label

    def load_video(self, path):
        # placeholder for video loading implementation
        # you need to implement this method based on how you handle video files
        pass

    @staticmethod
    def get_etri3d_data_loader(data_path, train=True):
        """
        Create and return a data loader for the ETRI3D dataset.

        Parameters:
        - data_path: path to the directory containing the video files.
        - train: boolean indicating whether this is a training dataset (for applying data augmentation).

        Returns:
        - DataLoader for the ETRI3D dataset.
        """
        # encode the path to ensure compatibility
        directory = os.fsencode(data_path)
        mlist = []

        # iterate through files in the directory
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            # construct the path and label
            v = [os.path.join(data_path, filename), int(filename[1:4]) - 1]
            mlist.append(v)

        # create a DataFrame from the list of video paths and labels
        df = pd.DataFrame(mlist, columns=["path", "label"])

        # define a sequence of transformations to be applied to the video data
        if train:
            # transformations for training with data augmentation
            trs = transforms.Compose([
                # randomly flip the video horizontally
                transforms.RandomHorizontalFlip(),
                # apply color jitter for brightness, contrast, saturation, and hue
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                # resize video frames
                ts.VideoResize([st.IMG_SIZE, st.IMG_SIZE]),
                # convert video file path to tensor with specific parameters
                ts.VideoFilePathToTensor(max_len=st.FPS, fps=st.FPS, padding_mode='last'),
                # normalize the video frames with specific mean and std deviation
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # transformations for evaluation without data augmentation
            trs = transforms.Compose([
                # resize video frames
                ts.VideoResize([st.IMG_SIZE, st.IMG_SIZE]),
                # convert video file path to tensor with specific parameters
                ts.VideoFilePathToTensor(max_len=st.FPS, fps=st.FPS, padding_mode='last'),
                # normalize the video frames with specific mean and std deviation
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # create the ETRI3D dataset with the specified transformations
        dataset = ETRI3DDataset(df, transform=trs)

        # create a DataLoader for the ETRI3D dataset
        data_loader = DataLoader(dataset, batch_size=st.BATCH_SIZE, shuffle=train, drop_last=True, pin_memory=st.PIN_MEMORY, num_workers=st.NUM_WORKERS)

        return data_loader



# class for the NTU RGB+D 60 dataset, inheriting from PyTorch's Dataset
class NTURGBD60Dataset(Dataset):
    # initialize the dataset with directories, action labels, and optional transformations
    def __init__(self, data_dir, labels_dir, actions_file, transform=None, mode='train', subset_size=None, img_size=224):
        
        # directory where the video data is stored
        self.data_dir = data_dir
        # directory where the label files are stored
        self.labels_dir = labels_dir
        # optional transformations to apply to the video data
        self.transform = transform
         # mode of the dataset, e.g., 'train' or 'test'
        self.mode = mode
        # optional parameter to limit the dataset to a subset of its full size
        self.subset_size = subset_size
         # size to which each frame of the video will be resized
        self.img_size = img_size
        # load the actions and their corresponding labels from the actions file
        self.actions = self.load_actions(actions_file)
        # load the video data paths and their corresponding labels
        self.data, self.labels = self.load_data()
        
        # Define transformations for training mode
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # Randomly flip the video horizontally
                VideoRandomHorizontalFlip(),
                # Randomly crop the video
                VideoRandomCrop((img_size, img_size)),
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Optionally convert the video to grayscale
                VideoGrayscale(num_output_channels=3),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])
        else:
            # Define transformations for evaluation mode
            self.transform = transforms.Compose([
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])

        print(f"Dataset initialized with {len(self.data)} samples.")

    
    # load the action labels from the specified Excel file
    def load_actions(self, actions_file):
        # read the actions from the Excel file into a DataFrame
        actions_df = pd.read_excel(actions_file)
        # create a dictionary mapping each action label (starting at 0) to its action name
        actions_dict = dict(zip(actions_df['Label'] - 1, actions_df['Action']))  # Adjust labels to start at 0
        # print the dictionary of actions for debugging purposes
        print("\n\nActions dict", actions_dict,"\n\n")
        # print the dictionary of actions for debugging purposes
        return actions_dict

    # load video paths and corresponding labels
    def load_data(self):
        # list to store the paths of the videos
        data = []
        # list to store the labels corresponding to each video
        labels = []
        # iterate through all files in the data directory with a progress bar
        for file_name in tqdm(os.listdir(self.data_dir), desc="Loading videos"):
            # check if the file is a video file
            if file_name.endswith('.mp4') or file_name.endswith('.avi'):
                # get the full path to the video file
                video_path = os.path.join(self.data_dir, file_name)
                 # generate the corresponding label file name
                label_file = os.path.splitext(file_name)[0] + '.txt'
                # get the full path to the label file
                label_path = os.path.join(self.labels_dir, label_file)
                # check if the label file exists
                if os.path.exists(label_path):
                    # open the label file and read the label
                    with open(label_path, 'r') as f:
                         # convert the label to an integer
                        label_id = int(f.read().strip())
                        # append the video path to the data list
                        data.append(video_path)
                        # append the label to the labels list
                        labels.append(label_id)
        # return the lists of video paths and labels
        return data, labels

    # return the total number of samples in the dataset
    def __len__(self):
        # return the length of the data list
        return len(self.data)

    # get a video and its corresponding label by index
    def __getitem__(self, idx):
        # retrieve the path to the video at the given index
        video_path = self.data[idx]
         # load and preprocess the video at the given path
        video = self.load_video(video_path, self.img_size)
        # apply transformations if specified and if the video is not already a tensor
        if self.transform and not isinstance(video, torch.Tensor):
            video = self.transform(video)
        # retrieve the label corresponding to the video
        label = self.labels[idx]
        # return the video and its label as a tuple
        return video, label

    # load and preprocess the video from the given path
    def load_video(self, video_path, img_size):
        # open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        # check if the video file was opened successfully
        if not cap.isOpened():
          # raise an error if the video could not be opened
            raise ValueError(f"Error opening video stream or file: {video_path}")
        frames = []
        # list to store the frames of the video
        while cap.isOpened():
            # loop until the video is fully read
            ret, frame = cap.read()
            # read the next frame from the video
            if not ret:
                # break the loop if no more frames are returned
                break
            # convert the frame from BGR to RGB color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize the frame to the specified image size
            frame = cv2.resize(frame, (img_size, img_size))
            # append the processed frame to the list of frames
            frames.append(frame)
        # release the video capture object
        cap.release()
        # convert the list of frames to a tensor
        frames_tensor = torch.tensor(np.array(frames))
        # return the tensor containing the video frames
        return frames_tensor

   # get the action name corresponding to the given label_id
    def get_action_name(self, label_id):
       # retrieve the action name from the dictionary, defaulting to "Unknown action" if not found
        action_name = self.actions.get(label_id, "Unknown action")
        # return the action name
        return action_name

# class for the PKU-MMD Phase 1 dataset, inheriting from PyTorch's Dataset
class PKUMMDPhase1Dataset(Dataset):
  # class for the PKU-MMD Phase 1 dataset, inheriting from PyTorch's Dataset
    def __init__(self, data_dir, labels_dir, actions_file, transform=None, mode='train', subset_size=None, img_size=224):
        # directory where the video data is stored
        self.data_dir = data_dir
        # directory where the label files are stored
        self.labels_dir = labels_dir
        # optional transformations to apply to the video data
        self.transform = transform
        # mode of the dataset, e.g., 'train' or 'test'
        self.mode = mode
        # optional parameter to limit the dataset to a subset of its full size
        self.subset_size = subset_size
        # size to which each frame of the video will be resized
        self.img_size = img_size
        # load the actions and their corresponding labels from the actions file
        self.actions = self.load_actions(actions_file)
        # load the video data paths and their corresponding labels
        self.data, self.labels = self.load_data()
        # Define transformations for training mode
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # Randomly flip the video horizontally
                VideoRandomHorizontalFlip(),
                # Randomly crop the video
                VideoRandomCrop((img_size, img_size)),
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Optionally convert the video to grayscale
                VideoGrayscale(num_output_channels=3),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])
        else:
            # Define transformations for evaluation mode
            self.transform = transforms.Compose([
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])

        print(f"Dataset initialized with {len(self.data)} samples.")

    # load the action labels from the specified Excel file
    def load_actions(self, actions_file):
        # load the action labels from the specified Excel file
        actions_df = pd.read_excel(actions_file)
        # create a dictionary mapping each action label (starting at 0) to its action name
        actions_dict = dict(zip(actions_df['Label'] - 1, actions_df['Action']))  # Ajuster les labels pour commencer à 0
        # return the dictionary of actions
        return actions_dict

    # load video paths and their corresponding labels with start and end frames
    def load_data(self):
      # list to store the paths of the videos
        data = []
        # list to store the labels corresponding to each video segment
        labels = []
        # counter to track the number of loaded samples
        count = 0
        # iterate through all files in the data directory with a progress bar
        for file_name in tqdm(os.listdir(self.data_dir), desc="Chargement des vidéos"):
            # check if the file is a video file
            if file_name.endswith('.mp4') or file_name.endswith('.avi'):
              # split the file name into base name and view extension
                base_name, view_ext = file_name.rsplit('-', 1)
                # further split the view extension into view and file extension
                view, ext = view_ext.split('.')
                # get the full path to the video file
                video_path = os.path.join(self.data_dir, file_name)
                # check if the video file exists
                if os.path.exists(video_path):
                  # generate the corresponding label file name
                    label_file = file_name.replace('.avi', '.txt').replace('.mp4', '.txt')
                     # get the full path to the label file
                    label_path = os.path.join(self.labels_dir, label_file)
                    # check if the label file exists
                    if os.path.exists(label_path):
                       # open the label file and read the labels
                        with open(label_path, 'r') as f:
                            # iterate through each line in the label file
                            for line in f:
                                # split the line into its components
                                parts = line.strip().split(',')
                                # check if the line contains the expected number of parts
                                if len(parts) == 4:
                                    # extract the label id, start frame, and end frame
                                    label_id, start_frame, end_frame, _ = parts
                                    try:
                                        # convert the label id to an integer and adjust it to start at 0
                                        label_id = int(label_id) - 1  # Ajuster le label_id pour commencer à 0
                                        # convert the start frame to an integer
                                        start_frame = int(start_frame)
                                        # convert the end frame to an integer
                                        end_frame = int(end_frame)
                                    # handle any conversion errors
                                    except ValueError:
                                        continue
                                    # append the video path and frame range to the data list
                                    data.append((video_path, start_frame, end_frame))
                                    # append the label to the labels list
                                    labels.append(label_id)
                                    # increment the sample counter
                                    count += 1
                                    if self.subset_size and count >= self.subset_size:
                                        # if a subset size is specified and reached, return the data
                                        return data, labels

        # return the lists of video paths and labels
        return data, labels

    # return the total number of samples in the dataset
    def __len__(self):
      # return the length of the data list
        return len(self.data)

    #ethod to get a video segment and its corresponding label by index
    def __getitem__(self, idx):
        # retrieve the video information (path, start frame, end frame) at the given index
        video_info = self.data[idx]
        # unpack the video information
        video_path, start_frame, end_frame = video_info
        # load and preprocess the video segment
        video = self.load_video(video_info, self.img_size)
        # apply transformations if specified and if the video is not already a tensor
        if self.transform and not isinstance(video, torch.Tensor):
            video = self.transform(video)
        # retrieve the label corresponding to the video segment
        label = self.labels[idx]
        # return the video and its label as a tuple
        return video, label

    #method to load and preprocess a video segment from the given start to end frame
    def load_video(self, video_info, img_size):

        # unpack the video information
        video_path, start_frame, end_frame = video_info
        # print the video information for debugging purposes
        print("video_info", video_info)
        # open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        # check if the video file was opened successfully
        if not cap.isOpened():
            # raise an error if the video could not be opened
            raise ValueError(f"Error opening video stream or file: {video_path}")
        # list to store the frames of the video segment
        frames = []
        # counter to track the current frame number
        current_frame = 0
        # loop until the video is fully read
        while cap.isOpened():
            # read the next frame from the video
            ret, frame = cap.read()
            # break the loop if no more frames are returned or the end frame is reached
            if not ret or current_frame > end_frame:
                break
            # process the frame if it is within the specified range
            if current_frame >= start_frame:
                # convert the frame from BGR to RGB color space
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # resize the frame to the specified image size
                frame = cv2.resize(frame, (img_size, img_size))
                # append the processed frame to the list of frames
                frames.append(frame)
            # increment the frame counter
            current_frame += 1
        # release the video capture object
        cap.release()
        # convert the list of frames to a tensor
        frames_tensor = torch.tensor(np.array(frames))
        # return the tensor containing the video frames
        return frames_tensor
    
    #method to get the action name corresponding to the given label_id
    def get_action_name(self, label_id):
        # retrieve the action name from the dictionary, defaulting to "Unknown action" if not found
        action_name = self.actions.get(label_id, "Unknown action")
        # return the action name
        return action_name


# class for the PKU-MMD Phase 2 dataset, inheriting from PyTorch's Dataset
class PKUMMDPhase2Dataset(Dataset):
    # initialize the dataset with directories for data and labels, action labels file, and optional transformations
    def __init__(self, data_dir, labels_dir, actions_file, transform=None, mode='train', subset_size=None, img_size=224):
        # directory where the video data is stored
        self.data_dir = data_dir
        # directory where the label files are stored
        self.labels_dir = labels_dir
        # optional transformations to apply to the video data
        self.transform = transform
        # mode of the dataset, e.g., 'train' or 'test'
        self.mode = mode
        # optional parameter to limit the dataset to a subset of its full size
        self.subset_size = subset_size
        # size to which each frame of the video will be resized
        self.img_size = img_size
        # load the actions and their corresponding labels from the actions file
        self.actions = self.load_actions(actions_file)
        # load the video data paths and their corresponding labels
        self.data, self.labels = self.load_data()
        # Define transformations for training mode
        if self.mode == 'train':
            self.transform = transforms.Compose([
                # Randomly flip the video horizontally
                VideoRandomHorizontalFlip(),
                # Randomly crop the video
                VideoRandomCrop((img_size, img_size)),
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Optionally convert the video to grayscale
                VideoGrayscale(num_output_channels=3),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])
        else:
            # Define transformations for evaluation mode
            self.transform = transforms.Compose([
                # Resize the video to the specified image size
                VideoResize((img_size, img_size)),
                # Convert video file to tensor
                VideoFilePathToTensor(max_len=cfg.FPS),
            ])

        print(f"Dataset initialized with {len(self.data)} samples.")

    #method to load the action labels from the specified Excel file
    def load_actions(self, actions_file):
       # read the actions from the Excel file into a DataFrame
        actions_df = pd.read_excel(actions_file)
        # create a dictionary mapping each action label (starting at 0) to its action name
        actions_dict = dict(zip(actions_df['Label'] - 1, actions_df['Action']))  # Adjust labels to start from 0
       # return the dictionary of actions
        return actions_dict

    # load video paths and their corresponding labels with start and end frames
    def load_data(self):
        # list to store the paths of the videos
        data = []
        # list to store the labels corresponding to each video segment
        labels = []
         # counter to track the number of loaded samples
        count = 0
        # iterate through all files in the data directory with a progress bar
        for file_name in tqdm(os.listdir(self.data_dir), desc="Chargement des vidéos"):
            # iterate through all files in the data directory with a progress bar
            if file_name.endswith('.mp4') or file_name.endswith('.avi'):
              # split the file name into base name and view extension
                base_name, view_ext = file_name.rsplit('-', 1)
                # further split the view extension into view and file extension
                view, ext = view_ext.split('.')
                 # get the full path to the video file
                video_path = os.path.join(self.data_dir, file_name)
                # check if the video file exists
                if os.path.exists(video_path):
                    # generate the corresponding label file name
                    label_file = file_name.replace('_color.avi', '.txt').replace('_color.mp4', '.txt')
                    # get the full path to the label file
                    label_path = os.path.join(self.labels_dir, label_file)
                    # check if the label file exists
                    if os.path.exists(label_path):
                        # iterate through each line in the label file
                        print(f"Label file found: {label_path}")
                        # print a message indicating that the label file was found
                        with open(label_path, 'r') as f:
                            # split the line into its components
                            for line in f:
                                # check if the line contains the expected number of parts
                                parts = line.strip().split(',')
                                # extract the label id, start frame, and end frame
                                if len(parts) == 4:
                                    label_id, start_frame, end_frame, _ = parts
                                    try:
                                        # convert the label id to an integer and adjust it to start at 0
                                        label_id = int(label_id) - 1  # Adjust label_id to start from 0
                                        # convert the start frame to an integer
                                        start_frame = int(start_frame)
                                        # convert the end frame to an integer
                                        end_frame = int(end_frame)
                                        # check if the label id is within the valid range
                                        if label_id < 0 or label_id >= cfg.NUM_CLASSES:
                                            # check if the label id is within the valid range
                                            print(f"Skipping invalid class index {label_id} in line: {line.strip()}")
                                            # print a message and skip if the label id is invalid
                                            continue
                                    # handle any conversion errors
                                    except ValueError:
                                      # print a message indicating the line is invalid
                                        print(f"Invalid label line: {line}")
                                        continue
                                    # append the video path and frame range to the data list
                                    data.append((video_path, start_frame, end_frame))
                                    # append the label to the labels list
                                    labels.append(label_id)
                                    # increment the sample counter
                                    count += 1
                                    # if a subset size is specified and reached, return the data
                                    if self.subset_size and count >= self.subset_size:
                                        # print a message indicating the subset size limit was reached
                                        print(f"Reached subset size: {self.subset_size}")
                                        return data, labels
                    else:
                        # print a message indicating the label file was not found
                        print(f"Label file not found: {label_path}")
                else:
                    # print a message indicating the video file was not found
                    print(f"Video file not found: {video_path}")
        # return the lists of video paths and labels
        return data, labels

    # return the total number of samples in the dataset
    def __len__(self):
        # return the length of the data list
        return len(self.data)

    # method to get a video segment and its corresponding label by index
    def __getitem__(self, idx):
        # retrieve the video information (path, start frame, end frame) at the given index
        video_info = self.data[idx]
        # unpack the video information
        video_path, start_frame, end_frame = video_info
        # load and preprocess the video segment
        video = self.load_video(video_info, self.img_size)
        # if the video is empty, return a dummy video
        if video.size(0) == 0:  
           # create a tensor of zeros as a dummy video
            video = torch.zeros((1, self.img_size, self.img_size, 3))
        # apply transformations if specified and if the video is not already a tensor
        if self.transform and not isinstance(video, torch.Tensor):          
            video = self.transform(video)
        # retrieve the label corresponding to the video segment
        label = self.labels[idx]
        # return the video and its label as a tuple
        return video, label

    # method to load and preprocess a video segment from the given start to end frame
    def load_video(self, video_info, img_size):
        # unpack the video information
        video_path, start_frame, end_frame = video_info
        # print the video information for debugging purposes
        print("video_info", video_info)
        # open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        # check if the video file was opened successfully
        if not cap.isOpened():
            # raise an error if the video could not be opened
            raise ValueError(f"Error opening video stream or file: {video_path}")
        # list to store the frames of the video segment
        frames = []
        # counter to track the current frame number
        current_frame = 0
        # loop until the video is fully read
        while cap.isOpened():
            # read the next frame from the video
            ret, frame = cap.read()
            # break the loop if no more frames are returned or the end frame is reached
            if not ret or current_frame > end_frame:
                break
            # process the frame if it is within the specified range
            if current_frame >= start_frame:
                # convert the frame from BGR to RGB color space
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # resize the frame to the specified image size
                frame = cv2.resize(frame, (img_size, img_size))
                # append the processed frame to the list of frames
                frames.append(frame)
            # increment the frame counter
            current_frame += 1
        # release the video capture object
        cap.release()
        # convert the list of frames to a tensor
        frames_tensor = torch.tensor(np.array(frames))
        # return the tensor containing the video frames
        return frames_tensor

    # method to get the action name corresponding to the given label_id
    def get_action_name(self, label_id):
        # retrieve the action name from the dictionary, defaulting to "Unknown action" if not found
        action_name = self.actions.get(label_id, "Unknown action")
        # return the action name
        return action_name