import torch  # import the PyTorch library for tensor operations and deep learning
import torch.nn as nn  # import the neural network module from PyTorch for building and training models
import torchvision  # import the torchvision library for computer vision tasks
from torchvision import transforms  # import the transforms module from torchvision
from torchvision.transforms import Compose  # import the Compose class from transforms
import os  # import the os module for interacting with the operating system
import time  # import the time module for time-related functions
import pandas as pd  # import the pandas library for data manipulation and analysis
from data import dataset as db  # import the dataset module from the data package and alias it as db
from data import transform as ts  # import the transform module from the data package and alias it as ts
import config_ as cfg  # import the config_ file and alias it as cfg
from einops import rearrange  # import the rearrange function from einops
from tqdm import tqdm  # import the tqdm library for displaying progress bars
from torch.cuda.amp import GradScaler, autocast  # import GradScaler and autocast from torch.cuda.amp for mixed precision training


#class for training and evaluating a PyTorch model
class TrainAndEvaluate:
    def __init__(self, model):
        #initialize the class with the given model
        self.model = model
        #initialize GradScaler for mixed precision training
        self.scaler = GradScaler()  


    #training function to train the model on a training set
    def train(self, data_loader, lr, weight_decay, intermediate_result_step, print_epoch_result_step):
        #initialize AdamW optimizer with the provided learning rate and weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        #initialize a learning rate scheduler (e.g., StepLR, ReduceLROnPlateau, etc.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        #use CrossEntropyLoss instead of NLLLoss
        criterion = nn.CrossEntropyLoss()  
        #move the criterion to the configured device 
        criterion.to(cfg.DEVICE)

        #set the model to training mode
        self.model.train()

        #loop over the specified number of epochs
        for epoch in range(1, cfg.NB_EPOCHES + 1):
            epoch_loss = 0
            epoch_accuracy = 0
            start_time = time.time()

            print("==========================================================")
            #create a progress bar to track the epoch's progress
            with tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{cfg.NB_EPOCHES}", unit='batch') as pbar:
              #iterate over the data loader
                for i, (data, target) in enumerate(data_loader):
                    #move the input data and targets to the configured device
                    data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                    
                    #check if the target class indices are within the valid range
                    if target.max() >= cfg.NUM_CLASSES or target.min() < 0:
                        raise ValueError(f"Class index out of bounds: {target}")

                    #zero the gradients of the optimizer
                    optimizer.zero_grad()

                    #use autocast for mixed precision computations
                    with autocast():
                       #forward pass through the model
                        outputs = self.model(data)
                        #compute the loss
                        loss = criterion(outputs, target)

                    #scale the gradient
                    self.scaler.scale(loss).backward()
                    #apply gradient clipping 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    #optimizer step
                    self.scaler.step(optimizer)
                    #update the scaler
                    self.scaler.update()
                    
                    #accumulate the loss for the current epoch
                    epoch_loss += loss.item()
                    #get the predictions by selecting the class with the highest score
                    _, preds = torch.max(outputs, 1)
                    #accumulate the number of correct predictions for accuracy calculation
                    epoch_accuracy += (preds == target).sum().item()

                    #update the progress bar with the current loss and accuracy
                    pbar.set_postfix(loss=epoch_loss/(i+1), accuracy=100.*epoch_accuracy/((i+1)*data_loader.batch_size))
                    pbar.update(1)

                    #print intermediate results at specified steps"
                    if i % intermediate_result_step == 0 and i != 0:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        print(f"Step {i}, Loss: {epoch_loss/(i+1):.4f}, Accuracy: {100.*epoch_accuracy/((i+1)*data_loader.batch_size):.2f}%, Elapsed Time: {elapsed_time:.2f} seconds")
                        print(f"Labels: {target}")
                        print(f"Predictions: {preds} \n")

            #step the scheduler at the end of each epoch
            scheduler.step()
            
            end_time = time.time()
            epoch_duration = end_time - start_time
            print("***************** \n")
            #print the results for the completed epoch
            print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds. Loss: {epoch_loss/len(data_loader):.4f}, Accuracy: {100.*epoch_accuracy/len(data_loader.dataset):.2f}%")

    #save the model's state dictionary to the specified path       
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

   #method to test the model's performance on a test dataset
    def test(self, data_loader, intermediate_result_step):
        #set the model to evaluation mode
        self.model.eval()
        eval_accuracy = 0
        start_time = time.time()
        #disable gradient calculation since we are in inference mode
        with torch.no_grad():
            #iterate over the data loader
            for i, (data, target) in enumerate(data_loader):
                #move the input data and targets to the configured device
                data, target = data.to(cfg.DEVICE), target.to(cfg.DEVICE)
                #forward pass through the model
                outputs = self.model(data)
                #get the predictions by selecting the class with the highest score
                _, preds = torch.max(outputs, 1)
                #accumulate the number of correct predictions for accuracy calculation
                eval_accuracy += (preds == target).sum().item()

                #print intermediate results at specified steps
                if i % intermediate_result_step == 0 and i != 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    print("*****************")
                    print(f"Step {i}, Accuracy: {100.*eval_accuracy/((i+1)*data_loader.batch_size):.2f}%, Elapsed Time: {elapsed_time:.2f} seconds")

        #record the current time to mark the end of the test
        end_time = time.time()
        #calculate the total duration of the test by subtracting the start time from the end time
        test_duration = end_time - start_time
        #print the final test results
        print(f"Test completed in {test_duration:.2f} seconds. Accuracy: {100.*eval_accuracy/len(data_loader.dataset):.2f}% \n")
