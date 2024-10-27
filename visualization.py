import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose
from einops import rearrange
import os
import time
import pandas as pd
from data import new_datasets as db
from data import transform as ts
import config_ as st
from matplotlib import rc
rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2

class Visualization:
    """ Visualization.
        
        Args:
            model (Model): The model on which the visualization will be done.
    """
    def __init__(self, model):
        self.model = model
        self.activation3 = []
        self.activation1 = []
        self.activation2 = []

        self.hooks_exist = False
        self.done_forward = False
    
    def plot_loss(self, loss_train):
        """ Plot the training loss curve.
            
        Args:
            loss_train (List): The loss list returned by TrainAndEvaluate.train
        """
        plt.plot(loss_train)
        plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['loss'], loc='upper left')
        plt.xticks(np.arange(1, len(loss_train) + 1, 2))
        plt.yticks(np.arange(min(loss_train), max(loss_train) + 1, 0.5))
        plt.gcf().set_size_inches(10, 10)
        plt.show()
    
    def set_hooks(self):
        """ Set the hooks to register intermediate outputs """
        try:
            # Enregistre les hooks pour les blocs de CNN spatiaux
            [m.register_forward_hook(lambda m, input, output: self.activation1.append(output.detach())) for m in self.model.blocks1]
            [m.register_forward_hook(lambda m, input, output: self.activation2.append(output.detach())) for m in self.model.blocks2]
            
            # Enregistre un hook pour le module d'attention dans finalBlock
            # Vérifie si le module d'attention existe et s'il est un sous-module
            if hasattr(self.model.finalBlock, 'attn'):
                self.model.finalBlock.attn.register_forward_hook(lambda m, input, output: self.activation3.append(output.detach()))
            else:
                print("Attention module not found in finalBlock. Hooks not set for attention.")
            
            self.hooks_exist = True
        except Exception as e:
            print(f"Erreur lors de la définition des hooks: {e}")
    
    def perform_forward(self, video):
        """ Perform a forward pass of the video through the model.
        
        NOTE : This method must be called after set_hooks 
            
        Args:
            video (Tensor): A video tensor from a data loader
        """
        try:
            x = video.unsqueeze(0)
            x = rearrange(x, 'b t w h c -> b c t w h')
            self.model.eval()
            self.model(x.to(st.DEVICE).type(torch.cuda.FloatTensor))
            self.done_forward = True
        except Exception as e:
            print(f"Erreur lors de la propagation avant: {e}")

    def plot_video(self, video):
        """ Plot the test video
            
        Args:
            video (Tensor): A video tensor from a data loader
        """
        try:
            x = video.unsqueeze(0)
            x = rearrange(x, 'b t w h c -> b c t w h')
            ani = self.__matplotlib_imshow_animate(x[0])
            return ani
        except Exception as e:
            print(f"Erreur lors de l'affichage de la vidéo: {e}")

    def plot_first_cnn_layer(self):
        """ Plot the first CNN layer output
        
        NOTE : set_hooks and perform_forward must be called before this method
        """
        try:
            if self.done_forward and self.hooks_exist:
                activations = self.activation1[0][0].cpu().numpy()
                fig, axes = plt.subplots(1, len(activations), figsize=(20, 20))
                for i, ax in enumerate(axes):
                    ax.imshow(activations[i], cmap='viridis')
                    ax.axis('off')
                plt.show()
        except Exception as e:
            print(f"Erreur lors de l'affichage de la première couche CNN: {e}")
  
    def plot_spatial_bloc_output(self):
        """ Plot the spatial bloc output
        
        NOTE : set_hooks and perform_forward must be called before this method
        """
        try:
            if self.done_forward and self.hooks_exist:
                ani = self.__matplotlib_imshow_animate(self.activation2[3][0].cpu(), three_channels=False)
                return ani
        except Exception as e:
            print(f"Erreur lors de l'affichage du bloc spatial: {e}")
        
    def plot_attention(self, video):

        """  
        Plot the attention effect as a heatmap applied to the original video
        
        Args:
            video (Tensor): The original video
        """

        try:
            fig, ax = plt.subplots()
            v = video.permute(1, 2, 3, 0)

            # Check the shape of the attention tensor
            attn = self.activation3[0].cpu()
            print(f"Forme de la tensor d'attention avant reshape: {attn.shape}")

            # Determine the correct size for reshape
            reshaped_size = attn.numel() // (attn.size(0) * attn.size(-2) * attn.size(-1))
            reshaped_size = (attn.size(0), reshaped_size, attn.size(-2), attn.size(-1))
            attn = attn.view(*reshaped_size)
            print(f"Forme de la tensor d'attention après reshape: {attn.shape}")

            imgs = []

            for i in range(attn.size(1)):
                # Original image
                img = v[i].numpy()
                f = img.max() - img.min()
                img -= img.min()
                img /= f
                img *= 255
                img = img.astype('uint8')

                # Attention
                out = self.__visualize_attn_feature_map(attn[:, i, :, :])
                out = self.__apply_heatmap(out, img)
                imgs.append(out)

            frames = [[ax.imshow(imgs[i], cmap=plt.cm.afmhot)] for i in range(len(imgs) - 1)]

            ani = animation.ArtistAnimation(fig, frames, interval=500)
            return ani

        except Exception as e:
            print(f"Erreur lors de l'affichage de l'attention: {e}")

    def __adjust_image(self, to_adjust_img):
        """ Adjust the pixel values to better visualize an intermediate output. 
            The idea is to amplify the values above 45% of the max value and set the ones below this value to 0
            
        Args:
            to_adjust_img (Tensor): A frame of a video tensor
        """
        mx = torch.max(to_adjust_img).item()
        img = to_adjust_img.numpy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] < 0.35 * mx:
                    img[i][j] = 0
                if img[i][j] > 0.45 * mx:
                    img[i][j] = 1
        return img

    def __matplotlib_imshow_animate(self, img, three_channels=True):
        """ Plot a video
            
        Args:
            img (Tensor): A video
            three_channels (Bool): True means that it's a video else it's an intermediate output
        """
        try:
            fig, ax = plt.subplots()
            if not three_channels:
                imgs = img.mean(dim=0)
                l = imgs.size()[0]
                for i in range(l):
                    self.__adjust_image(imgs[i])
            else:
                imgs = img.permute(1, 2, 3, 0)  # Permuting to (Bx)HxWxC format

            frames = [[ax.imshow(imgs[i], cmap=plt.cm.afmhot)] for i in range(len(imgs) - 1)]
            ani = animation.ArtistAnimation(fig, frames, interval=300)
            plt.show()
            return ani
        except Exception as e:
            print(f"Erreur lors de l'animation des images: {e}")

    def __apply_heatmap(self, weights, img):
        """ Create and apply a heatmap to an image using the attention weights
            
        Args:
            weights (Tensor): A one-channel heatmap weight (output of __visualize_attn_feature_map)
            img (Tensor): A three-channel image
        """
        try:
            img = cv2.resize(img, (100, 100))
            heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
            return heatmap
        except Exception as e:
            print(f"Erreur lors de l'application de la carte thermique: {e}")

    def __visualize_attn_feature_map(self, act):
        """ Create an image from the attention weights.
        
        Args:
            act (Tensor): The attention activation weights
        """
        try:
            xx = act
            f_output = []
            for i in range(xx.size(0)):
                imgs = xx[i]
                output = np.abs(imgs.numpy())
                output = cv2.resize(output, (100, 100))
                f = output.max() - output.min()
                output -= output.min()
                output /= f
                output *= 255
                f_output.append(output)
            f_output = sum(f_output)
            f = f_output.max() - f_output.min()
            f_output -= f_output.min()
            f_output /= f
            f_output *= 255
            f_output = 255 - f_output
            return f_output.astype('uint8')
        except Exception as e:
            print(f"Erreur lors de la visualisation de la carte des caractéristiques d'attention: {e}")
