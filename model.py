import imp  # import the imp module (though note that 'imp' is deprecated in favor of 'importlib')
from architecture_blocks.patch_embedding import SpeicalPatchEmbed, PatchEmbed  # import custom modules for various model components
from architecture_blocks.spatial_cnn import SpatialCNN  # import custom module for spatial CNN
from architecture_blocks.spatio_temp_transformer import SpatioTempTransformer  # import custom module for spatio-temporal transformer
import utils.utils as utils  # import utility functions from the utils module
import config_ as st  # import the configuration file and alias it as st
from functools import partial  # import utilities from the functools library
from collections import OrderedDict  # import utilities from the collections library
from timm.models.layers import trunc_normal_, DropPath, to_2tuple  # import specific functions and classes from the timm library for model layers
import torch  # import PyTorch
from torch import nn, einsum  # import submodules from PyTorch
from einops import rearrange  # import tensor manipulation functions from einops
from einops.layers.torch import Reduce  # import the Reduce layer from einops for PyTorch


# define the Model class inheriting from nn.Module
class Model(nn.Module):
    """ A Novel CNN-Transformer Architecture for HAR
    """
    # initialize the model with the number of classes and the type of attention mechanism
    def __init__(self, NUM_CLASSES, attention_type='fsattention'):
        # call the parent class's constructor
        super().__init__()

        # define the depth of each block in the model
        depth = [3, 4, 8, 3]
        # set the number of classes
        num_classes = NUM_CLASSES 
        # get image size and other configurations from settings
        img_size = st.IMG_SIZE
        in_chans = 3
        embed_dim = st.BLOCKS_DIM
        head_dim = st.BLOCKS_DIM[0]
        mlp_ratio = st.MLP_RATIO
        drop_rate = st.DROP_OUT
        attn_drop_rate = st.DROP_OUT
        drop_path_rate = st.DROP_PATH
        std = False

        # set the number of classes and features
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # define the normalization layer using LayerNorm
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        print("Initializing Patch Embedding Layers \n")
        # initialize the first patch embedding layer with special settings
        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        # initialize subsequent patch embedding layers
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        # define a dropout layer for the positionally embedded patches
        self.pos_drop = nn.Dropout(p=drop_rate)
        # calculate the stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        # calculate the number of heads for multi-head attention
        num_heads = [dim // head_dim for dim in embed_dim]
        
        print("Initializing Spatial CNN Blocks \n")
        # initialize the first set of spatial CNN blocks
        self.blocks1 = nn.ModuleList([
            SpatialCNN(
                dim=embed_dim[0], mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i])
            for i in range(depth[0])])
        # initialize the second set of spatial CNN blocks
        self.blocks2 = nn.ModuleList([
            SpatialCNN(
                dim=embed_dim[1], mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i+depth[0]])
            for i in range(depth[1])])

        # apply batch normalization for the final spatial dimension
        self.norm = utils.bn_3d(embed_dim[-1])

        print("Initializing Final Spatio-Temporal Transformer Block \n")
        # initialize the final spatio-temporal transformer block
        self.finalBlock = SpatioTempTransformer(
            dim=embed_dim[3], num_heads=num_heads[3], attention_type=attention_type,
            mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[depth[0]+depth[1]+depth[2]], 
            norm_layer=norm_layer)
        
        last_dim = embed_dim[3]
        self.pre_logits = nn.Identity()
        
        print("Initializing Classifier Head \n")
        # initialize the classifier head with a sequence of layers including normalization, linear layers, and dropout
        self.head = nn.Sequential(
            Reduce('b c t h w -> b c', 'mean'),  # reduce the spatial and temporal dimensions by averaging
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, last_dim*10),
            nn.ReLU(),
            nn.Linear(last_dim*10, last_dim*128),
            nn.LayerNorm(last_dim*128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(last_dim*128, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        # apply weight initialization to the model
        self.apply(self._init_weights)

        print("Initializing Weights \n")
        # custom weight initialization for specific layers in the model
        for name, p in self.named_parameters():
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    # method to initialize weights for different layers
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    # method to specify layers that should not apply weight decay
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # method to retrieve the classifier head
    def get_classifier(self):
        return self.head

    # method to reset the classifier with a new number of classes
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            
    # method to compute forward features up to the final block
    def forward_features(self, x):
        # apply the first patch embedding
        x = self.patch_embed1(x)
        # apply dropout after positional embedding
        x = self.pos_drop(x)
        
        # pass through the first set of CNN blocks
        for i, blk in enumerate(self.blocks1):
            x = blk(x)

        # apply the second patch embedding
        x = self.patch_embed2(x)
        
        # pass through the second set of CNN blocks
        for i, blk in enumerate(self.blocks2):
            x = blk(x)
       
        # apply the third patch embedding
        x = self.patch_embed3(x)
        
        # pass through the final spatio-temporal transformer block
        x = self.finalBlock(x)
        # apply pre-logits identity layer
        x = self.pre_logits(x)
        return x

    # method to compute the final output by applying the classifier head
    def forward(self, x):
        # compute the features
        x = self.forward_features(x)
        # apply the classifier head and return the final output
        x = self.head(x)
        return x
