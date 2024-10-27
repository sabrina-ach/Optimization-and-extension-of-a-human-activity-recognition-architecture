import config_ as cfg # import the configuration settings
import torch # import necessary libraries from PyTorch
from torch import nn, einsum
import utils.utils as utils # import utility functions
from utils.mlp import Mlp # import MLP class
from timm.models.layers import trunc_normal_, DropPath, to_2tuple # import specific functions and classes from the timm library for model layers
# import various attention mechanisms
from self_attention_mechanisms import FSAttention, ScaledDotProductAttention, MultiHeadAttention, RelativePositionalEncodingAttention, FullAttention, ZigzagAttention, BinaryAttention

# define the SpatioTempTransformer class inheriting from nn.Module
class SpatioTempTransformer(nn.Module):
    # initialize the transformer with specified dimensions, number of heads, and attention type
    def __init__(self, dim, num_heads, attention_type='fsattention', mlp_ratio=cfg.MLP_RATIO, drop=cfg.DROP_OUT, drop_path=cfg.DROP_PATH, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # call the parent class's constructor
        super().__init__()

        # set the attention mechanism based on the attention_type argument
        if attention_type == 'fsattention':
            self.attn = FSAttention(dim, num_heads)
        elif attention_type == 'scaled_dot_product':
            self.attn = ScaledDotProductAttention(dim, num_heads)
        elif attention_type == 'multihead':
            self.attn = MultiHeadAttention(dim, num_heads)
        elif attention_type == 'relative':
            self.attn = RelativePositionalEncodingAttention(dim, num_heads)
        elif attention_type == 'full':
            self.attn = FullAttention(dim, num_heads)
        elif attention_type == 'zigzag':
            self.attn = ZigzagAttention(dim, num_heads)
        elif attention_type == 'binary':
            self.attn = BinaryAttention(dim, num_heads)
        else:
            # raise an error if an unknown attention type is provided
            raise ValueError(f"Unknown attention type: {attention_type}")

        # define a 3x3x3 convolutional layer for positional embedding
        self.pos_embed = utils.conv_3x3x3(dim, dim, groups=dim)
        # define a normalization layer
        self.norm1 = norm_layer(dim)
        # define a stochastic depth layer (DropPath) or use identity if drop_path is zero
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # define another normalization layer
        self.norm2 = norm_layer(dim)
        # calculate the hidden dimension for the MLP layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        # define the MLP layer with the specified activation and dropout layers
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # forward pass of the SpatioTempTransformer
    def forward(self, x):
        # apply positional embedding to the input and add it to the input
        x = x + self.pos_embed(x)
        # get the shape of the input tensor
        B, C, T, H, W = x.shape
        # flatten the spatial dimensions and transpose for attention
        x = x.flatten(2).transpose(1, 2)
        # apply attention and DropPath, then add to the input
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # split the batch into chunks
        x = x.chunk(B, dim=0)
        # reintroduce batch dimension by stacking the chunks
        x = [temp[None] for temp in x]
        # concatenate the chunks and transpose back
        x = torch.cat(x, dim=0).transpose(1, 2)
        # flatten the temporal and spatial dimensions
        x = torch.flatten(x, start_dim=0, end_dim=1)
        # apply attention and DropPath, then add to the input
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # apply the MLP, DropPath, and add to the input
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # reshape the tensor back to its original shape
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        # return the final output
        return x
