import torch
from torch import nn, einsum
import utils.utils as utils
from utils.mlp import CMlp
import config_ as st
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

class SpatialCNN(nn.Module):
    """ A spatial bloc based on 3D Depth-Wise CNN """
    def __init__(self, dim, mlp_ratio=st.MLP_RATIO, drop=st.DROP_OUT,
                 drop_path=st.DROP_PATH, act_layer=nn.GELU):

        super().__init__()
        self.pos_embed = utils.conv_3x3x3(dim, dim, groups=dim) 
        self.norm1 = utils.bn_3d(dim)
        self.conv1 = utils.conv_1x1x1(dim, dim, 1)
        self.conv2 = utils.conv_1x1x1(dim, dim, 1)
        self.attn = utils.conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = utils.bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   