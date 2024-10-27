import torch # import the PyTorch library for tensor operations
import torch.nn as nn # import the neural network module from PyTorch
from torch import nn, einsum # import modules from PyTorch for neural networks and Einstein summation
from einops import rearrange # import the rearrange function from einops for tensor manipulation

# define the FSAttention class inheriting from nn.Module
class FSAttention(nn.Module):
    """Factorized Self-Attention"""

    # initialize the factorized self-attention mechanism with model dimensions, number of heads, and dropout rate
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.7):
        super().__init__()
        # calculate the inner dimension of the attention mechanism
        inner_dim = dim_head * heads
        # determine whether to apply a linear projection to the output
        project_out = not (heads == 1 and dim_head == dim)

        # store the number of attention heads
        self.heads = heads
        # scale factor for the attention scores
        self.scale = dim_head ** -0.5

        # define a softmax layer for attention
        self.attend = nn.Softmax(dim=-1)
        # define a linear layer to project input into queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # define the output projection and dropout layer if needed, otherwise use identity
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # forward pass of the factorized self-attention mechanism
    def forward(self, x):
        # extract the batch size, sequence length, and number of heads from the input tensor shape
        b, n, _, h = *x.shape, self.heads
        # project the input into queries, keys, and values, then split them along the last dimension
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # rearrange the queries, keys, and values to the appropriate shape for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # execute re-arrange on the 3 chunks of qkv 

        # calculate the attention scores using Einstein summation
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # apply softmax to the attention scores to obtain the attention weights
        attn = self.attend(dots)

        # calculate the output by applying the attention weights to the values using Einstein summation
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # rearrange the output to merge the heads back into a single dimension
        out = rearrange(out, 'b h n d -> b n (h d)')
        # apply the final linear projection and dropout (if defined), and return the output
        return self.to_out(out)

# define the ScaledDotProductAttention class inheriting from nn.Module
class ScaledDotProductAttention(nn.Module):
    # initialize the scaled dot-product attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(ScaledDotProductAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        # calculate the depth of each head
        self.depth = d_model // num_heads
        
        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)
    
    # function to split the input into multiple heads and rearrange the dimensions
    def split_heads(self, x, batch_size):
        # reshape the input tensor to separate heads and depth
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # permute the tensor to have the shape [batch_size, num_heads, seq_len, depth]
        return x.permute(0, 2, 1, 3)
    
    # forward pass of the scaled dot-product attention mechanism
    def forward(self, q, k, v, mask=None):
        # get the batch size from the query tensor
        batch_size = q.size(0)
        
        # apply the query, key, and value projections, and split into heads
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        
        # calculate the dot product between queries and transposed keys
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        # scale the attention scores by the square root of the key dimension
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        # apply the mask to the attention scores, if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # apply softmax to obtain the attention weights
        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        # calculate the output by applying the attention weights to the values
        output = torch.matmul(attention_weights, v)
        
        # permute the output to merge heads and rearrange dimensions
        output = output.permute(0, 2, 1, 3).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = output.view(batch_size, -1, self.d_model)
        # apply the final linear layer to get the output
        output = self.fc(output)
        
        # return the final output and the attention weights
        return output, attention_weights




# define the MultiHeadAttention class inheriting from nn.Module
class MultiHeadAttention(nn.Module):
    # initialize the multi-head attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(MultiHeadAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        
        # assert that the model dimension is divisible by the number of heads
        assert d_model % num_heads == 0
        
        # calculate the depth of each head
        self.depth = d_model // num_heads
        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)
        
    # method to split the input tensor into multiple heads
    def split_heads(self, x, batch_size):
        # reshape the input tensor to separate the heads
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # permute the dimensions to bring the number of heads first
        return x.permute(0, 2, 1, 3)
    
    # forward pass of the multi-head attention mechanism
    def forward(self, q, k, v, mask=None):
        # get the batch size from the query tensor
        batch_size = q.size(0)
        
        # apply the query projection and split into heads
        q = self.split_heads(self.wq(q), batch_size)
        # apply the key projection and split into heads
        k = self.split_heads(self.wk(k), batch_size)
        # apply the value projection and split into heads
        v = self.split_heads(self.wv(v), batch_size)
        
        # apply scaled dot-product attention to the projected queries, keys, and values
        scaled_attention, _ = ScaledDotProductAttention(self.d_model, self.num_heads)(q, k, v, mask)
        # permute the dimensions to merge the heads back together
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = scaled_attention.view(batch_size, -1, self.d_model)
        
        # apply the final linear layer to get the output
        output = self.fc(output)
        # return the final output
        return output

# define the RelativePositionalEncodingAttention class inheriting from nn.Module
class RelativePositionalEncodingAttention(nn.Module):
    # initialize the attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(RelativePositionalEncodingAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        # calculate the depth of each head
        self.depth = d_model // num_heads
        
        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)
        
    # forward pass of the relative positional encoding attention mechanism
    def forward(self, q, k, v, pos):
        # get the batch size from the query tensor
        batch_size = q.size(0)
        
        # apply the query projection, reshape and transpose to split into heads
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the key projection, reshape and transpose to split into heads
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the value projection, reshape and transpose to split into heads
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        
        # calculate the relative position scores using the queries and positional encodings
        relative_position_scores = torch.matmul(q, pos.transpose(-2, -1))
        # calculate the attention scores using the queries and keys
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # add the relative position scores to the attention scores
        attention_scores += relative_position_scores
        # scale the attention scores by the square root of the depth
        attention_scores /= torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        # apply the softmax function to obtain attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        # calculate the output by weighted sum of the values and transpose back
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = output.view(batch_size, -1, self.d_model)
        
        # apply the final linear layer to get the output and return it
        return self.fc(output)

# define the FullAttention class inheriting from nn.Module
class FullAttention(nn.Module):
    # initialize the full attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(FullAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        # calculate the depth of each head
        self.depth = d_model // num_heads

        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)

    # forward pass of the full attention mechanism
    def forward(self, q, k, v):
        # get the batch size from the query tensor
        batch_size = q.size(0)

        # apply the query projection, reshape and transpose to split into heads
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the key projection, reshape and transpose to split into heads
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the value projection, reshape and transpose to split into heads
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # calculate the attention scores by performing matrix multiplication between queries and transposed keys
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # apply the softmax function to obtain attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        # calculate the output by performing matrix multiplication between attention weights and values, then transpose back
        output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = output.view(batch_size, -1, self.d_model)

        # apply the final linear layer to get the output and return it along with attention weights
        return self.fc(output), attention_weights

# define the ZigzagAttention class inheriting from nn.Module
class ZigzagAttention(nn.Module):
    # initialize the zigzag attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(ZigzagAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        # calculate the depth of each head
        self.depth = d_model // num_heads

        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)

    # forward pass of the zigzag attention mechanism
    def forward(self, q, k, v):
        # get the batch size from the query tensor
        batch_size = q.size(0)

        # apply the query projection, reshape, and transpose to split into heads
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the key projection, reshape, and transpose to split into heads
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the value projection, reshape, and transpose to split into heads
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # separate the odd-indexed elements from the query, key, and value tensors
        q_odd = q[:, :, ::2, :]
        k_odd = k[:, :, ::2, :]
        v_odd = v[:, :, ::2, :]
        
        # separate the even-indexed elements from the query, key, and value tensors
        q_even = q[:, :, 1::2, :]
        k_even = k[:, :, 1::2, :]
        v_even = v[:, :, 1::2, :]

        # calculate the attention scores for the odd-indexed elements
        attention_scores_odd = torch.matmul(q_odd, k_odd.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # apply the softmax function to obtain attention weights for the odd-indexed elements
        attention_weights_odd = torch.nn.functional.softmax(attention_scores_odd, dim=-1)
        # calculate the output for the odd-indexed elements
        output_odd = torch.matmul(attention_weights_odd, v_odd)

        # calculate the attention scores for the even-indexed elements
        attention_scores_even = torch.matmul(q_even, k_even.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # apply the softmax function to obtain attention weights for the even-indexed elements
        attention_weights_even = torch.nn.functional.softmax(attention_scores_even, dim=-1)
        # calculate the output for the even-indexed elements
        output_even = torch.matmul(attention_weights_even, v_even)

        # concatenate the odd and even outputs along the sequence length dimension
        output = torch.cat((output_odd, output_even), dim=2).transpose(1, 2).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = output.view(batch_size, -1, self.d_model)

        # apply the final linear layer to get the output and return it along with attention weights
        return self.fc(output), attention_weights_odd, attention_weights_even

# define the BinaryAttention class inheriting from nn.Module
class BinaryAttention(nn.Module):
    # initialize the binary attention mechanism with model dimensions and number of heads
    def __init__(self, d_model, num_heads):
        # call the parent class's constructor
        super(BinaryAttention, self).__init__()
        # store the number of attention heads
        self.num_heads = num_heads
        # store the dimensionality of the model
        self.d_model = d_model
        # calculate the depth of each head
        self.depth = d_model // num_heads

        # define a linear layer for the query projection
        self.wq = nn.Linear(d_model, d_model)
        # define a linear layer for the key projection
        self.wk = nn.Linear(d_model, d_model)
        # define a linear layer for the value projection
        self.wv = nn.Linear(d_model, d_model)
        # define a linear layer for the output projection
        self.fc = nn.Linear(d_model, d_model)

    # forward pass of the binary attention mechanism
    def forward(self, q, k, v):
        # get the batch size from the query tensor
        batch_size = q.size(0)

        # apply the query projection, reshape, and transpose to split into heads
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the key projection, reshape, and transpose to split into heads
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # apply the value projection, reshape, and transpose to split into heads
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # calculate the midpoint of the sequence length
        mid = q.size(2) // 2
        # split the queries, keys, and values into front and back halves
        q_front = q[:, :, :mid, :]
        k_front = k[:, :, :mid, :]
        v_front = v[:, :, :mid, :]

        q_back = q[:, :, mid:, :]
        k_back = k[:, :, mid:, :]
        v_back = v[:, :, mid:, :]

        # calculate the attention scores for the front half
        attention_scores_front = torch.matmul(q_front, k_front.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # apply the softmax function to obtain attention weights for the front half
        attention_weights_front = torch.nn.functional.softmax(attention_scores_front, dim=-1)
        # calculate the output for the front half
        output_front = torch.matmul(attention_weights_front, v_front)

        # calculate the attention scores for the back half
        attention_scores_back = torch.matmul(q_back, k_back.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        # apply the softmax function to obtain attention weights for the back half
        attention_weights_back = torch.nn.functional.softmax(attention_scores_back, dim=-1)
        # calculate the output for the back half
        output_back = torch.matmul(attention_weights_back, v_back)

        # concatenate the front and back outputs along the sequence length dimension
        output = torch.cat((output_front, output_back), dim=2).transpose(1, 2).contiguous()
        # reshape the output to combine the head dimension with the sequence length
        output = output.view(batch_size, -1, self.d_model)

        # apply the final linear layer to get the output and return it along with attention weights for both halves
        return self.fc(output), attention_weights_front, attention_weights_back
