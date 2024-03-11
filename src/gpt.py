import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.dataloder import char_load_data

_, char = char_load_data('../data/tinyshakespeare.txt')
vocab_size = len(char)

# hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
num_embed = 384
num_head = 6
num_layer = 6
dropout = 0.2

class LayerNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = weights @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out
    
    
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):

    def __init__(self, num_embd, num_head):
        super().__init__()
        head_size = num_embd // num_head
        self.self_attention = MultiHeadAttention(num_head, head_size)
        self.ffwd = FeedFoward(num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class GPTLanguageModel(nn.Module):
    """
    GPTLanguageModel is a class that represents a language model based on the GPT architecture.

    Args:
        vocab_size (int): The size of the vocabulary.
        num_embed (int): The dimensionality of the token embeddings.
        block_size (int): The size of the blocks in the model.
        num_head (int): The number of attention heads in each block.
        num_layer (int): The number of blocks in the model.

    Attributes:
        token_embedding_table (nn.Embedding): The embedding table for the tokens.
        positional_embedding_table (nn.Embedding): The embedding table for the positional encodings.
        blocks (nn.Sequential): The sequence of blocks in the model.
        layer_norm (nn.LayerNorm): The layer normalization module.
        linear (nn.Linear): The linear layer for generating logits.

    Methods:
        forward(idx, targets=None): Performs a forward pass through the model.
        generate(idx, max_new_tokens): Generates new tokens based on the given context.

    """

    def __init__(self, vocab_size, num_embed, block_size, num_head, num_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embed)
        self.positional_embedding_table = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[Block(num_embed, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(num_embed)
        self.linear = nn.Linear(num_embed, vocab_size)

    def _init_weights(self, module):
        # Initialize the weights of the linear and embedding layers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Performs a forward pass through the GPT language model.

        Args:
            idx (torch.Tensor): The input tensor of shape (B, T) containing the indices of the tokens.
            targets (torch.Tensor, optional): The target tensor of shape (B, T) containing the indices of the target tokens.

        Returns:
            logits (torch.Tensor): The output tensor of shape (B, T, vocab_size) containing the logits for each token.
            loss (torch.Tensor or None): The computed loss tensor if targets are provided, otherwise None.

        """
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)
        position_embeddings = self.positional_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens based on the given context.

        Args:
            idx (torch.Tensor): The input tensor of shape (B, T) containing the indices of the tokens in the current context.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            idx (torch.Tensor): The output tensor of shape (B, T+max_new_tokens) containing the indices of the generated tokens.

        """
        for _ in range(max_new_tokens):
            idx_context = idx[:, -block_size:]
            logits, loss = self(idx_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx