from src.gpt import GPTLanguageModel
import torch
from tqdm import tqdm
from utils.dataloder import char_load_data
from utils.tokenizer import CharTokenizer


data, char = char_load_data('../data/tinyshakespeare.txt')
tokenizer = CharTokenizer(char)
data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
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

# train, val split
thresh = int(0.9 * len(data))
train_data = data[:thresh] # 90%
val_data = data[thresh:] # 10%

def get_batches(split):
    """
    generate batched in and out pairs
    """
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


train_losses = []
val_losses = []
for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # appending the losses
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])

    # sample a batch of data
    xb, yb = get_batches('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))