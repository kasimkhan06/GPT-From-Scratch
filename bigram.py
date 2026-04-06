import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8 
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

#----------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt

with open('input.txt' , 'r' , encoding='utf-8') as f:
  text = f.read()

# creating vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenization and mapping
# tokenization
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# encode the strings to their respected embd
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])
 
# Train and test splits
data = torch.tensor(encode(text) , dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  #generate a small batch of data of inputs  and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size , (batch_size,))
  x = torch.stack([data[i : i+block_size] for i in ix])
  y = torch.stack([data[i+1 : i+block_size+1] for i in ix])
  x , y = x.to(device) , y.to(device)

  return x , y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      X , Y = X.to(device) , Y.to(device)
      logits,loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()

  model.train()
  return out

class Head(nn.Module):

  """ Implements Self Attention """
  def __init__(self, head_size):

    super().__init__()
    self.key = nn.Linear(n_embd , head_size , bias = False)
    self.query = nn.Linear(n_embd , head_size , bias = False)
    self.value = nn.Linear(n_embd , head_size , bias=False)
    self.register_buffer('tril' , torch.tril(torch.ones(block_size,block_size)))

  def forward(self, x):

    B , T , C = x.shape
    k = self.key(x) # (B, T , C)
    q = self.query(x) # (B, T , C)

    # compute attention scores
    wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,16) @ (B, 16 , T) -> (B , T , T)
    wei = wei.masked_fill(self.tril[:T , : T] == 0 , float('-inf'))
    wei = F.softmax(wei , dim =-1)

    v = self.value(x) # (B , T , C)
    out = wei @ v # (B , T , T) @ ( B , T , C) -> (B , T , C)
    return out

# Multihead self attention
class MultiHeadAttention(nn.Module):
  """multiple heads of self-attention in parallel"""
  def __init__(self, num_heads , head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)

# Simple Bigram Model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size , n_embd)
    self.position_embedding_table = nn.Embedding(block_size , n_embd)
    self.sa_head = Head(n_embd)
    self.lm_head = nn.Linear(n_embd , vocab_size)

  def forward(self, idx , targets=None):

    B,T = idx.shape
    #idx and targets are both of shape (batch_size , T , C)
    token_emb = self.token_embedding_table(idx) # (B,T,C) (4,8,65)
    pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T , C)
    x = token_emb + pos_emb
    x = self.sa_head(x)
    logits = self.lm_head(x) # (B , T , vocab_size)

    if targets is None:
      loss = None

    else:
      B , T , C = logits.shape
      logits , targets = logits.to(device) , targets.to(device)
      
      logits = logits.view(B*T , C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits , loss

  def generate(self , idx , max_new_tokens):
    #idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      
      # crop the context
      idx_cond = idx[: , -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)

      # focus only on the last time step
      logits = logits[: , -1, :]
      # apply softmax
      probs = F.softmax(logits , dim=-1) # (B,C)

      #sample from the distribution
      idx_next = torch.multinomial(probs , num_samples=1) # (B , 1)

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next) , dim=1) # (B , T+1)

    return idx

model = BigramLanguageModel()
m = model.to(device)  

# optimizer
optimizer = torch.optim.AdamW(m.parameters() , lr=1e-3)

# training loop
for iter in range(max_iters):
  
  # train val loss
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter} : train los {losses['train']:.4f} , val loss : {losses['val']:.4f}")
  
  # sample a batch of data
  xb , yb = get_batch('train')

  # evaluate the loss
  logits, loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


#generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context , max_new_tokens=500)[0].tolist()))
