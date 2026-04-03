import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8 
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

# Simple Bigram Model
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size , vocab_size)

  def forward(self, idx , targets=None):

    #idx and targets are both of shape (batch_size , T , C)
    logits = self.token_embedding_table(idx) # (B,T,C) (4,8,65)

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

      # get the predictions
      logits, loss = self(idx)

      # focus only on the last time step
      logits = logits[: , -1, :]
      # apply softmax
      probs = F.softmax(logits , dim=-1) # (B,C)

      #sample from the distribution
      idx_next = torch.multinomial(probs , num_samples=1) # (B , 1)

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next) , dim=1) # (B , T+1)

    return idx

model = BigramLanguageModel(vocab_size)
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
