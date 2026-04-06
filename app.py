import torch
import torch.nn as nn
from torch.nn import functional as F
import streamlit as st

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------

# torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# creating vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenization and mapping
# tokenization
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode the strings to their respected embd
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs  and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    """Implements Self Attention"""

    def __init__(self, head_size):

        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x)  # (B, T , C)
        q = self.query(x)  # (B, T , C)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,16) @ (B, 16 , T) -> (B , T , T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B , T , C)
        out = wei @ v  # (B , T , T) @ ( B , T , C) -> (B , T , C)
        return out


# Multihead self attention
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non linear"""

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
    """Transformer Block : communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple Bigram Model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both of shape (batch_size , T , C)
        token_emb = self.token_embedding_table(idx)  # (B,T,C) (4,8,65)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T , C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B , T , vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits, targets = logits.to(device), targets.to(device)

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop the context
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax
            probs = F.softmax(logits, dim=-1)  # (B,C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B , 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B , T+1)

        return idx


# --- 4. Streamlit UI Logic ---
@st.cache_resource
# NEW WORKING CODE
@st.cache_resource
def load_model_weights():
    model = BigramLanguageModel()
    checkpoint = torch.load("best_model.pth", map_location=device)

    # Check if the file is a full checkpoint or just the state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


st.title("🤖 GPT Trainied on tiny shakespeare dataset")
st.write(
    "This model was trained on Shakespeare's works using a Transformer architecture."
)

with st.sidebar:
    st.header("Settings")
    length = st.slider("Number of tokens to generate", 50, 10000, 300)

if st.button("Generate Text"):
    model = load_model_weights()

    with st.spinner("Generating..."):
        # Start with a newline character as context
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_indices = model.generate(context, max_new_tokens=length)[0].tolist()
        result = decode(generated_indices)

    st.text_area("Generated Output:", value=result, height=400)
