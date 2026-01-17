import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # 独立的句子，并行计算数量
block_size = 8 #    上下文长度
max_iters = 3000 #  训练迭代次数
eval_interval = 300 # 评估间隔
learning_rate = 1e-2 # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # 评估时的batch数
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text))) # 所有出现过的字符
vocab_size = len(chars) # 词表的长度 65

stoi = { ch:i for i,ch in enumerate(chars) } # 映射字典， key: char, value: idx
itos = { i:ch for i,ch in enumerate(chars) } # 映射字典， key: idx, value: char
encode = lambda s: [stoi[c] for c in s] # encoder: 给一个字符串，输出对应索引
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: 给定索引，解码字符串

# 划分训练和验证集 （ 9：1）
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# 数据加载
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机获取batch_size个起始字符索引
    x = torch.stack([data[i:i+block_size] for i in ix]) # x输入字符序列
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y：目标字符序列（右移一位）
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx：当前上下文字符序列（整数索引），形状 (B, T)
        for _ in range(max_new_tokens):
            # get the predictions, 调用forward
            logits, loss = self(idx)
            # 取最后一个时间步，因为 Bigram 模型只依赖当前最后一个字符
            logits = logits[:, -1, :] # becomes (B, C)
            # 转成概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 根据概率采样下一个字符
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 拼接到现有序列
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx # 形状 (B, T + max_new_tokens)

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 每 eval_interval = 300 步，评估训练集和验证集的平均 loss
    # estimate_loss() 是前面定义的无梯度函数
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 获取一个batch的数据
    xb, yb = get_batch('train')

    # 前向 + loss
    logits, loss = model(xb, yb)
    # 清除上一步梯度
    optimizer.zero_grad(set_to_none=True)
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
