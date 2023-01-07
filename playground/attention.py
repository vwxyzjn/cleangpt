import torch
import torch.nn as nn
import torch.nn.functional as F
# https://youtu.be/acxqoltilME

print("===Simple attention===")
seq_len = 3
embed_dim = 4
x1 = torch.rand(embed_dim)
x2 = torch.rand(embed_dim)
x3 = torch.rand(embed_dim)

attention_scores = torch.stack([x1 @ x1, x1 @ x2, x1 @x3], dim=0)
print("attention_scores", attention_scores)
attention_weights = F.softmax(attention_scores, dim=0)
print("attention_weights", attention_weights)
output = x1 * attention_weights[0] + x2 * attention_weights[1] + x3 * attention_weights[2]
print("output", output)

x = torch.stack([x1, x2, x3], dim=0)

attention_scores = x @ x.T
print("attention_scores", attention_scores)
attention_weights = F.softmax(attention_scores, dim=1)
print("attention_weights", attention_weights)
output = attention_weights @ x
print("output", output)

## Downside: no learnable parameters beyond the embedding x

print("===Scaled dot product attention===")
x1 = torch.rand(embed_dim)
x2 = torch.rand(embed_dim)
x3 = torch.rand(embed_dim)

wq = torch.rand(embed_dim, embed_dim)
wk = torch.rand(embed_dim, embed_dim)
wv = torch.rand(embed_dim, embed_dim)

q1 = x1 @ wq
k1 = x1 @ wk
v1 = x1 @ wv
q2 = x2 @ wq
k2 = x2 @ wk
v2 = x2 @ wv
q3 = x3 @ wq
k3 = x3 @ wk
v3 = x3 @ wv

attention_scores = torch.stack([q1 @ k1, q1 @ k2, q1 @ k3], dim=0) / torch.sqrt(torch.tensor(embed_dim))
print("attention_scores", attention_scores)
attention_weights = F.softmax(attention_scores, dim=0)
print("attention_weights", attention_weights)
output = v1 * attention_weights[0] + v2 * attention_weights[1] + v3 * attention_weights[2]
print("output", output)

x = torch.stack([x1, x2, x3], dim=0)
q = x @ wq
k = x @ wk
v = x @ wv
attention_scores = q @ k.T / torch.sqrt(torch.tensor(embed_dim))
print("attention_scores", attention_scores)
attention_weights = F.softmax(attention_scores, dim=1)
print("attention_weights", attention_weights)
output = attention_weights @ v
print("output", output)

print("===Multi-head self-attention===")
batch_size = 1
seq_len = 3
embed_dim = 12
num_heads = 3
head_dim = embed_dim // num_heads
x1 = torch.rand(embed_dim)
x2 = torch.rand(embed_dim)
x3 = torch.rand(embed_dim)
x = torch.stack([x1, x2, x3], dim=0)

wq = torch.rand(embed_dim, embed_dim // num_heads)
wk = torch.rand(embed_dim, embed_dim // num_heads)
wv = torch.rand(embed_dim, embed_dim // num_heads)

q = x @ wq
k = x @ wk
v = x @ wv

attention_scores = q @ k.T / torch.sqrt(torch.tensor(head_dim))
print("attention_scores", attention_scores)
attention_weights = F.softmax(attention_scores, dim=1)
print("attention_weights", attention_weights)
output = attention_weights @ v
print("output", output)

x = x.unsqueeze(0)
B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
wq = torch.cat((wq, wq, wq), dim=1)
wk = torch.cat((wk, wk, wk), dim=1)
wv = torch.cat((wv, wv, wv), dim=1)
wq2 = nn.Linear(embed_dim, embed_dim)

q = x @ wq
k = x @ wk
v = x @ wv
q = q.reshape(B, T, num_heads, head_dim).transpose(2, 1) # (B, num_heads, T, head_dim)
k = k.reshape(B, T, num_heads, head_dim).transpose(2, 1) # (B, num_heads, T, head_dim)
v = v.reshape(B, T, num_heads, head_dim).transpose(2, 1) # (B, num_heads, T, head_dim)

attention_scores = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(head_dim))
print("attention scores", attention_scores)
