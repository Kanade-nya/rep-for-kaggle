import torch

a = torch.tensor([[[7],[8],[9]],[[4],[5],[6]]]) # [2,3,1]
print(a.shape)
b = torch.tensor([[[1],[6],[4]]])# [1,3,1]
print(b.shape)
c = a + b
print(c)
print(c.shape)