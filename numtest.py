import numpy as np
import torch

cir = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,0,0,0],[0,1,2,0,0,0,0,0,0,0,0,0,0,0]])

print(torch.argmax(cir,dim=1))