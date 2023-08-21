import torch
from gpt3.gp3 import GPT3

x = torch.randint(0, 256, (1, 1024)).cuda()

model = GPT3()

model(x)