import torch
import torchvision

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')