import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
use_cuda = torch.cuda.is_available()
print(use_cuda)