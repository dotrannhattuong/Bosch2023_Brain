import os 
import torch
import numpy as np
from torch import nn
from PIL import Image
from glob import glob
from torchvision import models
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary

import random
import math
import time
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

device = 'cuda'

