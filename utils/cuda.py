
import torch

use_gpu = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
device = torch.device('cuda' if use_gpu else 'cpu')
