# import torch
# import numpy as np

# print(torch.__version__)
# print(np.__version__)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

import os
from from_root import from_root

print(from_root())
