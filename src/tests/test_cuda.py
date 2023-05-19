import torch

print(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(torch.cuda.get_device_name(0))