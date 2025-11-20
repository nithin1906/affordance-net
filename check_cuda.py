import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0 name:", torch.cuda.get_device_name(0))
    x = torch.randn(3,3).cuda()
    print("tensor on cuda ok:", x.device)
else:
    print("CUDA NOT AVAILABLE ❌")
