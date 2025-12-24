import torch

print("MPS is built:", torch.backends.mps.is_built())
print("MPS is available:", torch.backends.mps.is_available())

# Test MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(5, device=device)
    print("Tensor on MPS device:", x)

