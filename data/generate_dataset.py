import torch

x = torch.randn((1000, 2))
y = x[:, 0] + x[:, 1] + 1

torch.save({"x": x, "y": y}, "dataset.pt")
dataset = torch.load("dataset.pt")
