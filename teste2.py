import torch
from src.data import SPGDataset

dataset = SPGDataset("./data/spg/", 3, split="train-1")
label = []
for data in dataset:
    label.append(torch.tensor(data['label']))

label = torch.cat(label)
print(label.unique())

import os
total_labels = 0
for filename in os.listdir("data/spg/categories/"):
    with open(f"data/spg/categories/{filename}") as file:
        for line in file:
            if line.strip():
                total_labels += 1

print(total_labels)