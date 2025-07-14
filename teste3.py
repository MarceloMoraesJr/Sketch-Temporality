import os

categories1 = []
with open("./scripts/categories.txt", "r") as file:
    for line in file:
        if line[-1] == "\n":
            line = line[:-1]
        categories1.append(line)

categories2 = os.listdir("../sketch_representations/data/quickdraw/preprocessed")
categories2 = [s.split(".")[0] for s in categories2]

print(set(categories1) - set(categories2))