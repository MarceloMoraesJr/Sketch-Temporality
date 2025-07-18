from src.data import SPGDataset, Perturbations

data1 = SPGDataset("./data/spg/", perturbations=Perturbations(False, False, False, False))[67]
data2 = SPGDataset("./data/spg/", perturbations=Perturbations(True, False, False, False))[67]
