import os 
import numpy as np
import torch
import pathlib
import json
from torch.utils.data import Dataset
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm

from .perturbations_config import PerturbationsConfig

    
#SOURCE CODE FROM: https://soulhackerslabs.com/normalizing-feature-scaling-point-clouds-for-machine-learning-8138c6e69f5
def normalize_absolute_coords(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points


#SOURCE CODE FROM: https://github.com/leosampaio/sketchformer/blob/master/prep_data/sketch_token/create_token_dict.py
def normalize_relative_coords(points, bounds):
    # get bounds of sketch and use them to normalise
    min_x, max_x, min_y, max_y = bounds
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    points = points.astype(np.float32)
    points /= max_dim
    return points


def preprocess_sketch(sketch, perturbations, pos_generator, stroke_generator):
    data = np.zeros((len(sketch), 10), dtype=np.float32)

    abs_point = np.zeros(2)
    stroke_id = 0
    stroke_pos = 0
    
    data[:, :2] = sketch[:, :2]

    for i in range(len(sketch)):
        abs_point += sketch[i, :2]
        data[i, 2:4] = abs_point
        data[i, -3] = stroke_id
        data[i, -2] = stroke_pos

        if sketch[i, 2] == 0:
            data[i, 4] = 1
            stroke_pos += 1
        else:
            data[i, 5] = 1
            stroke_id += 1
            stroke_pos = 0
    
    data[-1, 5] = 0
    data[-1, 6] = 1

    pos_relative = data[:, [0, 1]]
    pos_absolute = data[:, [2, 3]]
    pen_state = data[:, [4, 5, 6]]

    stroke_id = data[:, -3].astype(np.int32)
    stroke_pos = data[:, -2].astype(np.int32)
    label = sketch[:, -1].astype(np.int32)

    #Position Perturbations
    pos_idx = np.arange(len(sketch))
    if perturbations.inter_stroke:
        pos_idx = pos_generator.permutation(len(sketch))
    elif perturbations.intra_stroke:
        for stroke in np.unique(stroke_id):
            pos_idx[stroke == stroke_id] = pos_idx[stroke == stroke_id][pos_generator.permutation((stroke == stroke_id).sum())]
    elif perturbations.intra_stroke_rev:
        unique_stroke_id = np.unique(stroke_id)
        for stroke in unique_stroke_id[pos_generator.permutation(len(unique_stroke_id))[:len(unique_stroke_id)//2 + 1]]:
            pos_idx[stroke == stroke_id] = pos_idx[stroke == stroke_id][::-1]

    pos_absolute = pos_absolute[pos_idx]
    label = label[pos_idx]

    #Stroke Order Perturbation
    stroke_idx = np.arange(len(sketch))
    if perturbations.stroke_order:
        unique_groups = np.unique(stroke_id)
        permuted_strokes = unique_groups[stroke_generator.permutation(len(unique_groups))]
        mapping = {old.item(): new.item() for old, new in zip(unique_groups, permuted_strokes)}
        permuted_strokes = np.array([mapping[g.item()] for g in stroke_id])
        original_index = np.arange(len(permuted_strokes))
        composite_index = permuted_strokes * len(permuted_strokes) + original_index

        stroke_id = permuted_strokes
        stroke_idx = np.argsort(composite_index)

    pos_absolute = pos_absolute[stroke_idx]
    pos_relative = np.concat((pos_absolute[0, None], np.diff(pos_absolute, axis=0)), axis=0)
    label = label[stroke_idx]

    stroke_id = stroke_id[stroke_idx]
    pen_state[-1] = [0., 1., 0.]
    pen_state = pen_state[stroke_idx]
    pen_state[-1] = [0., 0., 1.]
    stroke_pos = stroke_pos[stroke_idx]

    
    #Store Perturbation    
    data[:, [0, 1]] = pos_relative
    data[:, [2, 3]] = pos_absolute
    data[:, [4, 5, 6]] = pen_state

    data[:, -3] = stroke_id.astype(np.float32)
    data[:, -2] = stroke_pos.astype(np.float32)
    data[:, -1] = label.astype(np.float32)

    # Relative Coords Normalization
    bounds = np.zeros(4, dtype=np.float32)
    bounds[[0, 2]] = np.min(pos_absolute, axis=0)
    bounds[[1, 3]] = np.max(pos_absolute, axis=0)
    data[:, :2] = normalize_relative_coords(pos_relative, bounds)

    # Absolute Coords Normalization
    data[:, 2:4] = normalize_absolute_coords(pos_absolute)
    return data



class SPGDataset(Dataset):
    def __init__(self, root, num_splits=5, split='train-0', train_sample=650, valid_sample=50, test_sample=100, sample_seed=42, perturbations = PerturbationsConfig()):
        super().__init__()

        self.root_path = pathlib.Path(root)
        self.raw_path = self.root_path.joinpath("raw")
        self.categories_path = self.root_path.joinpath("categories")

        perturbation_path = [k for k,v in perturbations.__dict__.items() if v]
        perturbation_path = "_".join(perturbation_path)
        preprocessed_path = "preprocessed"
        if len(perturbation_path) > 0:
            preprocessed_path += "_" + perturbation_path

        self.preprocessed_path = self.root_path.joinpath(preprocessed_path)
        
        self.perturbations = perturbations

        #TODO - implement and move the dataset download script into this code
        if not self.raw_path.exists():
            pass 
        
        if not self.preprocessed_path.exists():
            self.num_splits = num_splits
            self.train_sample = train_sample
            self.valid_sample = valid_sample
            self.test_sample = test_sample
            self.sample_seed = sample_seed
            self.preprocessed_path.mkdir()
            self._preprocess()

        data = []
        for filename in os.listdir(self.preprocessed_path):
            data.append(np.load(self.preprocessed_path.joinpath(filename), encoding='latin1', allow_pickle=True)[split])

        self.data = np.concat(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pos_relative = self.data[index][:, [0, 1]]
        pos_absolute = self.data[index][:, [2, 3]]
        pen_state = self.data[index][:, [4, 5, 6]]

        stroke_id = self.data[index][:, -3].astype(np.int32)
        stroke_pos = self.data[index][:, -2].astype(np.int32)
        label = self.data[index][:, -1].astype(np.int32)

        return {
            'pos_relative': pos_relative,
            'pos_absolute': pos_absolute, 
            'pen_state': pen_state, 
            'stroke_id': stroke_id, 
            'stroke_pos': stroke_pos, 
            'label': label
        }
    
    def _download(self):
        pass

    def _preprocess(self):
        pos_generator = np.random.RandomState(self.sample_seed)
        stroke_generator = np.random.RandomState(self.sample_seed)

        total_labels = 0
        for filename in tqdm(os.listdir(self.raw_path)):
            with open(self.raw_path.joinpath(filename), 'r') as file:
                raw_data = json.load(file)

            processed_sketches = []
            for sketch in raw_data['train_data']:
                sketch = np.array(sketch, dtype=np.float32)
                sketch = preprocess_sketch(sketch, self.perturbations, pos_generator, stroke_generator)
                sketch[:, -1] = sketch[:, -1] + total_labels
                processed_sketches.append(sketch)
            
            new_data = {}
            spliter = ShuffleSplit(n_splits=self.num_splits, test_size=self.test_sample, random_state=self.sample_seed)
            idx = np.arange(len(raw_data['train_data']))
            for i, (train_valid_idx, test_idx) in enumerate(spliter.split(idx)):
                train_idx, valid_idx = train_test_split(train_valid_idx, test_size=self.valid_sample, random_state=i)
                new_data[f'train-{i}'] = np.array([processed_sketches[idx] for idx in train_idx], dtype=object)
                new_data[f'valid-{i}'] = np.array([processed_sketches[idx] for idx in valid_idx], dtype=object)
                new_data[f'test-{i}'] = np.array([processed_sketches[idx] for idx in test_idx], dtype=object)

            filename = filename.split('.')[0]
            with open(self.categories_path.joinpath(f"{filename}.txt")) as file:
                for line in file:
                    if line.strip():
                        total_labels += 1
            np.savez_compressed(self.preprocessed_path.joinpath(filename), **new_data)


    
    #Adapted from https://stackoverflow.com/questions/55041080/how-does-pytorch-dataloader-handle-variable-size-data
    @staticmethod
    def collate_fn_padd(batch):
        '''
        Padds batch of variable length
        '''

        new_batch = {}
        for key in batch[0].keys():
            new_batch[key] = [torch.tensor(data[key]) for data in batch]

        ## get sequence lengths
        new_batch['length'] = torch.tensor([t.shape[0] for t in new_batch['pos_relative']])
        ## padd
        new_batch['pos_relative'] = torch.nn.utils.rnn.pad_sequence(new_batch['pos_relative'])
        new_batch['pos_relative'] = new_batch['pos_relative'].permute(1, 0, 2)

        new_batch['pos_absolute'] = torch.nn.utils.rnn.pad_sequence(new_batch['pos_absolute'])
        new_batch['pos_absolute'] = new_batch['pos_absolute'].permute(1, 0, 2)
        ## compute mask
        new_batch['mask'] = torch.zeros(new_batch['pos_relative'].shape[0], new_batch['pos_relative'].shape[1], dtype=torch.bool)
        
        for i, length in enumerate(new_batch['length']):
            new_batch['mask'][i, :length] = True

        new_batch['pen_state'] = torch.nn.utils.rnn.pad_sequence(new_batch['pen_state']).permute(1, 0, 2)
        new_batch['stroke_id'] = torch.nn.utils.rnn.pad_sequence(new_batch['stroke_id']).permute(1, 0)
        new_batch['stroke_pos'] = torch.nn.utils.rnn.pad_sequence(new_batch['stroke_pos']).permute(1, 0)
        new_batch['label'] = torch.nn.utils.rnn.pad_sequence(new_batch['label']).permute(1, 0).to(torch.int64)
        
        new_batch['batch_size'] = torch.tensor(len(batch))
        new_batch['batch_length'] = torch.tensor(new_batch['pos_relative'].shape[1])

        return new_batch