import torch
from copy import deepcopy

class InputHandler():
    token_to_id = {
        "POS": 0,
        "SOS": 1,
        "NAR": 2
    }

    def __init__(self):
        pass

    def _right_shift(self, batch, token_id):
        B, L = batch['batch_size'], batch['batch_length']

        new_batch = deepcopy(batch)

        new_batch['token_id'] = torch.zeros(B, L+1, dtype=batch['stroke_id'].dtype)
        new_batch['token_id'][:, 0] = token_id

        new_batch['pos'] = torch.zeros(B, L+1, 2, dtype=batch['pos'].dtype)
        new_batch['pos'][:, 1:, :] = batch['pos']

        new_batch['pen_state'] = torch.zeros(B, L+1, 3, dtype=batch['pen_state'].dtype)
        new_batch['pen_state'][:, 1:, :] = batch['pen_state']

        new_batch['stroke_id'] = torch.zeros(B, L+1, dtype=batch['stroke_id'].dtype)
        new_batch['stroke_id'][:, 1:] = batch['stroke_id']

        new_batch['stroke_pos'] = torch.zeros(B, L+1, dtype=batch['stroke_pos'].dtype)
        new_batch['stroke_pos'][:, 1:] = batch['stroke_pos']

        new_batch['mask'] = torch.ones(B, L+1, dtype=batch['mask'].dtype)
        new_batch['mask'][:, 1:] = batch['mask']

        return new_batch
    
    def _prepare(self, batch, device):
        pos_info_keys = ['pen_state', 'stroke_id', 'stroke_pos']
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['pos_info'] = {k: batch[k] for k in pos_info_keys}
        batch = {k: v for k, v in batch.items() if k not in pos_info_keys}

        return batch
    
    def seq(self, batch):
        B, L = batch['batch_size'], batch['batch_length']
        new_batch = deepcopy(batch)
        new_batch['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype) + InputHandler.token_to_id['POS']

        new_batch = self._prepare(new_batch, batch['pos'].device)

        return new_batch

    def seq2seq(self, batch, right_shift):
        B, L = batch['batch_size'], batch['batch_length']

        new_batch = {'encoder': batch}

        new_batch['encoder']['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype) + InputHandler.token_to_id['POS']

        if right_shift:
            new_batch['decoder'] = self._right_shift(batch, InputHandler.token_to_id['SOS'])
        else:
            new_batch['decoder'] = deepcopy(batch)
            new_batch['decoder']['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype) + InputHandler.token_to_id['NAR']
            new_batch['decoder']['pos'] = torch.zeros_like(batch['pos'])

        new_batch['encoder'] = self._prepare(new_batch['encoder'], batch['pos'].device)
        new_batch['decoder'] = self._prepare(new_batch['decoder'], batch['pos'].device)

        return new_batch

    def sample_sketch(self, batch_size, num_points_per_stroke, num_strokes, device):
        batch = {}
        batch['batch_size'] = torch.tensor(batch_size)
        batch['batch_length'] = torch.tensor(num_strokes * num_points_per_stroke)
        batch['stroke_id'] = torch.arange(num_strokes, dtype=torch.int32).repeat_interleave(num_points_per_stroke).unsqueeze(0).repeat(batch_size,1)
        batch['stroke_pos'] = torch.arange(num_points_per_stroke).repeat(num_strokes).unsqueeze(0).repeat(batch_size, 1)
        batch['pen_state'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, 3, dtype=torch.float32)
        batch['pen_state'][:, :, 0] = 1.0
    
        idx = torch.arange(num_points_per_stroke-1, num_strokes * num_points_per_stroke, num_points_per_stroke)
        batch['pen_state'][:, idx, 0] = 0.0
        batch['pen_state'][:, idx, 1] = 1.0
        batch['pen_state'][:, idx, 2] = 0.0
        
        batch['pen_state'][:, num_strokes * num_points_per_stroke - 1, 1] = 0.0
        batch['pen_state'][:, num_strokes * num_points_per_stroke - 1, 2] = 1.0
        
        batch['token_id'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, dtype=torch.int32) + InputHandler.token_to_id['NAR']
        batch['pos'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, 2)
        batch['mask'] = torch.ones(batch_size, num_strokes * num_points_per_stroke, dtype=torch.bool)

        batch = self._prepare(batch, device)
        return batch
    

    def ar_prediction(self, decoder_batch, pred=None):
        i = 0 if pred is None else pred.size(1)

        new_decoder_batch = deepcopy(decoder_batch)

        new_decoder_batch['pos'] = new_decoder_batch['pos'][:, [0]] if i == 0 else pred[:, -1:]
        new_decoder_batch['pos_info'] = {k: decoder_batch['pos_info'][k][:, [i]] for k in ['pen_state', 'stroke_id', 'stroke_pos']}
        new_decoder_batch['token_id'] = decoder_batch['token_id'][:, [i]]
        new_decoder_batch['mask'] = decoder_batch['mask'][:, [i]]

        return new_decoder_batch