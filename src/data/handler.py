import torch
from copy import deepcopy

class InputHandler():
    token_to_id = {
        "POS": 0,
        "SOS": 1,
        "NAR": 2
    }

    def __init__(self, input_relative_coords=True, output_relative_coords=True, mode="train", autoregressive=True):
        self.input_relative_coords = input_relative_coords
        self.output_relative_coords = output_relative_coords
        self.set_mode(mode)
        self.autoregressive = autoregressive

    def set_mode(self, mode):
        assert mode in ["train", "validation", "test"]
        self.mode = mode

    def _right_shift(self, batch, token_id):
        B, L = batch['batch_size'], batch['batch_length']

        model_input = deepcopy(batch)

        model_input['token_id'] = torch.zeros(B, L+1, dtype=batch['stroke_id'].dtype, device=batch['stroke_id'].device)
        model_input['token_id'][:, 0] = token_id

        model_input['pos'] = torch.zeros(B, L+1, 2, dtype=batch['pos'].dtype, device=batch['pos'].device)
        model_input['pos'][:, 1:, :] = batch['pos']

        model_input['pen_state'] = torch.zeros(B, L+1, 3, dtype=batch['pen_state'].dtype, device=batch['pen_state'].device)
        model_input['pen_state'][:, 1:, :] = batch['pen_state']

        model_input['stroke_id'] = torch.zeros(B, L+1, dtype=batch['stroke_id'].dtype, device=batch['stroke_id'].device)
        model_input['stroke_id'][:, 1:] = batch['stroke_id']

        model_input['stroke_pos'] = torch.zeros(B, L+1, dtype=batch['stroke_pos'].dtype, device=batch['stroke_pos'].device)
        model_input['stroke_pos'][:, 1:] = batch['stroke_pos']

        model_input['mask'] = torch.ones(B, L+1, dtype=batch['mask'].dtype, device=batch['mask'].device)
        model_input['mask'][:, 1:] = batch['mask']

        return model_input
    
    def _prepare(self, model_input):
        pos_info_keys = ['pen_state', 'stroke_id', 'stroke_pos']
        model_input['pos_info'] = {k: model_input[k] for k in pos_info_keys}
        model_input = {k: v for k, v in model_input.items() if k not in pos_info_keys}

        return model_input
    
    def _prepare_encoder_only_input(self, batch):
        B, L = batch['batch_size'], batch['batch_length']
        model_input = deepcopy(batch)
        model_input['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype) + InputHandler.token_to_id['POS']
        model_input['pos'] = model_input['pos_relative'] if self.input_relative_coords else model_input['pos_absolute']
        del model_input['pos_relative']
        del model_input['pos_absolute']

        model_input = self._prepare(model_input)

        return model_input

    def _prepare_autoencoder_input(self, batch):
        B, L = batch['batch_size'], batch['batch_length']

        model_input = {'encoder': batch}
        model_input['encoder']['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype, device=batch['stroke_id'].device) + InputHandler.token_to_id['POS']

        model_input['encoder']['pos'] = batch['pos_relative'] if self.input_relative_coords else batch['pos_absolute']
        model_input['ground_truth'] = batch['pos_relative'] if self.output_relative_coords and self.mode != "test" else batch['pos_absolute']

        del model_input['encoder']['pos_relative']
        del model_input['encoder']['pos_absolute']

        if self.autoregressive:
            model_input['decoder'] = self._right_shift(model_input['encoder'], InputHandler.token_to_id['SOS'])
        else:
            model_input['decoder'] = deepcopy(model_input['encoder'])
            model_input['decoder']['token_id'] = torch.zeros(B, L, dtype=batch['stroke_id'].dtype, device=batch['stroke_id'].device) + InputHandler.token_to_id['NAR']
            model_input['decoder']['pos'] = torch.zeros_like(batch['pos'], device=model_input['encoder']['pos'].device)

        model_input['encoder'] = self._prepare(model_input['encoder'])
        model_input['decoder'] = self._prepare(model_input['decoder'])

        return model_input
    
    def __call__(self, batch, autoencoder=True):
        if autoencoder:
            return self._prepare_autoencoder_input(batch)
        
        return self._prepare_encoder_only_input(batch)
        
    # def sample_sketch(self, batch_size, num_points_per_stroke, num_strokes, device):
    #     batch = {}
    #     batch['batch_size'] = torch.tensor(batch_size)
    #     batch['batch_length'] = torch.tensor(num_strokes * num_points_per_stroke)
    #     batch['stroke_id'] = torch.arange(num_strokes, dtype=torch.int32).repeat_interleave(num_points_per_stroke).unsqueeze(0).repeat(batch_size,1)
    #     batch['stroke_pos'] = torch.arange(num_points_per_stroke).repeat(num_strokes).unsqueeze(0).repeat(batch_size, 1)
    #     batch['pen_state'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, 3, dtype=torch.float32)
    #     batch['pen_state'][:, :, 0] = 1.0
    
    #     idx = torch.arange(num_points_per_stroke-1, num_strokes * num_points_per_stroke, num_points_per_stroke)
    #     batch['pen_state'][:, idx, 0] = 0.0
    #     batch['pen_state'][:, idx, 1] = 1.0
    #     batch['pen_state'][:, idx, 2] = 0.0
        
    #     batch['pen_state'][:, num_strokes * num_points_per_stroke - 1, 1] = 0.0
    #     batch['pen_state'][:, num_strokes * num_points_per_stroke - 1, 2] = 1.0
        
    #     batch['token_id'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, dtype=torch.int32) + InputHandler.token_to_id['NAR']
    #     batch['pos'] = torch.zeros(batch_size, num_strokes * num_points_per_stroke, 2)
    #     batch['mask'] = torch.ones(batch_size, num_strokes * num_points_per_stroke, dtype=torch.bool)

    #     batch = self._prepare(batch, device)
    #     return batch



class OutputHandler():
    def __init__(self, output_relative_coords, mode="train", autoregressive=True):

        self.output_relative_coords = output_relative_coords
        self.autoregressive = autoregressive
        self.set_mode(mode)

    def set_mode(self, mode):
        assert mode in ["train", "validation", "test"]
        self.mode = mode

    def _to_absolute(self, points, mask):
        points = torch.cumsum(points, dim=1)
        points[~mask] = 0.0

        if mask is None:
            mask = torch.ones_like(points, dtype=torch.bool)

        # Compute valid point counts
        mask_float = mask.float().unsqueeze(-1)
        valid_counts = mask_float.sum(dim=-2, keepdim=True).clamp(min=1e-6)


        # Compute centroid over valid points
        centroid = (points * mask_float).sum(dim=-2, keepdim=True) / valid_counts
        centered = points - centroid

        # Compute distance norm per point (but only valid)
        dists = torch.sqrt((centered ** 2).sum(dim=-1))
        dists = dists * mask_float.squeeze(-1)
        furthest_dist = dists.max(dim=-1, keepdim=True)[0].clamp(min=1e-6)

        normalized = centered / furthest_dist.unsqueeze(-1)
        return normalized


    def __call__(self, recons_points, mask=None):
        # Undo the right-shift on predictions
        if self.autoregressive:
            recons_points = recons_points[:, :-1]

        # NOOP
        if self.mode != "test" or not self.output_relative_coords:
            return recons_points
        
        return self._to_absolute(recons_points, mask)