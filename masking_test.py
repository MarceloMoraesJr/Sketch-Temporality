import torch

#model_input['encoder']['pos'] = model_input['encoder']['pos'] + torch.randn(size=model_input['encoder']['pos'].shape, dtype=model_input['encoder']['pos'].dtype).to(model_input['encoder']['pos'].device) * self.noise_std

def add_noise(points, mask, noise_std=0.01, mask_rate=0.2, generator=None):
    B, N, D = points.shape
    points = points.clone()
    dropped_points = torch.zeros_like(mask)

    for b in range(B):
        valid_idx = mask[b].nonzero(as_tuple=True)[0]
        n_valid = valid_idx.numel()
        n_drop = int(mask_rate * n_valid)
        
        if n_drop > 0:
            drop_idx = valid_idx[torch.randperm(n_valid, generator=generator)[:n_drop]].to(points.device)
            dropped_points[b, drop_idx] = True
            points[b, drop_idx] = 0.0

    valid_mask = mask & (~dropped_points)
    points[valid_mask] = points[valid_mask] + torch.randn(size=points[valid_mask].shape, dtype=points.dtype, generator=generator).to(points.device) * noise_std

    return points


# Example
B, N, D = 2, 10, 3
generator = torch.Generator()
generator.manual_seed(42)
points = torch.randn(B, N, D).to("cuda:1")
mask = torch.tensor([[1]*7 + [0]*3, [1]*5 + [0]*5]).to(torch.bool).to("cuda:1")
points[~mask] = 0.0
points = add_noise(points, mask, 0.01, 0.3, generator)