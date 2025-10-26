import os
import argparse
import cv2

import torch
from torch.func import jvp

from ppd.utils.set_seed import set_seed
from ppd.models.ppd import PixelPerfectDepth

sampling_steps = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

breakpoint()
semantics_pth = "checkpoints/depth_anything_v2_vitl.pth"
model = PixelPerfectDepth(semantics_pth=semantics_pth, sampling_steps=sampling_steps, time_embedding='dual_v1')     # dtype: torch.float32
model.load_state_dict(torch.load('checkpoints/ppd.pth', map_location='cpu'), strict=False)
model = model.to(device)

image = torch.randn((1, 3, 1024, 768), device=device)
with torch.no_grad():
    semantics = model.semantics_prompt(image)

breakpoint()
z_t = torch.randn((1, 4, 1024, 768), device=device)
t = torch.tensor(1000).float().to(device)
r = torch.tensor(500).float().to(device)

def u_func(z_t, r, t): 
    return model.dit(x=z_t, semantics=semantics, timestep_1=t, timestep_2=r)

with torch.no_grad():
    model_input = (z_t, r, t)
    tangents = (z_t, torch.zeros_like(r), torch.ones_like(t))

    u_pred, dudt = jvp(u_func, model_input, tangents)

breakpoint()

