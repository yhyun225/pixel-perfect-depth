import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
from ppd.utils.set_seed import set_seed
from ppd.models.ppd import PixelPerfectDepth
    

if __name__ == '__main__':
    set_seed(666) # set random seed
    parser = argparse.ArgumentParser(description='Pixel-Perfect Depth')
    parser.add_argument('--img_path', type=str, default='assets/examples')
    parser.add_argument('--input_size', type=int, default=[1024, 768])
    parser.add_argument('--outdir', type=str, default='depth_vis')
    parser.add_argument('--semantics_pth', type=str, default='checkpoints/depth_anything_v2_vitl.pth')
    parser.add_argument('--sampling_steps', type=int, default=4)
    parser.add_argument('--pred_only', action='store_true', help='only display/save the predicted depth (no input image)')
    parser.add_argument('--save_npy', action='store_true', help='save raw depth prediction as .npy file (float32, unnormalized)')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = PixelPerfectDepth(semantics_pth=args.semantics_pth, sampling_steps=args.sampling_steps)     # dtype: torch.float32
    model.load_state_dict(torch.load('checkpoints/ppd.pth', map_location='cpu'), strict=False)

    model = model.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = sorted(filenames)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        image = cv2.imread(filename)            # 'np.ndarray', [h, w, c], ~[np.uint8(0), np.uint8(255)]
        H, W = image.shape[:2]
        
        depth, _ = model.infer_image(image)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.squeeze().cpu().numpy()
        
        vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        vis_depth = vis_depth.astype(np.uint8)
        # vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), vis_depth)
        # else:
            # split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
            # combined_result = cv2.hconcat([image, split_region, vis_depth])
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

        if args.save_npy:
            depth_npy_dir = 'depth_npy'
            os.makedirs(depth_npy_dir, exist_ok=True)
            npy_path = os.path.join(depth_npy_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
            np.save(npy_path, depth)
