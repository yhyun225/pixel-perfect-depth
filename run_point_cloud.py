import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import open3d as o3d
from ppd.utils.set_seed import set_seed
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.depth2pcd import depth2pcd
from ppd.moge.model.v2 import MoGeModel 
from ppd.models.ppd import PixelPerfectDepth

    

if __name__ == '__main__':
    set_seed(666) # set random seed
    parser = argparse.ArgumentParser(description='Pixel-Perfect Depth')
    parser.add_argument('--img_path', type=str, default='assets/examples')
    parser.add_argument('--input_size', type=int, default=[1024, 768])
    parser.add_argument('--outdir', type=str, default='depth_vis')
    parser.add_argument('--semantics_pth', type=str, default='checkpoints/depth_anything_v2_vitl.pth')
    parser.add_argument('--sampling_steps', type=int, default=20)
    parser.add_argument('--apply_filter', action='store_false', default=True)
    parser.add_argument('--pred_only', action='store_true', help='only display/save the predicted depth (no input image)')
    parser.add_argument('--save_npy', action='store_true', help='save raw depth prediction as .npy file (float32, unnormalized)')
    parser.add_argument('--save_pcd', action='store_true', help='save point cloud as .ply file')

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    moge = MoGeModel.from_pretrained("checkpoints/moge2.pt").to(DEVICE).eval()

    model = PixelPerfectDepth(semantics_pth=args.semantics_pth, sampling_steps=args.sampling_steps)
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
        
        image = cv2.imread(filename)
        H, W = image.shape[:2]
        depth, resize_image = model.infer_image(image)
        depth = depth.squeeze().cpu().numpy()

        # moge provide metric depth and intrinsic
        resize_H, resize_W = resize_image.shape[:2]
        moge_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        moge_image = torch.tensor(moge_image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1) 
        moge_depth, mask, intrinsic = moge.infer(moge_image)
        moge_depth[~mask] = moge_depth[mask].max()
        
        # relative depth -> metric depth
        metric_depth = recover_metric_depth_ransac(depth, moge_depth, mask)
        intrinsic[0, 0] *= resize_W 
        intrinsic[1, 1] *= resize_H
        intrinsic[0, 2] *= resize_W
        intrinsic[1, 2] *= resize_H

        # metric depth -> point cloud
        pcd = depth2pcd(metric_depth, intrinsic, color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB), input_mask=mask, ret_pcd=True)
        if args.apply_filter:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            pcd = pcd.select_by_index(ind)
        
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        vis_depth = vis_depth.astype(np.uint8)
        vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), vis_depth)
        else:
            split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([image, split_region, vis_depth])
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

        if args.save_npy:
            depth_npy_dir = 'depth_npy'
            os.makedirs(depth_npy_dir, exist_ok=True)
            npy_path = os.path.join(depth_npy_dir, os.path.splitext(os.path.basename(filename))[0] + '.npy')
            np.save(npy_path, depth)
        

        if args.save_pcd:
            depth_pcd_dir = 'depth_pcd'
            os.makedirs(depth_pcd_dir, exist_ok=True)
            pcd_path = os.path.join(depth_pcd_dir, os.path.splitext(os.path.basename(filename))[0] + '.ply')
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32))
            o3d.io.write_point_cloud(pcd_path, pcd)
