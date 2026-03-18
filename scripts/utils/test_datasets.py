




from torch.utils.data import Dataset
import json

import os
from os.path import dirname, join

from glob import glob

MAX_RETRIES = 1000
import matplotlib
import cv2
import numpy as np
from PIL import Image
from loguru import logger

def load_images(rgb_path, depth_path, depth_scale, max_depth):
    # Load RGB image and convert from BGR to RGB
    rgb_src = np.asarray(cv2.imread(rgb_path)[:, :, ::-1])
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")

    # Load depth image (usually 16-bit)
    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    # Convert depth to meters and clamp invalid values
    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0  # Remove values beyond max range

    # Create similarity depth (inverse depth) for model input
    # Only compute inverse for valid depth values
    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    return rgb_src, depth_low_res, simi_depth_low_res


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def concat_images(images):
    """Horizontally concatenate a list of PIL Images."""
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


class HAMMERDataset(Dataset):

    def __init__(self, jsonl_path,raw_type='d435'):

        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

        self.raw_type = raw_type

        self.depth_range = self.data[0].get('depth-range', [0.01, 6.0])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                item = self.data[idx]

                rgb = join(self.root,item['rgb'])

                if self.raw_type.lower() == 'd435':
                    raw_depth = join(self.root,item['d435_depth'])
                elif self.raw_type.lower() == 'l515':
                    raw_depth = join(self.root,item['l515_depth'])
                elif self.raw_type.lower() == 'tof':
                    raw_depth = join(self.root,item['tof_depth'])
                else:
                    raise ValueError(f"Invalid raw type: {self.raw_type}")
                
                gt_depth = join(self.root,item['depth'])
                
                if not (os.path.exists(rgb) and os.path.exists(raw_depth) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Error loading sample {idx} in HAMMERDataset: {e}. Retrying {attempt+1}/{MAX_RETRIES}...")
                    idx = np.random.randint(0, len(self.data))
                else:
                    logger.error(f"Failed to load sample after {MAX_RETRIES} retries.")
                    raise




class ClearPoseDataset(Dataset):

    def __init__(self, jsonl_path, max_length_each_sequence = 300):

        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []

        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None

        with open(jsonl_path, 'r') as f:
            for line in f:
                

                
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item['depth-range']

                rgb = sorted(glob(join(self.root,item['rgb'],'*'+item['rgb-suffix'])))[:max_length_each_sequence]
                raw_depth = sorted(glob(join(self.root,item['rgb'],'*'+item['raw_depth-suffix'])))[:max_length_each_sequence]
                gt_depth = sorted(glob(join(self.root,item['rgb'],'*'+item['depth-suffix'])))[:max_length_each_sequence]



                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)


                self.data.append(item)
        self.depth_range = depth_range

    def __len__(self):
        return len(self.rgbs)
        


    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                rgb = self.rgbs[idx]
                raw_depth = self.raw_depths[idx]
                gt_depth = self.gt_depths[idx]
                
                if not (os.path.exists(rgb) and os.path.exists(raw_depth) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")
                    
                return rgb, raw_depth, gt_depth
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Error loading sample {idx} in ClearPoseDataset: {e}. Retrying {attempt+1}/{MAX_RETRIES}...")
                    idx = np.random.randint(0, len(self.rgbs))
                else:
                    logger.error(f"Failed to load sample after {MAX_RETRIES} retries.")
                    raise


if __name__ == '__main__':


    # dataset = ClearPoseDataset('data/clearpose/test.jsonl')
    raw_type = 'l515'
    dataset  = HAMMERDataset('data/HAMMER/test.jsonl',raw_type=raw_type)
    depth_min = 0.1
    depth_max = 6.0
    depth_scale = 1000.0
    print(len(dataset))

    
    sample = dataset[0]
    

    rgb_src, depth_low_res, disp_depth_low_res = load_images(sample[0], sample[1], depth_scale,depth_max)
    logger.info(f'rgb_src: {rgb_src.shape}, depth_low_res: {depth_low_res.shape}, disp_depth_low_res: {disp_depth_low_res.shape}')


    #* resize to 518x ...
    from rgbddepth.util.transform import Resize
    resizer = Resize(width=518, height=518,resize_target=True,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method="lower_bound",image_interpolation_method=cv2.INTER_CUBIC)
    resized_res = resizer({"image": rgb_src, "depth": depth_low_res})
    rgb_src = resized_res["image"]
    depth_low_res = resized_res["depth"]
    logger.info(f'resized: {rgb_src.shape}, depth_low_res.shape: {depth_low_res.shape}')

    valid_mask = depth_low_res != 0
    valid_ratio = valid_mask.sum() / np.array(depth_low_res.shape).prod()
    logger.info(f'valid_ratio: {valid_ratio} ({valid_mask.sum()} / {np.array(depth_low_res.shape).prod()}) ')
    

    
    depth_visualization = colorize_depth_maps(
        depth_low_res, min_depth=depth_min, max_depth=depth_max, cmap="Spectral"
    )
    depth_visualization = np.rollaxis(depth_visualization[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    depth_visualization = (depth_visualization * 255).astype(np.uint8)
    

    #* load GT, seperately 
    depth_GT = cv2.imread(sample[2], cv2.IMREAD_UNCHANGED)
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    depth_GT[depth_GT > depth_max] = 0.0 


    resized_res = resizer({"image": np.repeat(np.zeros_like(depth_GT)[:,:,None],3,axis=2), "depth": depth_GT})
    depth_GT = resized_res["depth"]
    logger.info(f'resized: {depth_GT.shape}')

    

    depth_GT_visualization = colorize_depth_maps(
        depth_GT, min_depth=depth_min, max_depth=depth_max, cmap="Spectral"
    )
    depth_GT_visualization = np.rollaxis(depth_GT_visualization[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    depth_GT_visualization = (depth_GT_visualization * 255).astype(np.uint8)

    

    #* concat and vis
    concated = concat_images([Image.fromarray(rgb_src), Image.fromarray(depth_visualization), Image.fromarray(depth_GT_visualization)])

    concated.save(f'concated_{raw_type}.png')


    """
    
    dataset = HAMMERDataset('data/HAMMER/test.jsonl')
    depth_min = 0.1
    depth_max = 6.0
    depth_scale = 1000.0
    print(len(dataset))

    
    sample = dataset[0]
    
    rgb_src, depth_low_res, disp_depth_low_res = load_images(sample[0], sample[1], depth_scale,depth_max)

    
    
    depth_visualization = colorize_depth_maps(
        depth_low_res, min_depth=depth_min, max_depth=depth_max, cmap="Spectral"
    )
    depth_visualization = np.rollaxis(depth_visualization[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    depth_visualization = (depth_visualization * 255).astype(np.uint8)

    #* load GT, seperately 
    

    depth_GT = cv2.imread(sample[2], cv2.IMREAD_UNCHANGED)
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    depth_GT[depth_GT > depth_max] = 0.0 

    depth_GT_visualization = colorize_depth_maps(
        depth_GT, min_depth=depth_min, max_depth=depth_max, cmap="Spectral"
    )
    depth_GT_visualization = np.rollaxis(depth_GT_visualization[0], 0, 3)  # Convert from [C,H,W] to [H,W,C]
    depth_GT_visualization = (depth_GT_visualization * 255).astype(np.uint8)

    

    #* concat and vis
    concated = concat_images([Image.fromarray(rgb_src), Image.fromarray(depth_visualization), Image.fromarray(depth_GT_visualization)])

    concated.save('hammer_concated.png')

    
    
    """




    

    

    


