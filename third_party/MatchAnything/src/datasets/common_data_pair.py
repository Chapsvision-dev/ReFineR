import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from PIL import Image

from src.utils.dataset import read_megadepth_gray

class CommonDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 testNpairs=300,
                 fp16=False,
                 fix_bias=False,
                 sample_ratio=1.0,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.sample_ratio = sample_ratio

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score > 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = -3.0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        if mode == 'test':
            self.pair_infos = self.scene_info['pair_infos'][:testNpairs].copy()
        else:
            self.pair_infos = self.scene_info['pair_infos'].copy()

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        self.load_origin_rgb = kwargs["load_origin_rgb"]
        self.read_gray = kwargs["read_gray"]
        self.normalize_img = kwargs["normalize_img"]
        self.resize_by_stretch = kwargs["resize_by_stretch"]
        depth_max_size = 4000 if 'MTV_cross_modal_data' not in npz_path else 6000
        self.depth_max_size = depth_max_size if depth_padding else 2000  # the upperbound of depthmaps size in megadepth.

        self.dataset_name = self.scene_info['dataset_name'] if "dataset_name" in self.scene_info else npz_path.split(root_dir)[1].split('/')[1]
        self.gt_matches = self.scene_info['gt_matches'] if 'gt_matches' in self.scene_info else None # sparse matches produced by teacher models, used for training
        self.gt_matches_padding_n = kwargs["gt_matches_padding_n"]
        self.gt_2D_warp = self.scene_info['gt_2D_transforms'] if "gt_2D_transforms" in self.scene_info else None
        self.gt_2D_matches = self.scene_info['gt_2D_matches'] if "gt_2D_matches" in self.scene_info else None # Used for eval
        self.intrins = self.scene_info['intrinsics'] if 'intrinsics' in self.scene_info else None
        self.poses = self.scene_info['poses'] if 'poses' in self.scene_info else None
        
        self.fp16 = fp16
        self.fix_bias = fix_bias
        if self.fix_bias:
            self.df = 1          
  
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        if isinstance(self.pair_infos[idx], np.ndarray):
            idx0, idx1 = self.pair_infos[idx][0], self.pair_infos[idx][1]
            img_path0, img_path1 = self.scene_info['image_paths'][idx0][0], self.scene_info['image_paths'][idx1][1]
            K_0 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx0][0]).float()
            K_1 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx1][1]).float()

        else:
            if len(self.pair_infos[idx]) == 3:
                (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
            elif len(self.pair_infos[idx]) == 2:
                (idx0, idx1), overlap_score = self.pair_infos[idx]
            else:
                raise NotImplementedError

            img_path0, img_path1 = self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]
            K_0 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx0]).float()
            K_1 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx1]).float()

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.dataset_name, img_path0)
        img_name1 = osp.join(self.root_dir, self.dataset_name, img_path1) # Often transformed image based on img0, e.g., depth estimation or Diffusion
        # Note: should be pixel aligned with img0

        image0, mask0, scale0, origin_img_size0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1, origin_img_size1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)

        if self.gt_2D_warp is not None:
            gt_warp = np.concatenate([self.gt_2D_warp[idx], [[0,0,1]]]) # 3 * 3
        else:
            gt_warp = np.zeros((3, 3))

        depth0 = depth1 = torch.zeros([self.depth_max_size, self.depth_max_size], dtype=torch.float)

        homo_mask0 = torch.zeros((1, image0.shape[-2], image0.shape[-1]))
        homo_mask1 = torch.zeros((1, image1.shape[-2], image1.shape[-1]))
        gt_matches = torch.zeros((self.gt_matches_padding_n, 4), dtype=torch.float)

        if self.poses is None:
            T_0to1 = T_1to0 = torch.zeros((4,4), dtype=torch.float)  # (4, 4)
        else:
            # read and compute relative poses
            T0 = self.poses[idx0]
            T1 = self.poses[idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()

        if self.fp16:
            data = {
                'image0': image0.half(),  # (1, h, w)
                'depth0': depth0.half(),  # (h, w)
                'image1': image1.half(),
                'depth1': depth1.half(),
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'homography': torch.from_numpy(gt_warp.astype(np.float32)),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'origin_img_size0': origin_img_size0,
                'origin_img_size1': origin_img_size1,
                'scale0': scale0.half(),  # [scale_w, scale_h]
                'scale1': scale1.half(),
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_path0, img_path1),
            }
        else:
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'homography': torch.from_numpy(gt_warp.astype(np.float32)),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'origin_img_size0': origin_img_size0, # H W
                'origin_img_size1': origin_img_size1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_path0, img_path1),
                'rel_pair_names': (img_path0, img_path1)
            }
        
        if self.gt_2D_matches is not None:
            data.update({'gt_2D_matches': torch.from_numpy(self.gt_2D_matches[idx]).to(torch.float)}) # N * 4

        if self.gt_matches is not None:
            gt_matches_ = self.gt_matches[idx]
            if isinstance(gt_matches_, str):
                gt_matches_ = np.load(osp.join(self.root_dir, self.dataset_name, gt_matches_), allow_pickle=True)
            gt_matches_ = torch.from_numpy(gt_matches_).to(torch.float) # N * 4: mkpts0, mkpts1
            # Warp mkpts1 by sampled homo:
            num = min(len(gt_matches_), self.gt_matches_padding_n)
            gt_matches[:num] = gt_matches_[:num]

            data.update({"gt_matches": gt_matches, 'gt_matches_mask': torch.ones((1,), dtype=torch.bool), 'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float), "homo_sample_normed": torch.zeros((3,3), dtype=torch.float)})

        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                if self.fix_bias:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            size=((image0.shape[1]-1)//8+1, (image0.shape[2]-1)//8+1),
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
                else:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            scale_factor=self.coarse_scale,
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
            if self.fp16:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
            else:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        
        if self.load_origin_rgb:
            data.update({"image0_rgb_origin": torch.from_numpy(np.array(Image.open(img_name0).convert("RGB"))).permute(2,0,1) / 255., "image1_rgb_origin": torch.from_numpy(np.array(Image.open(img_name1).convert("RGB"))).permute(2,0,1)/ 255.})

        return data
