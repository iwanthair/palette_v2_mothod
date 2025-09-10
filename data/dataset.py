import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class FloorPlanDataset(data.Dataset):
    def __init__(self, heatmap_dir, traj_dir, target_dir=None, image_size=(128, 128)):
        self.heatmap_dir = heatmap_dir
        self.traj_dir = traj_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is None:
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.traj_dir))
        else:
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.traj_dir)) == len(os.listdir(self.target_dir))

        self.filenames = sorted([
            fname for fname in os.listdir(self.heatmap_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_heatmap = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]

        heatmap_path = os.path.join(self.heatmap_dir, fname)
        traj_path = os.path.join(self.traj_dir, fname)
        

        heatmap_img = Image.open(heatmap_path).convert('RGB')  # 3 channel
        traj_img = Image.open(traj_path).convert('L')          # 1 channel
        

        heatmap_tensor = self.transform_heatmap(heatmap_img)
        traj_tensor = self.transform_gray(traj_img)
        

        cond_image = torch.cat([heatmap_tensor, traj_tensor], dim=0)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')      # 1 channel
            target_tensor = self.transform_gray(target_img)
            return {
                'gt_image': target_tensor,
                'cond_image': cond_image,
                'path': fname
            }

        else:
            target_tensor = torch.zeros_like(traj_tensor)  # dummy tensor if no target
            return {
                'gt_image': target_tensor,
                'cond_image': cond_image,
                'path': fname
            }
        

class FloorPlanDatasetTrajOnly(data.Dataset):
    def __init__(self, traj_dir, target_dir=None, image_size=(128, 128)):
        self.traj_dir = traj_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is None:
            assert len(os.listdir(self.traj_dir)) == len(os.listdir(self.target_dir))

        self.filenames = sorted([
            fname for fname in os.listdir(self.traj_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        traj_path = os.path.join(self.traj_dir, fname)

        traj_img = Image.open(traj_path).convert('L')          # 1 channel
        traj_tensor = self.transform_gray(traj_img)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')      # 1 channel
            target_tensor = self.transform_gray(target_img)
            return {
                'gt_image': target_tensor,
                'cond_image': traj_tensor,
                'path': fname
            }

        else:
            target_tensor = torch.zeros_like(traj_tensor)  # dummy tensor if no target
            return {
                'gt_image': target_tensor,
                'cond_image': traj_tensor,
                'path': fname
            }
        
class FloorPlanDatasetHeatMapOnly(data.Dataset):
    def __init__(self, heatmap_dir, target_dir=None, image_size=(128, 128)):
        self.heatmap_dir = heatmap_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is None:
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.target_dir))

        self.filenames = sorted([
            fname for fname in os.listdir(self.heatmap_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_heatmap = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]

        heatmap_path = os.path.join(self.heatmap_dir, fname)
        heatmap_img = Image.open(heatmap_path).convert('RGB')  # 3 channel
        
        heatmap_tensor = self.transform_heatmap(heatmap_img)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')      # 1 channel
            target_tensor = self.transform_gray(target_img)
            return {
                'gt_image': target_tensor,
                'cond_image': heatmap_tensor,
                'path': fname
            }

        else:
            # only one channel, so create dummy tensor with same shape
            target_tensor = torch.zeros_like(heatmap_tensor[0:1, :, :])
            return {
                'gt_image': target_tensor,
                'cond_image': heatmap_tensor,
                'path': fname
            }