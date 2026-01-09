import numpy as np

import torch
import torch.utils.data

import random
from matplotlib import pyplot as plt

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=320, include_depth=True, include_rgb=True,
                 random_rotate=False, random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:

            rotations = [0, np.pi/2, np.pi, np.pi*3/2]
            rot = random.choice(rotations)
            # rot = np.pi/2
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
            # zoom_factor = 1
        else:
            zoom_factor = 1

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)
        else:
            rgb_img = []
        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        # import cv2
        # rgb = np.transpose(rgb_img, (1, 2, 0))
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # cv2.imshow('a', 4*rgb)
        # cv2.imshow('b', pos_img)
        # cv2.waitKey(0)

        width_img = np.clip(width_img, 0.0, 150.0)/150.0

        cmap = 'bwr_r'
        # plt.figure(1)
        # plt.imshow(pos_img, interpolation='nearest', cmap=cmap, origin='upper')
        # plt.colorbar(shrink=1)
        # plt.axis('off')
        #
        # plt.figure(2)
        # plt.imshow(ang_img, interpolation='nearest', cmap=cmap, origin='upper')
        # plt.colorbar(shrink=1)
        # plt.axis('off')
        #
        # plt.figure(3)
        # plt.imshow(width_img*150, interpolation='nearest', cmap=cmap, origin='upper')
        # # plt.imshow(pos_img)
        # plt.colorbar(shrink=1)
        # plt.axis('off')
        #
        # plt.figure(4)
        # plt.imshow(pos_img * 0.2, interpolation='nearest', cmap=cmap, origin='upper')
        # # plt.imshow(pos_img)
        # plt.colorbar(shrink=1)
        # plt.axis('off')
        # plt.show()

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(np.expand_dims(depth_img, 0))
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        # return x, (pos, cos, sin, width), rgb_img, depth_img, idx, rot, zoom_factor
        return x, (pos, cos, sin, width), idx, rot, zoom_factor
    def __len__(self):
        return len(self.grasp_files)
