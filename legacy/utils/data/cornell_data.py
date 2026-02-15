import os
import glob

from utils.data.grasp_data import GraspDatasetBase
from utils.dataset_processing import grasp, image
from skimage.transform import rotate, resize
from cv2 import getRotationMatrix2D, warpAffine
from numpy import transpose
from numpy import pi as pi


class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
        graspf.sort()
        l = len(graspf)
        print(l)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

        self.depth_img = []
        self.rgb_img = []

        for i in range(len(self.grasp_files)):
            depth = self.load_depth(i)
            rgb = self.load_rgb(i)
            self.depth_img.append(depth)
            self.rgb_img.append(rgb)

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.offset((-top, -left))
        gtbbs.rotate(rot, (self.output_size // 2, self.output_size // 2))
        gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = self.depth_img[idx]
        mat = getRotationMatrix2D((depth_img.shape[1]//2, depth_img.shape[0]//2), rot/pi*180, 1 / zoom)
        depth_img = warpAffine(depth_img, mat, (depth_img.shape[1], depth_img.shape[0]))
        return depth_img

    def get_rgb(self, idx, rot=0, zoom=1.0):
        rgb_img = self.rgb_img[idx]
        # rgb_img = transpose(rgb_img, (1, 2, 0))
        mat = getRotationMatrix2D((rgb_img.shape[1]//2, rgb_img.shape[0]//2), rot/pi*180, 1 / zoom)
        rgb_img = warpAffine(rgb_img, mat, (rgb_img.shape[1], rgb_img.shape[0]))
        rgb_img = transpose(rgb_img, (2, 0, 1))

        return rgb_img

    def load_depth(self, idx, rot=0, zoom=1.0):
        # start = time.time()
        print(self.depth_files[idx])

        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        # t_load = time.time()
        # print('depth: load depth image:{:.3f}s'.format(t_load - start))

        center, left, top = self._get_crop_attrs(idx)
        # t_cal_crop = time.time()
        # print('depth: cal crop:{:.3f}s'.format(t_cal_crop - t_load))

        # depth_img.rotate(rot, center)
        # t_rotate = time.time()
        # print('depth: rotate depth image:{:.3f}s'.format(t_rotate - t_cal_crop))

        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # t_crop = time.time()
        # print('depth: crop depth image:{:.3f}s'.format(t_crop - t_rotate))

        # depth_img.normalise()
        # t_norm = time.time()
        # print('depth: norm depth image:{:.3f}s'.format(t_norm - t_crop))

        # depth_img.zoom(zoom)
        # t_zoom = time.time()
        # print('depth: zoom depth image:{:.3f}s'.format(t_zoom - t_norm))

        depth_img.resize((self.output_size, self.output_size))
        # t_resize = time.time()
        # print('depth: resize depth image:{:.3f}s'.format(t_resize - t_zoom))

        return depth_img.img

    def load_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # start = time.time()

        rgb_img = image.Image.from_file(self.rgb_files[idx])
        # t_load = time.time()
        # print('rgb: load depth image:{:.3f}s'.format(t_load - start))

        center, left, top = self._get_crop_attrs(idx)
        # t_cal_crop = time.time()
        # print('rgb: cal crop:{:.3f}s'.format(t_cal_crop - t_load))

        # rgb_img.rotate(rot, center)
        # t_rotate = time.time()
        # print('rgb: rotate depth image:{:.3f}s'.format(t_rotate - t_cal_crop))

        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # t_crop = time.time()
        # print('rgb: crop depth image:{:.3f}s'.format(t_crop - t_rotate))

        # rgb_img.zoom(zoom)
        # t_zoom = time.time()
        # print('rgb: zoom depth image:{:.3f}s'.format(t_zoom - t_crop))

        rgb_img.resize((self.output_size, self.output_size))
        # t_resize = time.time()
        # print('rgb: resize depth image:{:.3f}s'.format(t_resize - t_zoom))

        if normalise:
            # rgb_img.normalise()
            # rgb_img.img = rgb_img.img.transpose((2, 0, 1))
            # t_norm = time.time()
            # print('depth: norm depth image:{:.3f}s'.format(t_norm - t_resize))
            pass

        return rgb_img.img


if __name__ == '__main__':
    dataset = CornellDataset('/home/wangchuxuan/PycharmProjects/grasp/data', output_size=320,
                             start=0.385, end=0.386)
    import torch.utils.data as D
    import time
    train_data = D.DataLoader(dataset, batch_size=1)
    # T = time.time()
    # count = 0
    # while True:
    #     count += 1
    #     count1 = 0
    #     for data in train_data:
    #         count1 += 1
    #         T0 = time.time()
    #         if count1 % 100 == 0:
    #             print('第{}次循环，第{}个数据，用时{:.5f}s'.format(count, count1, T0 - T))
    #         T = time.time()

    for x, y, _, _, _, _, _ in train_data:
        print(x.shape)
        pass


