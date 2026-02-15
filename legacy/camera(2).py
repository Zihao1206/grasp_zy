import sys
# from cv2 import imshow, destroyAllWindows, waitKey
from numpy import asanyarray, array
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import numpy as np
import time
import datetime
import os


class RS():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)
        self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        # print(self.depth_scale)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        for i in range(5):
            self.pipeline.wait_for_frames()
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        # self.decimator = rs.hole_filling_filter()
        # self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        print(self.intr)
        self.intr = np.array([
            [604.335, 0, 316.187],
            [0, 604.404, 248.611],
            [0, 0, 1]
            ])
        

        # 跳过启动时的绿屏
        depth_image, color_image = self.get_img()
        # imshow('start', color_image)
        # waitKey(1000)
        time.sleep(1)
        # destroyAllWindows()

    def get_img(self):
        """
        update the frame and align it, then get the depth and the RGB image
        :return: (depth, color)
        """
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        depth_frame = self.aligned_frames.get_depth_frame()
        # depth_frame = self.decimator.process(depth_frame)
        color_frame = self.aligned_frames.get_color_frame()

        depth_image = asanyarray(depth_frame.get_data())
        color_image = asanyarray(color_frame.get_data())

        return depth_image, color_image

    def get_coordinate(self, x, y):
        """
        :param x: target position in the cv coordinate system, in pixel
        :param y:
        :return:(x, y, z) in meters in the camera coordinate system
        """
        pc = rs.pointcloud()
        points = rs.points()
        pc.map_to(self.aligned_frames.get_color_frame())
        points = pc.calculate(self.aligned_frames.get_depth_frame())
        vtx = asanyarray(points.get_vertices())
        # return np.vstack((vtx['f0'], vtx['f1'], vtx['f2'])).astype(np.float64).T
        central_point = int(y) * self.width + int(x)
        return float(vtx[central_point][0]), float(vtx[central_point][1]), float(vtx[central_point][2])

    def stop(self):
        self.pipeline.stop()


if __name__ == '__main__':
    import cv2
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = 'real_png_{}'.format(dt)
    data_save = '/home/zh/zh/grasp_zy_zhiyuan/others/photos'
    # data_save = '/home/jet/zoneyung/tu_0925'
    if not os.path.exists(data_save):
        os.makedirs(data_save)
    # rs = RS(640, 480)
    # depth_img, color_img = rs.get_img()

    # depth_img = np.array(depth_img).astype(np.float32)
    # depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    # mask = (depth_img == 0).astype(np.uint8) ## depth image 0->true

    # depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
    # depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
    # # Back to original size and value range.
    # depth_img = depth_img[1:-1, 1:-1]
    # depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0) / 1000

    # color_img = color_img[:, 80:560, :]
    # depth_img = depth_img[:, 80:560]


    pipeline = rs.pipeline()
    config = rs.config()
    align_to = rs.stream.color
    align = rs.align(align_to)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    try:
        frame_count = 0
        while True:
            # 等待下一帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = asanyarray(depth_frame.get_data())
            
            depth_img = np.array(depth_image).astype(np.float32)
            depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            mask = (depth_img == 0).astype(np.uint8) ## depth image 0->true

            depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
            depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
            # Back to original size and value range.
            depth_img = depth_img[1:-1, 1:-1]
            depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0) / 1000
            
            color_image = color_image[:, 80:560, :]
            depth_image = depth_image[:, 80:560]
            
            
            cv2.imshow("Capture_Video", color_image)  # 窗口显示，显示名为 Capture_Video
            k = cv2.waitKey(30) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            if k == ord('s'):  # 若检测到按键 ‘s’，打印字符串
                cv2.imwrite(os.path.join(data_save, 'zy_{}r.jpg'.format(frame_count)),
                            color_image)
                cv2.imwrite(os.path.join(data_save, 'zy_{}d.tiff'.format(frame_count)),
                depth_image)
                frame_count +=1
            elif k & 0xFF == ord('q'):
            # 按 'q' 键退出
                break
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
    # plt.figure(1)
    # plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.figure(2)
    # plt.imshow(depth_img)
    # plt.axis('off')
    # plt.show()

    # plt.imshow(depth_img)
    # plt.axis('off')
    # plt.savefig(os.path.join(data_save, '{}.tiff'.format(dt)))
    # plt.show()
    # cv2.imshow('imgOri', depth_img)
    # cv2.waitKey(0)
    # depth_img = np.array(depth_img).astype(np.float32)
    # depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    # mask = (depth_img == 0).astype(np.uint8)
    #
    # depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
    # depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
    # # Back to original size and value range.
    # depth_img = depth_img[1:-1, 1:-1]
    # depth_img = cv2.GaussianBlur(depth_img, (3, 3), 0) / 1000
    # c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(os.path.join(data_save, '{}.png'.format(dt)),
    #             color_img)
    # cv2.imwrite(os.path.join(data_save, '{}d.tiff'.format(dt)),
    #             depth_img)
    # time.sleep(5)
    # cv2.imwrite('c.png', c)
    # d, c = rs.get_img()
    # # import numpy as np
    # # print(np.max(d), np.min(d))
    # # import cv2
    # depth_img = d / 1000
    # depth_img = np.array(depth_img).astype(np.float32)
    # depth_img = cv2.copyMakeBorder(depth_img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    # mask = (depth_img == 0).astype(np.uint8)
    #
    # depth_img = depth_img.astype(np.float32)  # Has to be float32, 64 not supported.
    # depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
    # # Back to original size and value range.
    # depth_img = depth_img[1:-1, 1:-1]
    # # cv2.imshow('d', d/1000)
    # # cv2.imshow('c', c)
    # # cv2.waitKey(0)
    # plt.figure(1)
    # plt.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.figure(2)
    # plt.imshow(depth_img, cmap='bone')
    # plt.axis('off')
    # plt.show()






