import glob
import random
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.draw import polygon
from PIL import Image, ImageDraw


from utils import torch_utils
import math

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    # Loads class labels at 'path'
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %38s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %38s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))


def coco_class_weights():  # frequency of each class in coco train2014
    weights = 1 / torch.FloatTensor(
        [187437, 4955, 30920, 6033, 3838, 4332, 3160, 7051, 7677, 9167, 1316, 1372, 833, 6757, 7355, 3302, 3776, 4671,
         6769, 5706, 3908, 903, 3686, 3596, 6200, 7920, 8779, 4505, 4272, 1862, 4698, 1962, 4403, 6659, 2402, 2689,
         4012, 4175, 3411, 17048, 5637, 14553, 3923, 5539, 4289, 10084, 7018, 4314, 3099, 4638, 4939, 5543, 2038, 4004,
         5053, 4578, 27292, 4113, 5931, 2905, 11174, 2873, 4036, 3415, 1517, 4122, 1980, 4464, 1190, 2302, 156, 3933,
         1877, 17630, 4337, 4624, 1075, 3468, 135, 1380])
    weights /= weights.sum()
    return weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def point2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(torch.empty(x.shape[0], 4)) if x.dtype is torch.float32 else np.zeros_like(torch.empty(x.shape[0], 4))
    y[:, 0] = (x[:, 0] + x[:, 4]) / 2
    y[:, 1] = (x[:, 1] + x[:, 5]) / 2
    y[:, 2] = np.sqrt((x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2)
    y[:, 3] = np.sqrt((x[:, 2] - x[:, 4]) ** 2 + (x[:, 3] - x[:, 5]) ** 2)
    return y

def xywha2points(x):
    # Convert bounding box format from [x1, y1, w, h, theta] to [x, y, x, y,x,y,x,y,x,y]

    list=[]
    cos = torch.cos(x[:, 4])
    sin = torch.sin(x[:, 4])
    # cos = np.cos(x[:, 4])
    # sin = np.sin(x[:, 4])

    
    x1 = x[:, 0] - x[:, 2] / 2 * cos # x-w*cos theta/2
    y1 = x[:, 1] + x[:, 2] / 2 * sin # y+w*sin theta/2
    x2 = x[:, 0] + x[:, 2] / 2 * cos # x+w*cos theta/2
    y2 = x[:, 1] - x[:, 2] / 2 * sin # x-w*sin theta/2

    for i in range(x.shape[0]):
        list.append(
            [[x1[i] - x[i, 3]/2 * sin[i], y1[i] - x[i, 3]/2 * cos[i]],
             [x2[i] - x[i, 3]/2 * sin[i], y2[i] - x[i, 3]/2 * cos[i]],
             [x2[i] + x[i, 3]/2 * sin[i], y2[i] + x[i, 3]/2 * cos[i]],
             [x1[i] + x[i, 3]/2 * sin[i], y1[i] + x[i, 3]/2 * cos[i]], [x[i, 4]]
             ])

    # list.append(points)
    return list

def xywa2points(x):
    # Convert bounding box format from [x1, y1, w, h, theta] to [x, y, x, y,x,y,x,y,x,y]

    list=[]
    cos = torch.cos(x[:, 3])
    sin = torch.sin(x[:, 3])
    # cos = np.cos(x[:, 4])
    # sin = np.sin(x[:, 4])

    
    x1 = x[:, 0] - x[:, 2] / 2 * cos # x-w*cos theta/2
    y1 = x[:, 1] + x[:, 2] / 2 * sin # y+w*sin theta/2
    x2 = x[:, 0] + x[:, 2] / 2 * cos # x+w*cos theta/2
    y2 = x[:, 1] - x[:, 2] / 2 * sin # x-w*sin theta/2

    for i in range(x.shape[0]):
        list.append(
            [[x1[i] - x[i, 2]/4 * sin[i], y1[i] - x[i, 2]/4 * cos[i]],
             [x2[i] - x[i, 2]/4 * sin[i], y2[i] - x[i, 2]/4 * cos[i]],
             [x2[i] + x[i, 2]/4 * sin[i], y2[i] + x[i, 2]/4 * cos[i]],
             [x1[i] + x[i, 2]/4 * sin[i], y1[i] + x[i, 2]/4 * cos[i]], [x[i, 3]]
             ])

    # list.append(points)
    return list

# def xywha2points1(x):
#     # Convert bounding box format from [x1, y1, w, h, theta] to [x, y, x, y,x,y,x,y,x,y]

#     list=[]
#     cos = torch.cos(torch.abs(x[:, 4]))
#     sin = torch.sin(torch.abs(x[:, 4]))
#     # cos = np.cos(x[:, 4])
#     # sin = np.sin(x[:, 4])

    
#     x1 = x[:, 0] - x[:, 2] / 2 * cos # x-w*cos theta/2
#     y1 = x[:, 1] - x[:, 2] / 2 * sin # y+w*sin theta/2
#     x2 = x[:, 0] + x[:, 2] / 2 * cos # x+w*cos theta/2
#     y2 = x[:, 1] + x[:, 2] / 2 * sin # x-w*sin theta/2

#     for i in range(x.shape[0]):
#         list.append(
#             [[x1[i] + x[i, 3]/2 * sin[i], y1[i] - x[i, 3]/2 * cos[i]],
#              [x2[i] + x[i, 3]/2 * sin[i], y2[i] - x[i, 3]/2 * cos[i]],
#              [x2[i] - x[i, 3]/2 * sin[i], y2[i] + x[i, 3]/2 * cos[i]],
#              [x1[i] - x[i, 3]/2 * sin[i], y1[i] + x[i, 3]/2 * cos[i]],
#              ])

#     # list.append(points)
#     return list

def show_box(image, boxes):
    for box in boxes:
        cv2.line(image, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), (0, 0, 255), 2)
        cv2.line(image, (int(box[1][0]), int(box[1][1])), (int(box[2][0]), int(box[2][1])), (255, 0, 0), 2)
        cv2.line(image, (int(box[2][0]), int(box[2][1])), (int(box[3][0]), int(box[3][1])), (0, 0, 255), 2)
        cv2.line(image, (int(box[3][0]), int(box[3][1])), (int(box[0][0]), int(box[0][1])), (255, 0, 0), 2)
    cv2.imshow("image", image)



def show_processed_image(image, boxes, count, index):
    # draw = ImageDraw.Draw(image)
    # draw.line([(50, 50), (200, 200)], fill=(255, 0, 0))
    # image.show()
    for box in boxes:
        # print("box : ", box)
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[2]/2
        angle = box[3]

        anglePi = angle  #-angle * math.pi / 180.0
        if(torch.is_tensor(anglePi)):
            cos = math.cos(anglePi)
            sin = math.sin(anglePi)
        else:
            cos = math.cos(anglePi)
            sin = math.sin(anglePi)


        x1 = x - width / 2 * cos # x-w*cos theta/2
        y1 = y + width / 2 * sin # y+w*sin theta/2
        x2 = x + width / 2 * cos # x+w*cos theta/2
        y2 = y - width / 2 * sin # x-w*sin theta/2
        # x1 = x - 0.5 * width
        # y1 = y - 0.5 * height

        # x0 = x + 0.5 * width
        # y0 = y1

        # x2 = x1
        # y2 = y + 0.5 * height

        # x3 = x0
        # y3 = y2

        x0n = x1 - height/2 * sin
        y0n = y1 - height/2 * cos

        x1n = x2 - height/2 * sin
        y1n = y2 - height/2 * cos

        x2n = x2 + height/2 * sin
        y2n = y2 + height/2 * cos

        x3n = x1 + height/2 * sin
        y3n = y1 + height/2 * cos

        cv2.line(image, (int(x0n), int(y0n)), (int(x1n), int(y1n)), (0, 0, 255), 2)
        cv2.line(image, (int(x1n), int(y1n)), (int(x2n), int(y2n)), (255, 0, 0), 2)
        cv2.line(image, (int(x2n), int(y2n)), (int(x3n), int(y3n)), (0, 0, 255), 2)
        cv2.line(image, (int(x0n), int(y0n)), (int(x3n), int(y3n)), (255, 0, 0), 2)
    # cv2.imshow("image", image)
    cv2.imwrite("img_{}_{}.jpg".format(count, index), image)
    # plt.show()
    pass

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


# def iou_cal(pre, tar):
#     if abs(tar[8] - pre[8]) > 1/6 * math.pi:
#         return 0
#     x1, y1 = np.array([tar[0],tar[2],tar[4],tar[6]]), np.array([tar[1],tar[3],tar[5],tar[7]])
#     x2, y2 = np.array([pre[0],pre[2],pre[4],pre[6]]), np.array([pre[1],pre[3],pre[5],pre[7]])
#     rr1, cc1 = polygon(x1, y1)
#     rr2, cc2 = polygon(x2, y2)
#     try:
#         r_max = max(rr1.max(), rr2.max()) + 1
#         c_max = max(cc1.max(), cc2.max()) + 1
#     except:
#         return 0
#     canvas = np.zeros((r_max, c_max))
#     canvas[rr1, cc1] += 1
#     canvas[rr2, cc2] += 1
#     union = np.sum(canvas > 0)
#     if union == 0:
#         return 0
#     intersection = np.sum(canvas == 2)
#     iou = intersection/union
#     return iou
def is_in_poly(p, poly):
    px, py = p[..., 0], p[..., 1]
    is_in = torch.zeros(p.shape[0],p.shape[0]).to(torch_utils.select_device())
    for i in range(4):
        next_i = i + 1 if i + 1 < 4 else 0 
        x1, y1 = poly[i]
        x2, y2 = poly[next_i]
        less = torch.lt(py, max(y1,y2))
        more = torch.gt(py, min(y1,y2))
        mid = less.eq(more)
        x = x1 + (py-y1) * (x2-x1) / (y2-y1)
        left = torch.lt(px, x)
        all = mid.eq(left).type(torch.int)
        is_in = is_in + all
    return is_in==1

def iou_cal(pre,  tar, device):
    if abs(tar[4][0] - pre[4][0]) > 1/6 * math.pi:
        return 0
    x, y = torch.meshgrid(torch.arange(100,dtype=torch.float32),torch.arange(100, dtype=torch.float32))
    mesh = torch.stack((x,y),2).to(device)
    canvas_pre = is_in_poly(mesh,pre)
    canvas_tar = is_in_poly(mesh,tar)
    intersection = torch.logical_and(canvas_pre,canvas_tar).sum()
    union = torch.logical_or(canvas_pre,canvas_tar).sum()
    if union==0:
        return 0
    iou = intersection/union
    return iou


def max_iou(pre_g, tar_g, device):
    max_iou = 0
    for gr in tar_g:
        iou = iou_cal(pre_g, gr, device)
        max_iou = max(max_iou, iou)
    
    return max_iou

# def scale_coords(img_size, coords, img0_shape):
#     # Rescale x1, y1, x2, y2 from 416 to image size
#     gain = float(img_size) / max(img0_shape)  # gain  = old / new
#     pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
#     pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
#     coords[:, [0, 2]] -= pad_x
#     coords[:, [1, 3]] -= pad_y
#     coords[:, :4] /= gain
#     coords[:, :4] = torch.clamp(coords[:, :4], min=0)
#     return coords

def scale_coords(img_size, coords, img0_shape):
    # Rescale x, y, w, h from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    coords[:, 0] -= pad_x
    coords[:, 1] -= pad_y
    coords[:, :3] /= gain
    # coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords

def scale_coord1(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2, x3, y3, x4, y4 from 416 to image size
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    for grasp in coords:
        grasp[0][0] -= pad_x
        grasp[0][0] /= gain
        grasp[0][1] -= pad_y
        grasp[0][1] /= gain
        grasp[1][0] -= pad_x
        grasp[1][0] /= gain
        grasp[1][1] -= pad_y
        grasp[1][1] /= gain
        grasp[2][0] -= pad_x
        grasp[2][0] /= gain
        grasp[2][1] -= pad_y
        grasp[2][1] /= gain
        grasp[3][0] -= pad_x
        grasp[3][0] /= gain
        grasp[3][1] -= pad_y
        grasp[3][1] /= gain
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou

def angel_match(anchor, tar):
    ang_a = anchor/180 * math.pi
    diff = torch.abs(ang_a-tar)
    return diff



def compute_loss(p, targets):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    loss, lxy, lwh, ltheta, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, ttheta, tconf, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    # BCE = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5],device='cuda'))

    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    # pos = [x.sum() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy

        # Compute losses
        # nT = pi0.size(2)
        k = 1  # nT / bs
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            lxy += k * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy
            lwh += k * MSE(pi[..., 2], twh[i])  # wh
            ltheta += k * MSE(pi[..., 3], ttheta[i])
            # lconf += (k * 16) * BCE(pi[..., 5], tconf[i][b, a, gj, gi])

        # pos_weight = FT([((gp[i]-pos[i]) / pos[i]).round()])
        # pos_weight = FT([gp[i] / min(gp) * 5.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * 8) * BCE(pi0[..., 4], tconf[i])
        # lconf += BCE(pi0[..., 4], tconf[i])
    loss =  lxy + lwh + ltheta + lconf

    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), ltheta.item()]
    for name, x in zip(['total', 'xy', 'wh', 'conf', 'ltheta'], losses):
        d[name] = x

    return loss, d


def build_targets(model, targets, pred):
    # targets = [image, class, x, y, w, h]
    # targets = [image, x, y, w, h, theta]
    if isinstance(model, nn.DataParallel):
        model = model.module
    yolo_layers = get_yolo_layers(model)

    # anchors = closest_anchor(model, targets)  # [layer, anchor, i, j]
    txy, twh, ttheta, tconf, indices = [], [], [], [], []
    for i, layer in enumerate(yolo_layers):
        nG = model.module_list[layer][0].nG  # grid size
        anchor_w = model.module_list[layer][0].anchor_w
        anchor_ang = model.module_list[layer][0].anchor_ang

        # iou of targets-anchors
        # gwh = targets[:, 3:5] * nG
        # gtheta = targets[:, 5]
        gwh = targets[:, 3] * nG
        gtheta = targets[:, 4]
        # iou = [wh_iou(x, gwh) for x in anchor_vec]
        angel_diff = [angel_match(x, gtheta) for x in anchor_ang]
        # iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor
        best_angle, a = torch.stack(angel_diff, 0).min(0)  # best iou and anchor

        # reject below threshold ious (OPTIONAL)
        reject = False
        if reject:
            j = best_angle < 15/180 *math.pi
            t, a, gwh = targets[j], a[j], gwh[j]
        else:
            t = targets

        # Indices
        # b, c = t[:, 0:2].long().t()  # target image, class
        b = t[:, 0].long().t()  # target image
        gxy = t[:, 1:3] * nG
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))

        # XY coordinates
        txy.append(gxy - gxy.floor())

        # Width and height
        # anchor_wh = torch.full((anchor_vec.size(0), anchor_vec.size(1)), anchor_vec[0,0]).to(torch_utils.select_device())
        # print(anchor_w[a])
        twh.append(torch.log(gwh / anchor_w[a]))  # yolo method
        # twh.append(torch.sqrt(gwh / anchor_vec[a]) / 2)  # power method

        # exam = anchor_vec[a][1]

        # Class
        # print(anchor_ang[a])
        ttheta.append((t[:, 4]-anchor_ang[a]/180 *math.pi)/(15/180 * math.pi))

        # Conf
        tci = torch.zeros_like(pred[i][..., 0])
        tci[b, a, gj, gi] = 1  # conf
        tconf.append(tci)

    return txy, twh, ttheta, tconf, indices


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Filter out confidence scores below threshold
        class_prob, class_pred = torch.max(F.softmax(pred[:, 5:], 1), 1)
        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]
        class_prob = class_prob[v]
        class_pred = class_pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_prob, class_pred)
        detections = torch.cat((pred[:, :5], class_prob.float().unsqueeze(1), class_pred.float().unsqueeze(1)), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique().to(prediction.device)

        nms_style = 'OR'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in unique_labels:
            # Get the detections with class c
            dc = detections[detections[:, -1] == c]
            # Sort the detections by maximum object confidence
            _, conf_sort_index = torch.sort(dc[:, 4] * dc[:, 5], descending=True)
            dc = dc[conf_sort_index]

            # Non-maximum suppression
            det_max = []
            ind = list(range(len(dc)))
            if nms_style == 'OR':  # default
                while len(ind):
                    j = ind[0]
                    det_max.append(dc[j:j + 1])  # save highest conf detection
                    reject = bbox_iou(dc[j], dc[ind]) > nms_thres
                    [ind.pop(i) for i in reversed(reject.nonzero())]
                # while dc.shape[0]:  # SLOWER METHOD
                #     det_max.append(dc[:1])  # save highest conf detection
                #     if len(dc) == 1:  # Stop if we're at the last detection
                #         break
                #     iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                #     dc = dc[1:][iou < nms_thres]  # remove ious > threshold

                # Image      Total          P          R        mAP
                #  4964       5000      0.629      0.594      0.586

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc) > 0:
                    iou = bbox_iou(dc[0], dc[0:])  # iou with other boxes
                    i = iou > nms_thres

                    weights = dc[i, 4:5] * dc[i, 5:6]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[iou < nms_thres]

                # Image      Total          P          R        mAP
                #  4964       5000      0.633      0.598      0.589  # normal

            if len(det_max) > 0:
                det_max = torch.cat(det_max)
                # Add max detections to outputs
                output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nC = 80  # number classes
    x = np.zeros(nC, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nC)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def plot_results(start=0):
    # Plot YOLO training results file 'results.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v3.txt')
    # from utils.utils import *; plot_results()

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'Precision', 'Recall', 'mAP']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(results[i, x[start:]], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
