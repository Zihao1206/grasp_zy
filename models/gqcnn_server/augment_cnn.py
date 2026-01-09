""" CNN for network augmentation """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gqcnn_server.augment_cells import AugmentCell
from models.gqcnn_server import ops
from utils.parse_config import *
import math
import models.genotypes as gt

class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, cfg, input_size, C_in, C, n_layers, auxiliary, genotype,
                 stem_multiplier=2):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        module_def = parse_data_cfg(cfg)
        anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
        anchors = [float(x) for x in module_def['anchors'].split(',')]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.C_in = C_in
        self.C = C
        # self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        # aux head position
        self.aux_pos = 2*n_layers//3 if auxiliary else -1

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C_cur)
        )

        # C_pp, C_p, C_cur = C_cur, C_cur, C_cur
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.ModuleList()
        reduction_p = False
        self.stride = 1
        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                self.stride *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)

        # self.pos_out = nn.ConvTranspose2d(C_p, 1, kernel_size=4, stride=4)
        # self.cos_out = nn.ConvTranspose2d(C_p, 1, kernel_size=4, stride=4)
        # self.sin_out = nn.ConvTranspose2d(C_p, 1, kernel_size=4, stride=4)
        # self.width_out = nn.ConvTranspose2d(C_p, 1, kernel_size=4, stride=4)
        self.anchors[:, 0] = self.anchors[:, 0] / self.stride
        self.anchor_ang = self.anchors[:, 1].to(self.device)
        self.anchor_w = torch.flatten(self.anchors[:, 0]).to(self.device)
        self.anchor_wh = self.anchor_w.view(1, self.nA, 1, 1).to(self.device)
        self.anchor_a = self.anchor_ang.view(1, self.nA, 1, 1).to(self.device)

        # grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        # grid_y = grid_x.permute(0, 1, 3, 2)
        # self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)

        self.out = nn.Conv2d(C_p, 30, kernel_size=1)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            # s1 = cell(s0)
            # s0 = s1
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        # out = self.gap(s1)
        out = s1
        # out = out.view(out.size(0), -1) # flatten
        # logits = self.linear(out)
        # pos_out = self.pos_out(out)
        # cos_out = self.cos_out(out)
        # sin_out = self.sin_out(out)
        # width_out = self.width_out(out)
        self.nG = torch.FloatTensor([out.size(-1)]).to(self.device)
        # stride = 100 / self.nG
        # out = out.view(out.size(0), -1) # flatten
        # pos_out = self.pos_out(out)
        # cos_out = self.cos_out(out)
        # sin_out = self.sin_out(out)
        # width_out = self.width_out(out)

        p = self.out(out)
        p = p.view(out.size(0), self.nA, 5, out.size(-1), out.size(-1)).permute(0, 1, 3, 4, 2).contiguous()

        if self.training:
            return p
        else:
            grid_x = torch.arange(out.size(-1)).repeat((out.size(-1), 1)).view((1, 1, out.size(-1), out.size(-1))).float()
            grid_y = grid_x.permute(0, 1, 3, 2)
            grid_xy = torch.stack((grid_x, grid_y), 4).to(self.device)
            p[..., 0:2] = torch.sigmoid(p[..., 0:2]) + grid_xy  # xy
            p[..., 2] = torch.exp(p[..., 2]) * self.anchor_wh  # w method
            # p[..., 2:4] = ((torch.sigmoid(p[..., 2:4]) * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 3] = (p[..., 3] * 15 + self.anchor_a) *math.pi / 180
            p[..., 4] = torch.sigmoid(p[..., 4])  # conf
            p[..., :3] *= self.stride

            return p.view(out.size(0), -1, 5)

        # return pos_out, cos_out, sin_out, width_out
        # return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p

    def angel_match(self, anchor, tar):
        ang_a = anchor/180 * math.pi
        diff = torch.abs(ang_a-tar)
        return diff

    def compute_loss(self, xc, yc):
        # xc = self.forward(X)
        # y_pos, y_cos, y_sin, y_width = yc
        # pos_pred, cos_pred, sin_pred, width_pred = self.forward(xc)

        # p_loss = F.mse_loss(pos_pred, y_pos)
        # cos_loss = F.mse_loss(cos_pred, y_cos)
        # sin_loss = F.mse_loss(sin_pred, y_sin)
        # width_loss = F.mse_loss(width_pred, y_width)

        # return {
        #     'loss': p_loss + cos_loss + sin_loss + width_loss,
        #     'losses': {
        #         'p_loss': p_loss,
        #         'cos_loss': cos_loss,
        #         'sin_loss': sin_loss,
        #         'width_loss': width_loss
        #     },
        #     'pred': {
        #         'pos': pos_pred,
        #         'cos': cos_pred,
        #         'sin': sin_pred,
        #         'width': width_pred
        #     }
        # }
        pred = self.forward(xc)
        txy, twh, ttheta, tconf, indices = [], [], [], [], []
        nG = self.nG  # grid size
        anchor_w = self.anchor_w
        anchor_ang = self.anchor_ang

        # iou of targets-anchors
        # gwh = targets[:, 3:5] * nG
        # gtheta = targets[:, 5]
        gwh = yc[:, 3] * nG
        gtheta = yc[:, 4]
        # iou = [wh_iou(x, gwh) for x in anchor_vec]
        angel_diff = [self.angel_match(x, gtheta) for x in anchor_ang]
        # iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor
        best_angle, a = torch.stack(angel_diff, 0).min(0)  # best iou and anchor

        # reject below threshold ious (OPTIONAL)
        reject = False
        if reject:
            j = best_angle < 15/180 *math.pi
            t, a, gwh = yc[j], a[j], gwh[j]
        else:
            t = yc
        b = t[:, 0].long().t()  # target image
        gxy = t[:, 1:3] * nG
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))

        # XY coordinates
        txy.append(gxy - gxy.floor())
        twh.append(torch.log(gwh / anchor_w[a]))  # yolo method
        ttheta.append((t[:, 4]-anchor_ang[a]/180 *math.pi)/(15/180 * math.pi))
        tci = torch.zeros_like(pred[..., 0])
        tci[b, a, gj, gi] = 1  # conf
        tconf.append(tci)

        FT = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor
        loss, lxy, lwh, ltheta, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
        MSE = nn.MSELoss()
        CE = nn.CrossEntropyLoss()
        BCE = nn.BCEWithLogitsLoss()

        b, a, gj, gi = indices[0]  # image, anchor, gridx, gridy

        k = 1  # nT / bs
        if len(b) > 0:
            pi = pred[b, a, gj, gi]  # predictions closest to anchors
            lxy += k * MSE(torch.sigmoid(pi[..., 0:2]), txy[0])  # xy
            lwh += k * MSE(pi[..., 2], twh[0])  # wh
            ltheta += k * MSE(pi[..., 3], ttheta[0])
        # lconf += (k * 16) * BCE(pi[..., 5], tconf[i][b, a, gj, gi])

    # pos_weight = FT([((gp[i]-pos[i]) / pos[i]).round()])
    # pos_weight = FT([gp[i] / min(gp) * 5.])
    # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * 8) * BCE(pred[..., 4], tconf[0])
        # lconf += BCE(pi0[..., 4], tconf[i])
        loss =  lxy + lwh + ltheta + lconf
        return {
            'loss': loss
        }


if __name__ == 'main':
    import time
    file_path = '/home/jet/zoneyung/grasp_static/single_zy.txt'
    # file_path = 'C:/grasp_static/single_zy.txt'
    # file_path = '/home/jet/zoneyung/grasp_static/single_rgb.txt'
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            gene = line
            getypr = gt.from_str(gene)
    model = AugmentCNN('/home/jet/zoneyung/grasp_static/cornell.data', 100, 3, 8, 5, False, getypr).cuda()
    model.load_state_dict(torch.load('/home/jet/zoneyung/grasp_static/weights/tune_epoch_64_loss_0.0297_accuracy_1.000'))
    model.eval()

    input = torch.randn(1, 3, 100, 100).cuda()
    with torch.no_grad():
        for _ in range(10):
            _ = model(input)
    
    num_runs = 100
    totoal = 0
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input)
            end_time = time.time()
            totoal += (end_time - start_time)
    
    average = totoal / num_runs
    print(f"Average inference time: {average * 1000:.2f} ms")
