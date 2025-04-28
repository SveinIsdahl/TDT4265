#Inspired by https://github.com/amdegroot/ssd.pytorch and assignment 4 in TDT4265

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_block(nn.Module):
    def forward(self, x):
        return self.block(x)
    
    def __init__(self, in_channels, out_channels, num_conv):
        super().__init__()
        layers = []
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels 
        self.block = nn.Sequential(*layers)

    

class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = VGG16_block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.block2 = VGG16_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = VGG16_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = VGG16_block(256, 512, 3)
        self.block5 = VGG16_block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x) 
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        feature1 = x
        x = self.block5(x)
        x = self.pool5(x)
        return feature1, x

class ExtraLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(1024, 256, kernel_size=1)   
        self.relu8 = nn.ReLU(inplace=True) 
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(512, 128, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(256, 128, kernel_size=1)
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 

    def forward(self, x):
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        feature2 = x

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        feature3 = x

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.conv11(x)
        x = self.relu11(x)
        feature4 = x

        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        feature5 = x 

        return feature2, feature3, feature4, feature5

class PredictionHead(nn.Module):
    def __init__(self, in_channels, num_priors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loc_head = nn.Conv2d(in_channels, num_priors * 4, kernel_size=3, padding=1)
        self.conf_head = nn.Conv2d(in_channels, num_priors * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        loc = self.loc_head(x)
        conf = self.conf_head(x) 
        loc = loc.permute(0, 2, 3, 1).contiguous() #TODO: Check these value, seems like error could be from here
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.permute(0, 2, 3, 1).contiguous()
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return loc, conf

class SSD(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.vgg_backbone = VGG16Backbone()
        self.extra_layers = ExtraLayers()

        self.num_priors = [4, 6, 6, 6, 4, 4]

        self.pred_heads = nn.ModuleList()
        in_channels = [512, 1024, 512, 256, 256, 256]

        for i, channels in enumerate(in_channels):
             self.pred_heads.append(PredictionHead(channels, self.num_priors[i], num_classes))

        self.feature_maps = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self, x):
        feature1, x = self.vgg_backbone(x)
        feature2, feature3, feature4, feature5 = self.extra_layers(x)

        features = [feature1, feature2, feature3, feature4, feature5, F.avg_pool2d(feature5, kernel_size=3, stride=1, padding=1)] # Added average pooling for the last 1x1 layer if needed

        locations = []
        confidences = []

        for i, feature_map in enumerate(features):
            loc, conf = self.pred_heads[i](feature_map)
            locations.append(loc)
            confidences.append(conf)

        locations = torch.cat(locations, dim=1)
        confidences = torch.cat(confidences, dim=1)

        return locations, confidences

    def generate_anchors(self, img_size):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            sk = self.min_sizes[k] / img_size

            if k == len(self.feature_maps) - 1:
                 sk_prime = self.max_sizes[k] / img_size
                 s_prime_k = (sk * sk_prime)**0.5
            else:
                 sk_prime = self.max_sizes[k] / img_size
                 s_prime_k = (sk * sk_prime)**0.5

            for i, j in [(i, j) for i in range(f[0]) for j in range(f[1])]:
                cx = (j + 0.5) / f[1]
                cy = (i + 0.5) / f[0]

                anchors.append([cx, cy, sk, sk])
                #TODO: Fix anchor generation

                if k < len(self.feature_maps) - 1:
                     anchors.append([cx, cy, s_prime_k, s_prime_k])


        anchors = torch.tensor(anchors)
        anchors[:, :2] -= anchors[:, 2:] / 2
        anchors[:, 2:] += anchors[:, :2]
        return anchors

# Modified from: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD/
import torch
import tops
import torch.nn.functional as F
from ssd import utils
from typing import Optional

def calc_iou_tensor(box1_ltrb, box2_ltrb):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-src
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """

    N = box1_ltrb.size(0)
    M = box2_ltrb.size(0)

    be1 = box1_ltrb.unsqueeze(1).expand(-1, M, -1)
    be2 = box2_ltrb.unsqueeze(0).expand(N, -1, -1)

    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou

# This is from https://github.com/kuangliu/pytorch-ssd.
class AnchorEncoder(object):
    def __init__(self, anchors):
        self.anchors = anchors(order="ltrb")
        self.anchors_xywh = tops.to_cuda(anchors(order="xywh").unsqueeze(dim=0))
        self.nboxes = self.anchors.size(0)
        self.scale_xy = anchors.scale_xy
        self.scale_wh = anchors.scale_wh

    def encode(self, bboxes_in: torch.Tensor, labels_in: torch.Tensor, iou_threshold: float):

        ious = calc_iou_tensor(bboxes_in, self.anchors)
        #ious: shape [batch_size, num_anchors]
        best_target_per_anchor, best_target_per_anchor_idx = ious.max(dim=0)
        best_anchor_per_target, best_anchor_per_target_idx = ious.max(dim=1)

        # 2.0 is used to make sure every target has a prior assigned
        best_target_per_anchor.index_fill_(0, best_anchor_per_target_idx, 2.0)

        idx = torch.arange(0, best_anchor_per_target_idx.size(0), dtype=torch.int64)
        best_target_per_anchor_idx[best_anchor_per_target_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_target_per_anchor > iou_threshold
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_target_per_anchor_idx[masks]]
        bboxes_out = self.anchors.clone()
        bboxes_out[masks, :] = bboxes_in[best_target_per_anchor_idx[masks], :]
        # Transform format to xywh format
        bboxes_out = utils.bbox_ltrb_to_center(bboxes_out)
        return bboxes_out, labels_out

    def decode_output(self, bbox_delta: torch.Tensor, confs_in: Optional[torch.Tensor]):
        bbox_delta = bbox_delta.permute(0, 2, 1)

        bbox_delta[:, :, :2] = self.scale_xy*bbox_delta[:, :, :2]
        bbox_delta[:, :, 2:] = self.scale_wh*bbox_delta[:, :, 2:]

        bbox_delta[:, :, :2] = bbox_delta[:, :, :2]*self.anchors_xywh[:, :, 2:] + self.anchors_xywh[:, :, :2]
        bbox_delta[:, :, 2:] = bbox_delta[:, :, 2:].exp()*self.anchors_xywh[:, :, 2:]

        boxes_ltrb = utils.bbox_center_to_ltrb(bbox_delta)
        if confs_in is not None:
            confs_in = confs_in.permute(0, 2, 1)
            confs_in = F.softmax(confs_in, dim=-1)
        return boxes_ltrb, confs_in