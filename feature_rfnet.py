# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:20
# @Author  : xylon

import config
config.cfg.set_lib('rfnet',prepend=True) 

import cv2
import torch
import random
import argparse
import numpy as np

from rfnet.utils.common_utils import gct
from rfnet.utils.eval_utils import nearest_neighbor_distance_ratio_match
from rfnet.model.rf_des import HardNetNeiMask
from rfnet.model.rf_det_so import RFDetSO
from rfnet.model.rf_net_so import RFNetSO
from rfnet.config import cfg
from rfnet.utils.image_utils import im_rescale


def to_cv2_kp(kp):
    # kp is like [batch_idx, y, x, channel]
    return cv2.KeyPoint(kp[2], kp[1], 0)


# interface for pySLAM
class RfNetFeature2D: 
    def __init__(self,
                 num_features=2000, ):  
        print('Using RfNetFeature2D')   

        cfg.TRAIN.TOPK = num_features


        random.seed(cfg.PROJ.SEED)
        torch.manual_seed(cfg.PROJ.SEED)
        np.random.seed(cfg.PROJ.SEED)

        det = RFDetSO(
            cfg.TRAIN.score_com_strength,
            cfg.TRAIN.scale_com_strength,
            cfg.TRAIN.NMS_THRESH,
            cfg.TRAIN.NMS_KSIZE,
            cfg.TRAIN.TOPK,
            cfg.MODEL.GAUSSIAN_KSIZE,
            cfg.MODEL.GAUSSIAN_SIGMA,
            cfg.MODEL.KSIZE,
            cfg.MODEL.padding,
            cfg.MODEL.dilation,
            cfg.MODEL.scale_list,
        )
        des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
        model = RFNetSO(
            det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
        )

        use_gpu = torch.cuda.is_available()

        if use_gpu:
            print(f"{gct()} : to device GPU")
            device = torch.device("cuda")
        else:
            print(f"{gct()} : to device CPU")
            device = torch.device("cpu")

        model = model.to(device)

        resume = config.cfg.root_folder + '/thirdparty/rfnet' + '/weights/e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar'

        print(f"{gct()} : in {resume}")

        if use_gpu:
            checkpoint = torch.load(resume)
        else:
            checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        self.model = model
        self.device = device

    def to_cv2_kp(self, kp):
        # kp is like [batch_idx, y, x, channel]
        return cv2.KeyPoint(int(kp[2]/self.sw), int(kp[1]/self.sh), 0)

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        # frame is gray scale image
        # device: cuda or cpu
        # output_size: resacle size
        # return: kp (#keypoints, 4) des (#keypoints, 128)

        device = self.device

        #output_size = (240, 320) #h, w
        output_size = (190, 620) #h, w
        #output_size = (95, 310) #h, w
        img_raw = img = np.expand_dims(frame, -1)

        # Rescale
        img, _, _, sw, sh = im_rescale(img, output_size)
        self.sw = sw
        self.sh = sh
        img_info = np.array([sh, sw])
        #print('img_info: ', img_info)

        # to tensor
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = torch.from_numpy(img.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )
        img_info = torch.from_numpy(img_info)[None, :].to(device, dtype=torch.float)
        img_raw = torch.from_numpy(img_raw.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )

        # inference
        _, kp, des = self.model.inference(img, img_info, img_raw)

        keypoints = list(map(self.to_cv2_kp, kp))
        descriptors = des.cpu().detach().numpy()

        return keypoints, descriptors

