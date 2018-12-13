#!/usr/bin/env python

"""Takes a point cloud as input and output the corresponding semantic segmented point cloud."""


import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
sys.path.append('partition/cut-pursuit/src')
sys.path.append('partition/ply_c')
sys.path.append('partition')
import libcp
import libply_c
from graphs import *
from provider import *
sys.path.append('./providers')
from datasets import *

import torch
import torch.nn as nn
from providers.datasets import HelixDataset
from collections import defaultdict
import h5py
import os
from plyfile import PlyData, PlyElement
import open3d as o3d

sys.path.append('learning')
sys.path.append('partition')
import spg
import graphnet
import pointnet
import metrics
import provider
import s3dis_dataset
import custom_dataset

from pcl_segmentation import PointCloudSegmentation


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--input', default='', help='path to the point cloud')
    parser.add_argument('--model_path', default='results/s3dis/bw/cv1/model.pth.tar', help='pretrained model')
    parser.add_argument('--model_config', default='gru_10_0,f_13', help='configuration of the model')
    parser.add_argument('--edge_attribs', default='delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d', help='Edge attribute definition, see spg_edge_features() in spg.py for definitions.')
    parser.add_argument('--pc_attribs', default='xyzelspvXYZ', help='Point attributes fed to PointNets')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

                        
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
                        
    # Creating Model and Loading Weights.
    model = PointCloudSegmentation(args.model_path, args.model_config, args.edge_attribs, args.pc_attribs)
    
    # Initialization
    model.load_model()
    
    # Run Inferences and outputs Semantic Segmented Point Cloud
    model.process(args.input)
    
