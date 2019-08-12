#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:22:55 2019

@author: landrieuloic
"""

"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
import argparse
import numpy as np
import sys
sys.path.append("./learning")
from metrics import *

parser = argparse.ArgumentParser(description='Evaluation function for S3DIS')

parser.add_argument('--odir', default='./results/s3dis/best', help='Directory to store results')
parser.add_argument('--dataset', default='s3dis', help='Directory to store results')
parser.add_argument('--cvfold', default='123456', help='which fold to consider')

args = parser.parse_args()



if args.dataset == 's3dis':
    n_labels = 13
    inv_class_map = {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'}
    base_name = args.odir+'/cv'
elif args.dataset == 'vkitti':
    n_labels = 13
    inv_class_map = {0:'Terrain', 1:'Tree', 2:'Vegetation', 3:'Building', 4:'Road', 5:'GuardRail', 6:'TrafficSign', 7:'TrafficLight', 8:'Pole', 9:'Misc', 10:'Truck', 11:'Car', 12:'Van'}
    base_name = args.odir+'/cv'
    
C = ConfusionMatrix(n_labels)
C.confusion_matrix=np.zeros((n_labels, n_labels))


for i_fold in range(len(args.cvfold)):
    fold = int(args.cvfold[i_fold])
    cm = ConfusionMatrix(n_labels)
    cm.confusion_matrix=np.load(base_name+str(fold) +'/pointwise_cm.npy')
    print("Fold %d : \t OA = %3.2f \t mA = %3.2f \t mIoU = %3.2f" % (fold, \
        100 * ConfusionMatrix.get_overall_accuracy(cm) \
      , 100 * ConfusionMatrix.get_mean_class_accuracy(cm) \
      , 100 * ConfusionMatrix.get_average_intersection_union(cm)
      ))
    C.confusion_matrix += cm.confusion_matrix
    
print("\nOverall accuracy : %3.2f %%" % (100 * (ConfusionMatrix.get_overall_accuracy(C))))
print("Mean accuracy    : %3.2f %%" % (100 * (ConfusionMatrix.get_mean_class_accuracy(C))))
print("Mean IoU         : %3.2f %%\n" % (100 * (ConfusionMatrix.get_average_intersection_union(C))))
print("         Classe :   IoU")
for c in range(0,n_labels):
    print ("   %12s : %6.2f %% \t %.1e points" %(inv_class_map[c],100*ConfusionMatrix.get_intersection_union_per_class(C)[c], ConfusionMatrix.count_gt(C,c)))
