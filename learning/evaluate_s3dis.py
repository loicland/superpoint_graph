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
parser.add_argument('--cvfold', default='123456', help='which fold to consider')

args = parser.parse_args()

C = ConfusionMatrix
C.number_of_labels = 13
C.confusion_matrix=np.zeros((C.number_of_labels, C.number_of_labels))

class_map = {0:'ceiling', 1:'floor', 2:'wall', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'chair', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'}

if '1' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv1/pointwise_cm.npy')
if '2' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv2/pointwise_cm.npy')
if '3' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv3/pointwise_cm.npy')
if '4' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv4/pointwise_cm.npy')
if '5' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv5/pointwise_cm.npy')
if '6' in args.cvfold:
    C.confusion_matrix+=np.load(args.odir+'/cv6/pointwise_cm.npy')
   
    
    
print("\nOverall accuracy : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_overall_accuracy(C))))
print("Mean accuracy    : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_mean_class_accuracy(C))))
print("Mean IoU         : %3.2f %%\n" % (100 * np.mean(ConfusionMatrix.get_intersection_union_per_class(C))))
print("     Classe :  mIoU")
for c in range(0,C.number_of_labels):
    print ("   %8s : %6.2f %%" %(class_map[c],100*ConfusionMatrix.get_intersection_union_per_class(C)[c]))