#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    
call this function once the partition and inference was made to upsample
the prediction to the original point clouds
"""
import os.path
import glob
import numpy as np
import argparse
from provider import *
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3D')
parser.add_argument('--odir', default='./results/semantic3d', help='Directory to store results')
parser.add_argument('--ver_batch', default=5000000, type=int, help='Batch size for reading large files')
parser.add_argument('--db_test_name', default='testred')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.SEMA3D_PATH+'/'
#list of subfolders to be processed
if args.db_test_name == 'testred':
    area = 'test_reduced/'
elif args.db_test_name == 'testfull':
    area = 'test_full/'
#------------------------------------------------------------------------------
print("=================\n   " + area + "\n=================")
data_folder = root + "data/"               + area
fea_folder  = root + "features/"           + area
spg_folder  = root + "superpoint_graphs/"           + area
res_folder  = './' + args.odir + '/'
labels_folder =  root + "labels/"          + area
if not os.path.isdir(data_folder):
    raise ValueError("%s do not exists" % data_folder)
if not os.path.isdir(fea_folder):
    raise ValueError("%s do not exists" % fea_folder)
if not os.path.isdir(res_folder):
    raise ValueError("%s do not exists" % res_folder)  
if not os.path.isdir(root + "labels/"):
    os.mkdir(root + "labels/")   
if not os.path.isdir(labels_folder):
    os.mkdir(labels_folder)   
try:    
    res_file = h5py.File(res_folder + 'predictions_' + args.db_test_name + '.h5', 'r')   
except OSError:
    raise ValueError("%s do not exists" % res_file) 
    
files = glob.glob(data_folder+"*.txt")    
if (len(files) == 0):
    raise ValueError('%s is empty' % data_folder)
n_files = len(files)
i_file = 0
for file in files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    file_name_short = '_'.join(file_name.split('_')[:2])
    data_file  = data_folder + file_name + ".txt"
    fea_file   = fea_folder  + file_name_short + '.h5'
    spg_file   = spg_folder  + file_name_short + '.h5' 
    label_file = labels_folder + file_name_short + ".labels"
    i_file = i_file + 1
    print(str(i_file) + " / " + str(n_files) + "---> "+file_name_short)
    print("    reading the subsampled file...")
    geof, xyz, rgb, graph_nn, l = read_features(fea_file)
    graph_sp, components, in_component = read_spg(spg_file)
    n_ver = xyz.shape[0]
    del geof, rgb, graph_nn, l, graph_sp
    labels_red = np.array(res_file.get(area + file_name_short))
    print("    upsampling...")
    labels_full = reduced_labels2full(labels_red, components, n_ver)
    labels_ups = interpolate_labels_batch(data_file, xyz, labels_full, args.ver_batch)
    np.savetxt(label_file, labels_ups+1, delimiter=' ', fmt='%d')   # X is an array
