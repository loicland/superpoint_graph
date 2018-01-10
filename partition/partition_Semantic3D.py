#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
import os.path
import glob
import sys
import numpy as np
import argparse
import h5py
from graphs import *
from provider import *
from timeit import default_timer as timer
sys.path.append("./cut-pursuit/src")
sys.path.append("./ply_c")
import libcp
import libply_c
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3D')

#parameters
parser.add_argument('--ver_batch', default=5000000, type=int, help='Batch size for reading large files')
parser.add_argument('--voxel_width', default=0.05, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=.8, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=10, type=float, help='max length of super edges')
parser.add_argument('--n_labels', default=8, type=int, help='number of classes')
args = parser.parse_args()

#path to data
root = args.SEMA3D_PATH+'/'
#list of subfolders to be processed
areas = ["test_reduced/", "test_full/", "train/"]
#------------------------------------------------------------------------------
num_area = len(areas)
times = [0,0,0]
if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")
confusion_matrix = np.array([num_area, 1])
for area in areas:
    print("=================\n   " + area + "\n=================")
    data_folder = root + "data/"               + area
    ply_folder  = root + "clouds/"             + area
    fea_folder  = root + "features/"           + area
    spg_folder  = root + "/superpoint_graphs/" + area
    if not os.path.isdir(data_folder):
        raise ValueError("%s do not exists" % data_folder)
    if not os.path.isdir(ply_folder):
        os.mkdir(ply_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)
    files = glob.glob(data_folder+"*.txt")    
    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
    n_files = len(files)
    i_file = 0
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        file_name_short = '_'.join(file_name.split('_')[:2])
        data_file  = data_folder + file_name + ".txt"
        label_file = data_folder + file_name + ".labels"
        ply_file   = ply_folder  + file_name_short
        fea_file   = fea_folder  + file_name_short + '.h5'
        spg_file   = spg_folder  + file_name_short + '.h5' 
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> "+file_name_short)
        #--- build the geometric feature file h5 file ----------------------
        if os.path.isfile(fea_file):
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
            has_labels = len(labels)>0
        else :
            print("    creating the feature file...")
            start = timer()
            has_labels = (os.path.isfile(label_file))
            #---retrieving and subsampling the point clouds---
            if (has_labels):
                 xyz, rgb, labels = prune_labels(data_file, label_file, args.ver_batch, args.voxel_width, args.n_labels)
            else:
                 xyz, rgb = prune(data_file, args.ver_batch, args.voxel_width)
                 labels = []
            #---computing the nn graphs---
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
                    #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file):
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            start = timer()
            geof[:,3] = 2. * geof[:, 3] #increase importance of verticality (heuristic)
            graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(geof, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype = 'object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, args.n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
         #write various point cloud, uncomment for vizualization
        #write_ply_obj(ply_file + "_labels.ply", xyz, rgb, labels, room_object_indices)
        #prediction2ply(ply_file + "_ground_truth.ply", xyz, labels)
        #geof2ply(ply_file + "_geof.ply", xyz, geof)
        #partition2ply(ply_file + "_partition.ply", xyz, components)
