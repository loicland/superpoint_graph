#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:57:18 2017

@author: landrieuloic
"""

#------------------------------------------------------------------------------
#--- Loic landrieu Dec 2017 ---------------------------------------------------
#------------------------------------------------------------------------------
import os.path
import glob
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from timeit import default_timer as timer
sys.path.append("./cut-pursuit/src")
sys.path.append("./ply_c")
import libcp
import libply_c
from graphs import *
from provider import *
ver_batch = 5000000 #batch size when reading the large ply size
voxel_width = 0.05 #voxel size when subsampling (in m)
k_nn_geof = 45 #number of neighbors for the geometric features
k_nn_adj = 10 #adjacency structure for the minimal partition
lambda_edge_weight = 1. #parameter determine the edge weight for minimal part.
reg_strength = 1 #regularization strength for the minimal partition
d_se_max = 10 #max length of super edges
n_labels = 8
#root of the data directory
root = "/media/landrieuloic/Data/semantic3d/"
#list of subfolders to be processed
areas = ["reduced_set/", "test_set/", "train_set/"]
areas = ["training_set/", "reduced_set/"]
areas = ["training_set/"]
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
        data_file  = data_folder + file_name + ".txt"
        label_file = data_folder + file_name + ".labels"
        ply_file   = ply_folder  + file_name
        fea_file   = fea_folder  + file_name + '.h5'
        spg_file   = spg_folder  + file_name + '.h5' 
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> "+file_name)
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
                 xyz, rgb, labels = prune_labels(data_file, label_file, ver_batch, voxel_width, n_labels)
            else:
                 xyz, rgb = prune(data_file, ver_batch, voxel_width)
                 labels = []
            #---computing the nn graphs---
            graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
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
            features = np.hstack((geof, rgb / 255.)).astype('float32')
            features[:,3] = 2. * features[:, 3] #increase importance of verticality (heuristic)
            graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], reg_strength)
            components = np.array(components, dtype = 'object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, labels, n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
         #write various point cloud, uncomment for vizualization
        #write_ply_obj(ply_file + "_labels.ply", xyz, rgb, labels, room_object_indices)
        #prediction2ply(ply_file + "_ground_truth.ply", xyz, labels)
        #geof2ply(ply_file + "_geof.ply", xyz, geof)
        #partition2ply(ply_file + "_partition.ply", xyz, components)
