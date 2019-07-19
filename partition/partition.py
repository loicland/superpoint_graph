"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c
from graphs import *
from provider import *

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--ROOT_PATH', default='datasets/s3dis')
parser.add_argument('--dataset', default='s3dis', help='s3dis/sema3d/your_dataset')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.1, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
args = parser.parse_args()

#path to data
root = args.ROOT_PATH+'/'
#list of subfolders to be processed
if args.dataset == 's3dis':
    folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
    n_labels = 13
elif args.dataset == 'sema3d':
    folders = ["test_reduced/", "test_full/", "train/"]
    n_labels = 8
elif args.dataset == 'custom_dataset':
    folders = ["train/", "test/"]
    n_labels = 10 #number of classes
else:
    raise ValueError('%s is an unknown data set' % dataset)

times = [0,0,0] #time for computing: features / partition / spg

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

for folder in folders:
    print("=================\n   "+folder+"\n=================")
    
    data_folder = root   + "data/"              + folder
    cloud_folder  = root + "clouds/"            + folder
    fea_folder  = root   + "features/"          + folder
    spg_folder  = root   + "superpoint_graphs/" + folder
    if not os.path.isdir(data_folder):
        raise ValueError("%s does not exist" % data_folder)
        
    if not os.path.isdir(cloud_folder):
        os.mkdir(cloud_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)
    
    if args.dataset=='s3dis':    
        files = [os.path.join(data_folder, o) for o in os.listdir(data_folder) 
                if os.path.isdir(os.path.join(data_folder,o))]
    elif args.dataset=='sema3d':
        files = glob.glob(data_folder+"*.txt")
    elif args.dataset=='custom_dataset':
        #list all ply files in the folder
        files = glob.glob(data_folder+"*.ply")
        #list all las files in the folder
        files = glob.glob(data_folder+"*.las")
        
    if (len(files) == 0):
        raise ValueError('%s is empty' % data_folder)
        
    n_files = len(files)
    i_file = 0
    for file in files:
        file_name   = os.path.splitext(os.path.basename(file))[0]
        
        if args.dataset=='s3dis':
            data_file   = data_folder      + file_name + '/' + file_name + ".txt"
            cloud_file  = cloud_folder     + file_name
            fea_file    = fea_folder       + file_name + '.h5'
            spg_file    = spg_folder       + file_name + '.h5'
        elif args.dataset=='sema3d':
            file_name_short = '_'.join(file_name.split('_')[:2])
            data_file  = data_folder + file_name + ".txt"
            label_file = data_folder + file_name_short + ".labels"
            cloud_file = cloud_folder+ file_name_short
            fea_file   = fea_folder  + file_name_short + '.h5'
            spg_file   = spg_folder  + file_name_short + '.h5'
        elif args.dataset=='custom_dataset':
            #adapt to your hierarchy. The following 4 files must be defined
            data_file   = data_folder      + file_name + '.ply' #or .las
            cloud_file  = cloud_folder     + file_name
            fea_file    = fea_folder       + file_name + '.h5'
            spg_file    = spg_folder       + file_name + '.h5'
        
        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> "+file_name)
        #--- build the geometric feature file h5 file ---
        if os.path.isfile(fea_file) and not args.overwrite:
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
        else :
            print("    creating the feature file...")
            #--- read the data files and compute the labels---
            if args.dataset=='s3dis':
                xyz, rgb, labels, objects = read_s3dis_format(data_file)
                if args.voxel_width > 0:
                    xyz, rgb, labels, dump = libply_c.prune(xyz.astype('f4'), args.voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
            elif args.dataset=='sema3d':
                label_file = data_folder + file_name + ".labels"
                has_labels = (os.path.isfile(label_file))
                if (has_labels):
                    xyz, rgb, labels = read_semantic3d_format(data_file, n_labels, label_file, args.voxel_width, args.ver_batch)
                else:
                    xyz, rgb = read_semantic3d_format(data_file, 0, '', args.voxel_width, args.ver_batch)
                    labels = []
            elif args.dataset=='custom_dataset':
                #implement in provider.py your own read_custom_format outputing xyz, rgb, labels
                #example for ply files
                xyz, rgb, labels = read_ply(data_file)
                #another one for las files without rgb
                xyz = read_las(data_file)
                if args.voxel_width > 0:
                    #an example of pruning without labels
                    xyz, rgb, labels = libply_c.prune(xyz, args.voxel_width, rgb, np.array(1,dtype='u1'), 0)
                    #another one without rgb information nor labels
                    xyz = libply_c.prune(xyz, args.voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
                #if no labels available simply set here labels = []
                #if no rgb available simply set here rgb = [] and make sure to not use it later on
            start = timer()
            #---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
        #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not args.overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            start = timer()
            if args.dataset=='s3dis':
                features = np.hstack((geof, rgb/255.)).astype('float32')#add rgb as a feature for partitioning
                features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
            elif args.dataset=='sema3d':
                 features = geof
                 geof[:,3] = 2. * geof[:, 3]
            elif args.dataset=='custom_dataset':
                #choose here which features to use for the partition
                 features = geof
                 geof[:,3] = 2. * geof[:, 3]
                
            graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype = 'object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)
        
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
