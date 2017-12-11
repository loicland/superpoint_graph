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
#---parameters-----------------------------------------------------------------
k_nn_geof = 45 #number of neighbors for the geometric features
k_nn_adj = 10 #adjacency structure for the minimal partition
lambda_edge_weight = 1. #parameter determine the edge weight for minimal part.
reg_strength = .2 #regularization strength for the minimal partition
d_se_max = 10 #max length of super edges
#---path to data---------------------------------------------------------------
#root of the data directory
root = "/media/landrieuloic/Data/Stanford3D/"
#list of subfolders to be processed
areas = ["Area_1/","Area_2/","Area_3/","Area_4/","Area_5/","Area_6/"]
areas = ["Area_1/"]
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
for i_area in range(0,num_area):
    area = areas[i_area]
    print("=================\n   "+area+"\n=================")
    raw_folder = root + "raw/"               +area
    ply_folder = root + "clouds/"            +area
    fea_folder = root + "features/"          +area
    spg_folder = root + "/superpoint_graphs/"+area
    if not os.path.isdir(raw_folder):
        raise ValueError("%s do not exists" % raw_folder)
    if not os.path.isdir(ply_folder):
        os.mkdir(ply_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)
    rooms = [os.path.join(raw_folder, o) for o in os.listdir(raw_folder) 
                    if os.path.isdir(os.path.join(raw_folder,o))]
    if (len(rooms) == 0):
        raise ValueError('%s is empty' % raw_folder)
    n_rooms = len(rooms)
    i_room = 0
    #rooms = [rooms[9]]
    for room in rooms:
        room_name = os.path.splitext(os.path.basename(room))[0]
        room_raw_folder = raw_folder      + room_name + '/'
        room_raw_path   = room_raw_folder + room_name + ".txt"
        room_ply_path   = ply_folder      + room_name + '.ply' 
        room_fea_path   = fea_folder      + room_name + '.h5'
        room_spg_path   = spg_folder      + room_name + '.h5' 
        i_room = i_room + 1
        print(str(i_room) + " / " + str(n_rooms) + "---> "+room_name)
        #--- build the labeled ply file ---
        if os.path.isfile(room_ply_path):
            print("    reading the existing ply file...")
            xyz, rgb, room_labels, room_object_indices = read_ply(room_ply_path)
        else :
            print("    creating the ply file...")
            xyz, rgb, room_labels, room_object_indices = get_objects(room_raw_path)
            write_ply_obj(room_ply_path, xyz, rgb, room_labels, room_object_indices)
        #--- build the geometric feature file h5 file ---
        if os.path.isfile(room_fea_path):
            print("    reading the existing feature file...")
            geof, graph_nn = read_geof(room_fea_path)
        else :
            print("    creating the feature file...")
            start = timer()
            #---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_geof(room_fea_path, geof, xyz, rgb, graph_nn)
        #--compute the partition------
        if os.path.isfile(room_spg_path):
            print("    reading the existing superpoint graph file...")
            read_spg(room_spg_path)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            start = timer()
            features = np.hstack((geof, rgb/255.))
            features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
            graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            partition = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], reg_strength)
            components = np.array(partition[0], dtype = 'object')
            in_component = np.array(partition[1], dtype = 'uint32')
            end = timer()
            times[1] = times[1] + end - start
            del partition
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, room_labels, 13)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(room_spg_path, graph_sp, components, in_component)
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
