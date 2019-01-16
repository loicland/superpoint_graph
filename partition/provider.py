"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky

functions for writing and reading features and superpoint graph

"""
import os
import sys
import random
import glob
from plyfile import PlyData, PlyElement
import numpy as np
from numpy import genfromtxt
import h5py
#import laspy
from sklearn.neighbors import NearestNeighbors
sys.path.append("./ply_c")
sys.path.append("./partition/ply_c")
import libply_c
import colorsys
#------------------------------------------------------------------------------
def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
        color[components[i_com], :] = [random_color(), random_color()
        , random_color()]
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
    , ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def geof2ply(filename, xyz, geof):
    """write a ply with colors corresponding to geometric features"""
    color = np.array(255 * geof[:, [0, 1, 3]], dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def prediction2ply(filename, xyz, prediction, n_label, dataset):
    """write a ply with colors for each class"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
        color[np.where(prediction == i_label), :] = get_color_from_label(i_label, dataset)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def error2ply(filename, xyz, rgb, labels, prediction):
    """write a ply with green hue for correct classifcation and red for error"""
    if len(prediction.shape) > 1 and prediction.shape[1] > 1:
        prediction = np.argmax(prediction, axis = 1)
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    color_rgb = rgb/255;
    for i_ver in range(0, len(labels)):
        
        color_hsv = list(colorsys.rgb_to_hsv(color_rgb[i_ver,0], color_rgb[i_ver,1], color_rgb[i_ver,2]))
        if (labels[i_ver] == prediction[i_ver]) or (labels[i_ver]==0):
            color_hsv[0] = 0.333333
        else:
            color_hsv[0] = 0
        color_hsv[1] = min(1, color_hsv[1] + 0.3);
        color_hsv[2] = min(1, color_hsv[2] + 0.1);
        color_rgb[i_ver,:] = list(colorsys.hsv_to_rgb(color_hsv[0], color_hsv[1], color_hsv[2]))
    color_rgb = np.array(color_rgb*255, dtype='u1');
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color_rgb[:, i]        
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def spg2ply(filename, spg_graph):
    """write a ply displaying the SPG by adding edges between its centroid"""
    vertex_prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_val = np.empty((spg_graph['sp_centroids']).shape[0], dtype=vertex_prop)
    for i in range(0, 3):
        vertex_val[vertex_prop[i][0]] = spg_graph['sp_centroids'][:, i]
    edges_prop = [('vertex1', 'int32'), ('vertex2', 'int32')]
    edges_val = np.empty((spg_graph['source']).shape[0], dtype=edges_prop)
    edges_val[edges_prop[0][0]] = spg_graph['source'].flatten()
    edges_val[edges_prop[1][0]] = spg_graph['target'].flatten()
    ply = PlyData([PlyElement.describe(vertex_val, 'vertex'), PlyElement.describe(edges_val, 'edge')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def get_color_from_label(object_label, dataset):
    """associate the color corresponding to the class"""
    if dataset == 's3dis': #S3DIS
        object_label = {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 233, 229, 107], #'ceiling' .-> .yellow
            2: [  95, 156, 196], #'floor' .-> . blue
            3: [ 179, 116,  81], #'wall'  ->  brown
            4: [  81, 163, 148], #'column'  ->  bluegreen
            5: [ 241, 149, 131], #'beam'  ->  salmon
            6: [  77, 174,  84], #'window'  ->  bright green
            7: [ 108, 135,  75], #'door'   ->  dark green
            8: [  79,  79,  76], #'table'  ->  dark grey
            9: [  41,  49, 101], #'chair'  ->  darkblue
            10: [223,  52,  52], #'bookcase'  ->  red
            11: [ 89,  47,  95], #'sofa'  ->  purple
            12: [ 81, 109, 114], #'board'   ->  grey
            13: [233, 233, 229], #'clutter'  ->  light grey
            }.get(object_label, -1)
    elif (dataset == 'helix'): #Semantic3D
        object_label = {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 233, 229, 107], #'ceiling' .-> .yellow
            2: [  95, 156, 196], #'floor' .-> . blue
            3: [ 179, 116,  81], #'wall'  ->  brown
            4: [  81, 163, 148], #'column'  ->  bluegreen
            5: [ 241, 149, 131], #'beam'  ->  salmon
            6: [  77, 174,  84], #'window'  ->  bright green
            7: [ 108, 135,  75], #'door'   ->  dark green
            8: [  79,  79,  76], #'table'  ->  dark grey
            9: [  41,  49, 101], #'chair'  ->  darkblue
            10: [223,  52,  52], #'bookcase'  ->  red
            11: [ 89,  47,  95], #'sofa'  ->  purple
            12: [ 81, 109, 114], #'board'   ->  grey
            13: [233, 233, 229], #'clutter'  ->  light grey
            }.get(object_label, -1)
    elif (dataset == 'custom_s3dis'): #Semantic3D
        object_label = {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 233, 229, 107], #'ceiling' .-> .yellow
            2: [  95, 156, 196], #'floor' .-> . blue
            3: [ 179, 116,  81], #'wall'  ->  brown
            4: [  81, 163, 148], #'column'  ->  bluegreen
            5: [ 241, 149, 131], #'beam'  ->  salmon
            6: [  77, 174,  84], #'window'  ->  bright green
            7: [ 108, 135,  75], #'door'   ->  dark green
            8: [  79,  79,  76], #'table'  ->  dark grey
            9: [  41,  49, 101], #'chair'  ->  darkblue
            10: [223,  52,  52], #'bookcase'  ->  red
            11: [ 89,  47,  95], #'sofa'  ->  dark purple
            12: [ 81, 109, 114], #'board'   ->  grey
            13: [233, 233, 229], #'clutter'  ->  light grey
            14: [100, 200, 150], #'stair'  ->  light purple
            }.get(object_label, -1)
    elif (dataset == 'sema3d'): #Semantic3D
        object_label =  {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 200, 200, 200], #'man-made terrain'  ->  grey
            2: [   0,  70,   0], #'natural terrain'  ->  dark green
            3: [   0, 255,   0], #'high vegetation'  ->  bright green
            4: [ 255, 255,   0], #'low vegetation'  ->  yellow
            5: [ 255,   0,   0], #'building'  ->  red
            6: [ 148,   0, 211], #'hard scape'  ->  violet
            7: [   0, 255, 255], #'artifact'   ->  cyan
            8: [ 255,   8, 127], #'cars'  ->  pink
            }.get(object_label, -1)
    elif (dataset == 'custom_dataset'): #Custom set
        object_label =  {
            0: [0   ,   0,   0], #unlabelled .->. black
            1: [ 255, 0, 0], #'classe A' -> red
            2: [ 0, 255, 0], #'classeB' -> green
            }.get(object_label, -1)
    else: 
        raise ValueError('Unknown dataset: %s' % (dataset))
    if object_label == -1:
        raise ValueError('Type not recognized: %s' % (object_label))
    return object_label
#------------------------------------------------------------------------------
def read_s3dis_format(raw_path, label_out=True):
#S3DIS specific
    """extract data from a room folder"""
    room_ver = genfromtxt(raw_path, delimiter=' ')
    xyz = np.array(room_ver[:, 0:3], dtype='float32')
    rgb = np.array(room_ver[:, 3:6], dtype='uint8')
    if not label_out:
        return xyz, rgb
    n_ver = len(room_ver)
    del room_ver
    nn = NearestNeighbors(1, algorithm='kd_tree').fit(xyz)
    room_labels = np.zeros((n_ver,), dtype='uint8')
    room_object_indices = np.zeros((n_ver,), dtype='uint32')
    objects = glob.glob(os.path.dirname(raw_path) + "/Annotations/*.txt")
    i_object = 0
    for single_object in objects:
        object_name = os.path.splitext(os.path.basename(single_object))[0]
        print("        adding object " + str(i_object) + " : "  + object_name)
        object_class = object_name.split('_')[0]
        object_label = object_name_to_label(object_class)
        obj_ver = genfromtxt(single_object, delimiter=' ')
        distances, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
        room_labels[obj_ind] = object_label
        room_object_indices[obj_ind] = i_object
        i_object = i_object + 1
    return xyz, rgb, room_labels
#------------------------------------------------------------------------------
def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = {
        'ceiling': 1,
        'floor': 2,
        'wall': 3,
        'column': 4,
        'beam': 5,
        'window': 6,
        'door': 7,
        'table': 8,
        'chair': 9,
        'bookcase': 10,
        'sofa': 11,
        'board': 12,
        'clutter': 13,
        'stairs': 0,
        }.get(object_class, 0)
    return object_label
#------------------------------------------------------------------------------
def read_semantic3d_format(data_file, n_class, file_label_path, voxel_width, ver_batch):
    """read the format of semantic3d. 
    ver_batch : if ver_batch>0 then load the file ver_batch lines at a time.
                useful for huge files (> 5millions lines)
    voxel_width: if voxel_width>0, voxelize data with a regular grid
    n_class : the number of class; if 0 won't search for labels (test set)
    implements batch-loading for huge files
    and pruning"""
    i_rows = 0
    xyz = np.zeros((0, 3), dtype='float32')
    rgb = np.zeros((0, 3), dtype='uint8')
    labels = np.zeros((0, n_class+1), dtype='uint32')
    #---the clouds can potentially be too big to parse directly---
    #---they are cut in batches in the order they are stored---
    while True:
        try:
            if ver_batch>0:
                vertices = np.genfromtxt(data_file
                         , delimiter=' ', max_rows=ver_batch
                         , skip_header=i_rows)
            else:
                vertices = np.genfromtxt(data_file, delimiter=' ')
                break
                
        except StopIteration:
            #end of file
            break
        if len(vertices)==0:
            break
        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        rgb_full = np.array(vertices[:, 4:7], dtype='uint8')
        del vertices
        if n_class > 0:
            labels_full = np.genfromtxt(file_label_path, dtype="u1", delimiter=' '
                        , max_rows=ver_batch, skip_header=i_rows)
                
        if voxel_width > 0:
            if n_class > 0:
                xyz_sub, rgb_sub, labels_sub = libply_c.prune(xyz_full, voxel_width
                                             , rgb_full, labels_full, n_class)
                labels = np.vstack((labels, labels_sub))
            else:
                xyz_sub, rgb_sub, l = libply_c.prune(xyz_full, voxel_width
                                    , rgb_full, np.zeros(1, dtype='uint8'), 0)
            del xyz_full, rgb_full
            xyz = np.vstack((xyz, xyz_sub))
            rgb = np.vstack((rgb, rgb_sub))
        i_rows = i_rows + ver_batch        
    if n_class>0:
        return xyz, rgb, labels
    else:
        return xyz, rgb
#------------------------------------------------------------------------------
def read_ply(filename):
    """convert from a ply file. include the label and the object number"""
    #---read the ply file--------
    plydata = PlyData.read(filename)
    xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
    try:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.float32)
    if np.max(rgb) > 1:
        rgb = rgb
    try:
        object_indices = plydata['vertex']['object_index']
        labels = plydata['vertex']['label']
        return xyz, rgb, labels, object_indices
    except ValueError:
        try:
            labels = plydata['vertex']['label']
            return xyz, rgb, labels
        except ValueError:
            return xyz, rgb
#------------------------------------------------------------------------------
def read_las(filename):
    """convert from a las file with no rgb"""
    #---read the ply file--------
    try:
        inFile = laspy.file.File(filename, mode='r')
    except NameError:
        raise ValueError("laspy package not found. uncomment import in /partition/provider and make sure it is installed in your environment")
    N_points = len(inFile)
    x = np.reshape(inFile.X, (N_points,1))
    y = np.reshape(inFile.Y, (N_points,1))
    z = np.reshape(inFile.Z, (N_points,1))
    xyz = np.hstack((x,y,z)).astype('f4')
    return xyz
#------------------------------------------------------------------------------
def write_ply_obj(filename, xyz, rgb, labels, object_indices):
    """write into a ply file. include the label and the object number"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1')
            , ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')
            , ('object_index', 'uint32')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    vertex_all[prop[7][0]] = object_indices
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def write_ply_labels(filename, xyz, rgb, labels):
    """write into a ply file. include the label"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1')
            , ('blue', 'u1'), ('label', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    vertex_all[prop[6][0]] = labels
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def write_ply(filename, xyz, rgb):
    """write into a ply file"""
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        if (len(rgb)) >0 : 
            vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
        else:
            vertex_all[prop[i_prop+3][0]] = 0
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
def write_features(file_name, geof, xyz, rgb, graph_nn, labels):
    """write the geometric features, labels and clouds in a h5 file"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('linearity', data=geof[:, 0], dtype='float32')
    data_file.create_dataset('planarity', data=geof[:, 1], dtype='float32')
    data_file.create_dataset('scattering', data=geof[:, 2], dtype='float32')
    data_file.create_dataset('verticality', data=geof[:, 3], dtype='float32')
    data_file.create_dataset('source', data=graph_nn["source"], dtype='uint32')
    data_file.create_dataset('target', data=graph_nn["target"], dtype='uint32')
    data_file.create_dataset('distances', data=graph_nn["distances"], dtype='float32')
    data_file.create_dataset('xyz', data=xyz, dtype='float32')
    if len(rgb) > 0:
        data_file.create_dataset('rgb', data=rgb, dtype='uint8')
    if len(labels) > 0 and len(labels.shape)>1 and labels.shape[1]>1:
        data_file.create_dataset('labels', data=labels, dtype='uint32')
    else:
        data_file.create_dataset('labels', data=labels, dtype='uint8')
    data_file.close()
#------------------------------------------------------------------------------
def read_features(file_name):
    """read the geometric features, clouds and labels from a h5 file"""
    data_file = h5py.File(file_name, 'r')
    #fist get the number of vertices
    n_ver = len(data_file["linearity"])
    has_labels = len(data_file["labels"])
    #the labels can be empty in the case of a test set
    if has_labels:
        labels = np.array(data_file["labels"])
    else:
        labels = []
    #---create the arrays---
    geof = np.zeros((n_ver, 4), dtype='float32')
    #---fill the arrays---
    geof[:, 0] = data_file["linearity"]
    geof[:, 1] = data_file["planarity"]
    geof[:, 2] = data_file["scattering"]
    geof[:, 3] = data_file["verticality"]
    xyz = data_file["xyz"][:]
    try:
        rgb = data_file["rgb"][:]
    except KeyError as e:
        rgb = []
    source = data_file["source"][:]
    target = data_file["target"][:]
    distances = data_file["distances"][:]
    #---set the graph---
    graph_nn = dict([("is_nn", True)])
    graph_nn["source"] = source
    graph_nn["target"] = target
    graph_nn["distances"] = distances
    return geof, xyz, rgb, graph_nn, labels
#------------------------------------------------------------------------------
def write_spg(file_name, graph_sp, components, in_component):
    """save the partition and spg information"""
    if os.path.isfile(file_name):
        os.remove(file_name)
    data_file = h5py.File(file_name, 'w')
    grp = data_file.create_group('components')
    n_com = len(components)
    for i_com in range(0, n_com):
        grp.create_dataset(str(i_com), data=components[i_com], dtype='uint32')
    data_file.create_dataset('in_component'
                             , data=in_component, dtype='uint32')
    data_file.create_dataset('sp_labels'
                             , data=graph_sp["sp_labels"], dtype='uint32')
    data_file.create_dataset('sp_centroids'
                             , data=graph_sp["sp_centroids"], dtype='float32')
    data_file.create_dataset('sp_length'
                             , data=graph_sp["sp_length"], dtype='float32')
    data_file.create_dataset('sp_surface'
                             , data=graph_sp["sp_surface"], dtype='float32')
    data_file.create_dataset('sp_volume'
                             , data=graph_sp["sp_volume"], dtype='float32')
    data_file.create_dataset('sp_point_count'
                             , data=graph_sp["sp_point_count"], dtype='uint64')
    data_file.create_dataset('source'
                             , data=graph_sp["source"], dtype='uint32')
    data_file.create_dataset('target'
                             , data=graph_sp["target"], dtype='uint32')
    data_file.create_dataset('se_delta_mean'
                             , data=graph_sp["se_delta_mean"], dtype='float32')
    data_file.create_dataset('se_delta_std'
                             , data=graph_sp["se_delta_std"], dtype='float32')
    data_file.create_dataset('se_delta_norm'
                             , data=graph_sp["se_delta_norm"], dtype='float32')
    data_file.create_dataset('se_delta_centroid'
                             , data=graph_sp["se_delta_centroid"], dtype='float32')
    data_file.create_dataset('se_length_ratio'
                             , data=graph_sp["se_length_ratio"], dtype='float32')
    data_file.create_dataset('se_surface_ratio'
                             , data=graph_sp["se_surface_ratio"], dtype='float32')
    data_file.create_dataset('se_volume_ratio'
                             , data=graph_sp["se_volume_ratio"], dtype='float32')
    data_file.create_dataset('se_point_count_ratio'
                             , data=graph_sp["se_point_count_ratio"], dtype='float32')
#-----------------------------------------------------------------------------
def read_spg(file_name):
    """read the partition and spg information"""
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype='uint32')
    graph["target"] = np.array(data_file["target"], dtype='uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype='float32')
    graph["sp_length"] = np.array(data_file["sp_length"], dtype='float32')
    graph["sp_surface"] = np.array(data_file["sp_surface"], dtype='float32')
    graph["sp_volume"] = np.array(data_file["sp_volume"], dtype='float32')
    graph["sp_point_count"] = np.array(data_file["sp_point_count"], dtype='uint64')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype='float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype='float32')
    graph["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype='float32')
    graph["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype='float32')
    graph["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype='float32')
    graph["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype='float32')
    graph["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype='float32')
    graph["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype='float32')
    in_component = np.array(data_file["in_component"], dtype='uint32')
    n_com = len(graph["sp_length"])
    graph["sp_labels"] = np.array(data_file["sp_labels"], dtype='uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype=object)
    for i_com in range(0, n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype='uint32').tolist()
    return graph, components, in_component
#------------------------------------------------------------------------------
def reduced_labels2full(labels_red, components, n_ver):
    """distribute the labels of superpoints to their repsective points"""
    labels_full = np.zeros((n_ver, ), dtype='uint8')
    for i_com in range(0, len(components)):
        labels_full[components[i_com]] = labels_red[i_com]
    return labels_full
#------------------------------------------------------------------------------
def interpolate_labels_batch(data_file, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    i_rows = 0
    labels_f = np.zeros((0, ), dtype='uint8')
    #---the clouds can potentially be too big to parse directly---
    #---they are cut in batches in the order they are stored---
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    while True:
        try:
            if ver_batch>0:
                print("read lines %d to %d" % (i_rows, i_rows + ver_batch))
                vertices = np.genfromtxt(data_file
                         , delimiter=' ', max_rows=ver_batch
                         , skip_header=i_rows)
            else:
                vertices = np.genfromtxt(data_file
                         , delimiter=' ')
                break
        except StopIteration:
            #end of file
            break
        if len(vertices)==0:
            break
        xyz_full = np.array(vertices[:, 0:3], dtype='float32')
        del vertices
        distances, neighbor = nn.kneighbors(xyz_full)
        del distances
        labels_f = np.hstack((labels_f, labels[neighbor].flatten()))
        i_rows = i_rows + ver_batch
    return labels_f
#------------------------------------------------------------------------------
def interpolate_labels(xyz_up, xyz, labels, ver_batch):
    """interpolate the labels of the pruned cloud to the full cloud"""
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis = 1)
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(xyz)
    distances, neighbor = nn.kneighbors(xyz_up)
    return labels[neighbor].flatten()
