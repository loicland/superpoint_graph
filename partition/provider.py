#------------------------------------------------------------------------------
#------ PLY finctions for writing and converting ------------------------------
#----------- Loic Landrieu Dec. 2017 ------------------------------------------
#------------------------------------------------------------------------------
import random
import glob
from plyfile import PlyData, PlyElement
import numpy as np
import h5py
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
#------------------------------------------------------------------------------
#---------- partition2ply ---------------------------------------------------
#----- write a ply with random colors for each components ---------------------
#------------------------------------------------------------------------------
def partition2ply(filename, xyz, components):
    r = lambda: random.randint(0,255)
    color = np.zeros(xyz.shape)
    for i_com in range(0,len(components)):
        color[components[i_com],:] = [r(), r(), r()]
    prop=[('x', 'f4'),('y', 'f4'), ('z', 'f4'),('red', 'u1'),('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz),dtype=prop)
    for i in range(0,3):
        vertex_all[prop[i][0]] = xyz[:,i]
    for i in range(0,3):
        vertex_all[prop[i+3][0]] = color[:,i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
#---------- geof2ply ----------------------------------------------------------
#----- write a ply with colors corresponding to geometric features ------------
#------------------------------------------------------------------------------
def geopf2ply(filename, xyz, geof):
    color = np.array(255 * geof[:, [0, 1, 3]], dtype = 'uint8')
    prop=[('x', 'f4'),('y', 'f4'), ('z', 'f4'),('red', 'u1'),('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz),dtype=prop)
    for i in range(0,3):
        vertex_all[prop[i][0]] = xyz[:,i]
    for i in range(0,3):
        vertex_all[prop[i+3][0]] = color[:,i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
#---------- prediction2ply --------------------------------------------------
#----- write a ply with colors for each class------------ ---------------------
#------------------------------------------------------------------------------
def prediction2ply(filename, xyz, prediction, n_label):
    color = np.zeros(xyz.shape)
    for i_label in range(0, n_label + 1):
	    color[np.where(prediction==i_label),:] = get_color_from_label(i_label)
    prop=[('x', 'f4'),('y', 'f4'), ('z', 'f4'),('red', 'u1'),('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz),dtype=prop)
    for i in range(0,3):
        vertex_all[prop[i][0]] = xyz[:,i]
    for i in range(0,3):
        vertex_all[prop[i+3][0]] = color[:,i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
#---------- object_name_to_label ----------------------------------------------
#----- convert from object name in S3DIS to an int-----------------------------
#------------------------------------------------------------------------------
def object_name_to_label(object_class):
    object_label =  {
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
    if (object_label==0):
         raise ValueError('Type not recognized: %s' % (object_class))
    return object_label
#------------------------------------------------------------------------------
#----------  get_color_from_label----------------------------------------------
#----- associate the color corresponding to the class--------------------------
#------------------------------------------------------------------------------
def get_color_from_label(object_label, n_label):
	if (n_label == 13):
		object_label =  {
		        0: [0   ,   0,    0],
		        1: [ 233, 229, 107],
		        2: [  95, 156, 196],
		        3: [ 179, 116,  81],
		        4: [  81, 163, 148],
		        5: [ 241, 149, 131],
		        6: [  77, 174,  84],
		        7: [ 108, 135,  75],
		        8: [  79,  79,  76],
		        9: [  41,  49, 101],
		        10: [223,  52,  52],
		        11: [ 89,  47,  95],
		        12: [ 81, 109, 114],
		        13: [233, 233, 229],
		    }.get(object_label, -1)
	if (object_label==-1):
		raise ValueError('Type not recognized: %s' % (object_label))
	return object_label
#------------------------------------------------------------------------------
#---------- get_objects -------- ----------------------------------------------
#----- convert the room folder to a single ply file ---------------------------
#------------------------------------------------------------------------------
def get_objects(raw_path):
    room_ver = genfromtxt(raw_path, delimiter=' ')
    xyz = np.array(room_ver[:,0:3], dtype = 'float32')
    rgb = np.array(room_ver[:,3:6], dtype = 'float32')
    n_ver = len(room_ver)
    nn = NearestNeighbors(1, algorithm='kd_tree').fit(xyz)
    room_labels = np.zeros((n_ver,), dtype = 'uint8')
    room_object_indices = np.zeros((n_ver,), dtype = 'uint32')
    objects = glob.glob(raw_path + "Annotations/*.txt")
    i_object = 0
    for single_object in objects:
        object_name = os.path.splitext(os.path.basename(single_object))[0]
        print("        adding object " + str(i_object) + " : "  + object_name)
        object_class = object_name.split('_')[0]
        object_label = object_name_to_label(object_class)
        obj_ver = genfromtxt(single_object , delimiter=' ')
        distances, obj_ind = nn.kneighbors(obj_ver[:,0:3])
        room_labels[obj_ind] = object_label
        room_object_indices[obj_ind] = i_object
        i_object = i_object + 1
    return xyz, rgb, room_labels, room_object_indices
#------------------------------------------------------------------------------
#---------- write_ply_obj -----------------------------------------------------
#----- convert to a ply file. include the label and the object number----------
#------------------------------------------------------------------------------
def write_ply_obj(filename, xyz, rgb, labels, object_indices):
    prop=[('x', 'f4'),('y', 'f4'), ('z', 'f4'),('red', 'u1'),('green', 'u1'), ('blue', 'u1'), ('label', 'u1') , ('object_index', 'uint32')]
    vertex_all = np.empty(len(xyz),dtype=prop)
    for i_prop in range(0,3):
    	vertex_all[prop[i_prop][0]] = xyz[:,i_prop]
    for i_prop in range(0,3):
        vertex_all[prop[i_prop+3][0]] = rgb[:,i_prop]
    vertex_all[prop[6][0]] = labels
    vertex_all[prop[7][0]] = object_indices
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)
#------------------------------------------------------------------------------
#---------- write_ply -------- ------------------------------------------------
#----- convert to a ply file. include the label -------------------------------
#------------------------------------------------------------------------------
def write_ply(filename, xyz, rgb, labels):
	prop=[('x', 'f4'),('y', 'f4'), ('z', 'f4'),('red', 'u1'),('green', 'u1'), ('blue', 'u1'), ('label', 'u1')]
	vertex_all = np.empty(len(xyz),dtype=prop)
	for i_prop in range(0,3):
		vertex_all[prop[i_prop][0]] = xyz[:,i_prop]
	for i_prop in range(0,3):
		vertex_all[prop[i_prop+3][0]] = rgb[:,i_prop]
	vertex_all[prop[6][0]] = labels  
	ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
	ply.write(filename)
#------------------------------------------------------------------------------
#---------- read_ply -------- ------------------------------------------------
#----- convert from a ply file. include the label and the object number -------
#------------------------------------------------------------------------------
def read_ply(filename):
	#---read the ply file--------
	plydata = PlyData.read(filename)  
	xyz = np.stack([ plydata['vertex'][n] for n in ['x','y','z'] ], axis=1)
	try:
		rgb = np.stack([ plydata['vertex'][n] for n in ['red', 'green', 'blue'] ], axis=1).astype(np.float32)
	except ValueError:
		rgb = np.stack([ plydata['vertex'][n] for n in ['r', 'g', 'b'] ], axis=1).astype(np.float32)
	if (np.max(rgb)>1):
		rgb = rgb / 255.
		labels = plydata['vertex']['label']
	try:
		object_indices = plydata['vertex']['object_index']
		return xyz, rgb, labels, object_indices
	except ValueError:
		return xyz, rgb, labels
#------------------------------------------------------------------------------
#---------- write_room_geof --------------------------------------------
#----- write the geometric features in a h5 file ------------------------------
#------------------------------------------------------------------------------
def write_geof(file_name, geof, xyz, rgb, graph_nn):
    data_file = h5py.File(file_name, 'w')
    data_file.create_dataset('linearity'  , data=geof[:,0], dtype = 'float32')
    data_file.create_dataset('planarity'  , data=geof[:,1], dtype = 'float32')
    data_file.create_dataset('scattering' , data=geof[:,2], dtype = 'float32')
    data_file.create_dataset('verticality', data=geof[:,3], dtype = 'float32')
    data_file.create_dataset('source'     , data=graph_nn["source"], dtype = 'uint32')
    data_file.create_dataset('target'     , data=graph_nn["target"], dtype = 'uint32')
    data_file.create_dataset('distances'  , data=graph_nn["distances"], dtype = 'float32')
    data_file.create_dataset('xyz'        , data=xyz, dtype = 'float32')
    data_file.create_dataset('rgb'        , data=rgb, dtype = 'float32')
    data_file.close()
#------------------------------------------------------------------------------
#---------- read_geof ---------------------------------------------------------
#----- read the geometric features from a h5 file -----------------------------
#------------------------------------------------------------------------------
def read_geof(file_name):
    data_file = h5py.File(file_name, 'r')
    linearity = data_file["linearity"]
    n_ver = len(linearity)
    geof = np.zeros((n_ver,4), dtype = 'float32')
    geof[:,0] = linearity
    geof[:,1] = data_file["planarity"]
    geof[:,2] = data_file["scattering"]
    geof[:,3] = data_file["verticality"]
    n_edg = len(data_file["source"])
    source = np.zeros((n_edg,), dtype = 'uint32')
    target = np.zeros((n_edg,), dtype = 'uint32')
    distances = np.zeros((n_edg,), dtype = 'float32')
    source[:] = data_file["source"]
    target[:] = data_file["target"]
    distances[:] = data_file["distances"]
    graph = dict([("is_nn", True)])
    graph["source"] = source
    graph["target"] = target	
    graph["distances"] = distances
    return geof, graph
#------------------------------------------------------------------------------
#---------- write_spg ---------------------------------------------------------
#----- save the partition and spg information ---------------------------------
#------------------------------------------------------------------------------
def write_spg(file_name, graph_sp, components, in_component):
    data_file = h5py.File(file_name, 'w')
    data_file
    grp = data_file.create_group('components')
    n_com = len(components)
    for i_com in range(0,n_com):
        grp.create_dataset(str(i_com),data=components[i_com], dtype = 'uint32')
    data_file.create_dataset('in_component',data=in_component, dtype = 'uint32')
    data_file.create_dataset('sp_labels',data=graph_sp["sp_labels"], dtype = 'uint32')
    data_file.create_dataset('sp_centroids',data=graph_sp["sp_centroids"], dtype = 'float32')
    data_file.create_dataset('sp_length',data=graph_sp["sp_length"], dtype = 'float32')
    data_file.create_dataset('sp_surface',data=graph_sp["sp_surface"], dtype = 'float32')
    data_file.create_dataset('sp_volume',data=graph_sp["sp_volume"], dtype = 'float32')
    data_file.create_dataset('sp_point_count',data=graph_sp["sp_point_count"], dtype = 'float32')
    data_file.create_dataset('source',data=graph_sp["source"], dtype = 'uint32')
    data_file.create_dataset('target',data=graph_sp["target"], dtype = 'uint32')
    data_file.create_dataset('se_delta_mean',data=graph_sp["se_delta_mean"], dtype = 'float32')
    data_file.create_dataset('se_delta_std',data=graph_sp["se_delta_std"], dtype = 'float32')
    data_file.create_dataset('se_delta_norm',data=graph_sp["se_delta_norm"], dtype = 'float32')
    data_file.create_dataset('se_delta_centroid',data=graph_sp["se_delta_centroid"], dtype = 'float32')
    data_file.create_dataset('se_length_ratio',data=graph_sp["se_length_ratio"], dtype = 'float32')
    data_file.create_dataset('se_surface_ratio',data=graph_sp["se_surface_ratio"], dtype = 'float32')
    data_file.create_dataset('se_volume_ratio',data=graph_sp["se_volume_ratio"], dtype = 'float32')
    data_file.create_dataset('se_point_count_ratio',data=graph_sp["se_point_count_ratio"], dtype = 'float32')
#-----------------------------------------------------------------------------
#---------- read_spg ---------------------------------------------------------
#----- retrieve the partition and spg information ----------------------------
#-----------------------------------------------------------------------------
def read_spg(file_name):
    data_file = h5py.File(file_name, 'r')
    graph = dict([("is_nn", False)])
    graph["source"] = np.array(data_file["source"], dtype = 'uint32')
    graph["target"] = np.array(data_file["target"], dtype = 'uint32')
    graph["sp_centroids"] = np.array(data_file["sp_centroids"], dtype = 'float32')
    graph["sp_length"] = np.array(data_file["sp_length"], dtype = 'float32')
    graph["sp_surface"] = np.array(data_file["sp_surface"], dtype = 'float32')
    graph["sp_volume"] = np.array(data_file["sp_volume"], dtype = 'float32')
    graph["sp_point_count"] = np.array(data_file["sp_point_count"], dtype = 'uint32')
    graph["se_delta_mean"] = np.array(data_file["se_delta_mean"], dtype = 'float32')
    graph["se_delta_std"] = np.array(data_file["se_delta_std"], dtype = 'float32')
    graph["se_delta_norm"] = np.array(data_file["se_delta_norm"], dtype = 'float32')
    graph["se_delta_centroid"] = np.array(data_file["se_delta_centroid"], dtype = 'float32')
    graph["se_length_ratio"] = np.array(data_file["se_length_ratio"], dtype = 'float32')
    graph["se_surface_ratio"] = np.array(data_file["se_surface_ratio"], dtype = 'float32')
    graph["se_volume_ratio"] = np.array(data_file["se_volume_ratio"], dtype = 'float32')
    graph["se_point_count_ratio"] = np.array(data_file["se_point_count_ratio"], dtype = 'float32')
    in_component = np.array(data_file["in_component"], dtype = 'uint32')
    n_com = len(graph["sp_length"])
    graph["sp_labels"]  = np.array(data_file["sp_labels"], dtype = 'uint32')
    grp = data_file['components']
    components = np.empty((n_com,), dtype = object)
    for i_com in range(0,n_com):
        components[i_com] = np.array(grp[str(i_com)], dtype = 'uint32').tolist()
    return graph, components, in_component
 
