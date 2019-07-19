#------------------------------------------------------------------------------
#---------  Graph methods for SuperPoint Graph   ------------------------------
#---------     Loic Landrieu, Dec. 2017     -----------------------------------
#------------------------------------------------------------------------------
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
import numpy.matlib
#------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    #save the graph
    graph["source"] = source.flatten().astype('uint32')
    graph["target"] = neighbors.flatten().astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph
#------------------------------------------------------------------------------
def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
    """compute simulteneoulsy 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]
        
        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))
        
        edg_id = graph["source"] + n_ver * graph["target"]
        
        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]
       
        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    return graph, target2
#------------------------------------------------------------------------------
def compute_sp_graph(xyz, d_max, in_component, components, labels, n_labels):
    """compute the superpoint graph with superpoints and superedges features"""
    n_com = max(in_component)+1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    #---compute delaunay triangulation---
    tri = Delaunay(xyz)
    #interface select the edges between different components
    #edgx and edgxr converts from tetrahedrons to edges
	#done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r ,edg5r, edg6r))
    del edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)
    
    if d_max > 0:
        dist = np.sqrt(((xyz[edges[0,:]]-xyz[edges[1,:]])**2).sum(1))
        edges = edges[:,dist<d_max]
	
    #---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0,:] +  edge_comp[1,:]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    #marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    #---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='uint64')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["target"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["se_delta_mean"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_std"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_delta_norm"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_delta_centroid"] = np.zeros((n_sedg, 3), dtype='float32')
    graph["se_length_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_surface_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_volume_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    graph["se_point_count_ratio"] = np.zeros((n_sedg, 1), dtype='float32')
    if has_labels:
        graph["sp_labels"] = np.zeros((n_com, n_labels + 1), dtype='uint32')
    else:
        graph["sp_labels"] = []
    #---compute the superpoint features---
    for i_com in range(0, n_com):
        comp = components[i_com]
        if has_labels and not label_hist:
            graph["sp_labels"][i_com, :] = np.histogram(labels[comp]
                , bins=[float(i)-0.5 for i in range(0, n_labels + 2)])[0]
        if has_labels and label_hist:
            graph["sp_labels"][i_com, :] = sum(labels[comp,:])
        graph["sp_point_count"][i_com] = len(comp)
        xyz_sp = np.unique(xyz[comp, :], axis=0)
        if len(xyz_sp) == 1:
            graph["sp_centroids"][i_com] = xyz_sp
            graph["sp_length"][i_com] = 0
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        elif len(xyz_sp) == 2:
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            graph["sp_length"][i_com] = np.sqrt(np.sum(np.var(xyz_sp, axis=0)))
            graph["sp_surface"][i_com] = 0
            graph["sp_volume"][i_com] = 0
        else:
            ev = LA.eig(np.cov(np.transpose(xyz_sp), rowvar=True))
            ev = -np.sort(-ev[0]) #descending order
            graph["sp_centroids"][i_com] = np.mean(xyz_sp, axis=0)
            try:
                graph["sp_length"][i_com] = ev[0]
            except TypeError:
                graph["sp_length"][i_com] = 0
            try:
                graph["sp_surface"][i_com] = np.sqrt(ev[0] * ev[1] + 1e-10)
            except TypeError:
                graph["sp_surface"][i_com] = 0
            try:
                graph["sp_volume"][i_com] = np.sqrt(ev[0] * ev[1] * ev[2] + 1e-10)
            except TypeError:
                graph["sp_volume"][i_com] = 0
    #---compute the superedges features---
    for i_sedg in range(0, n_sedg):
        i_edg_begin = jump_edg[i_sedg]
        i_edg_end = jump_edg[i_sedg + 1]
        ver_source = edges[0, range(i_edg_begin, i_edg_end)]
        ver_target = edges[1, range(i_edg_begin, i_edg_end)]
        com_source = edge_comp[0, i_edg_begin]
        com_target = edge_comp[1, i_edg_begin]
        xyz_source = xyz[ver_source, :]
        xyz_target = xyz[ver_target, :]
        graph["source"][i_sedg] = com_source
        graph["target"][i_sedg] = com_target
        #---compute the ratio features---
        graph["se_delta_centroid"][i_sedg,:] = graph["sp_centroids"][com_source,:] - graph["sp_centroids"][com_target, :]
        graph["se_length_ratio"][i_sedg] = graph["sp_length"][com_source] / (graph["sp_length"][com_target] + 1e-6)
        graph["se_surface_ratio"][i_sedg] = graph["sp_surface"][com_source] / (graph["sp_surface"][com_target] + 1e-6)
        graph["se_volume_ratio"][i_sedg] = graph["sp_volume"][com_source] / (graph["sp_volume"][com_target] + 1e-6)
        graph["se_point_count_ratio"][i_sedg] = graph["sp_point_count"][com_source] / (graph["sp_point_count"][com_target] + 1e-6)
        #---compute the offset set---
        delta = xyz_source - xyz_target
        if len(delta) > 1:
            graph["se_delta_mean"][i_sedg] = np.mean(delta, axis=0)
            graph["se_delta_std"][i_sedg] = np.std(delta, axis=0)
            graph["se_delta_norm"][i_sedg] = np.mean(np.sqrt(np.sum(delta ** 2, axis=1)))
        else:
            graph["se_delta_mean"][i_sedg, :] = delta
            graph["se_delta_std"][i_sedg, :] = [0, 0, 0]
            graph["se_delta_norm"][i_sedg] = np.sqrt(np.sum(delta ** 2))
    return graph
