
# coding: utf-8

# # Point Cloud Segmentation

# This notebook aims at coding a single step to process point cloud : It details each processing step from a raw point cloud to a semantic segmented point cloud. The method will take the point cloud in input and will output the corresponding segmented point cloud

# # **Point Cloud Segmentation Steps**

# ## ***1.Partitioning Point Cloud***

# This subsubsection adapt the code used in partition/parition.py.

# In[ ]:


import os.path
import sys
import numpy as np
from timeit import default_timer as timer
sys.path.append('partition/cut-pursuit/src')
sys.path.append('partition/ply_c')
sys.path.append('partition')
import libcp
import libply_c
from graphs import *
from provider import *
sys.path.append('./providers')
from datasets import *

import torch
import torch.nn as nn
from providers.datasets import HelixDataset
from collections import defaultdict
import h5py
import os
from plyfile import PlyData, PlyElement
import open3d as o3d

sys.path.append('learning')
sys.path.append('partition')
import spg
import graphnet
import pointnet
import metrics
import provider
import s3dis_dataset
import custom_dataset


# ### 1.2.Partitionning 

# In[2]:


def _partition(path_to_pcl, k_nn_geof = 45, k_nn_adj = 10, lambda_edge_weight = 1., reg_strength = 0.03, d_se_max = 0, voxel_width = 0.03, ver_batch = 0, overwrite = 0 ):
        """ Large-scale Point Cloud Segmentation with Superpoint Graphs
        Input : - path_to_pcl : path to the raw point cloud .ply file.
                - k_nn_geof : number of neighbors for the geometric features, type=int
                - k_nn_adj : adjacency structure for the minimal partition, type=int
                - lambda_edge_weight : parameter determine the edge weight for minimal part, type=float
                - reg_strength : regularization strength for the minimal partition, type=float
                - d_se_max : max length of super edges, type=float
                - voxel_width : voxel size when subsampling (in m), type=float
                - ver_batch : Batch size for reading large files, 0 do disable batch loading, type=int
                - overwrite : Wether to read existing files or overwrite them, type=int
        """
        
        root, file =  os.path.dirname(os.path.dirname(os.path.split(os.path.abspath(path_to_pcl))[0])), os.path.split(os.path.abspath(path_to_pcl))[1]
        root = root + '/'
        
        helix_data = HelixDataset()
        folder = helix_data.folders[0]
        n_labels = len(helix_data.labels.keys())
        
        times = [0,0,0] #time for computing: features / partition / spg

        if not os.path.isdir(root + "clouds"):
            os.mkdir(root + "clouds")
        if not os.path.isdir(root + "features"):
            os.mkdir(root + "features")
        if not os.path.isdir(root + "superpoint_graphs"):
            os.mkdir(root + "superpoint_graphs")
            
        print("=================\n   "+ 'Start Partitioning {}'.format(file)+"\n=================")

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

        if not os.path.isfile(data_folder + file):
            raise ValueError('{} does not exist in {}'.format(file, data_folder))

        file_name   = os.path.splitext(os.path.basename(file))[0]

        data_file   = data_folder      + file_name + helix_data.extension
        cloud_file  = cloud_folder     + file_name
        fea_file    = fea_folder       + file_name + '.h5'
        spg_file    = spg_folder       + file_name + '.h5'
       
        #--- build the geometric feature file h5 file ---
        if os.path.isfile(fea_file) and not overwrite:
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
        else :
            print("    creating the feature file...")
            #--- read the data files and compute the labels---
            
            xyz = helix_data.read_pointcloud(data_file).astype(dtype='float32')
            if voxel_width > 0:
                xyz = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
            labels = []
            rgb = []
            
            start = timer()
            #---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
            #---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
        #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            features = geof
            geof[:,3] = 2. * geof[:, 3]
            
            graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
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
        print("=================\n   "+ 'Ended Partitioning {}'.format(file)+"\n=================")
        return root[:-1], folder + file_name


# In[3]:


#root, file = _partition('data/TEST/data/test/room_1900.ply')


# In[4]:


# normal if it fails
#_partition('data/TEST/data/test/room_19065.ply')


# ## ***2.Embedding Semantic Informations***

# ### 2.1.Import

# In[5]:


import torch
import torch.nn as nn
from providers.datasets import HelixDataset
from collections import defaultdict
import sys
import numpy as np
import h5py
import os
from plyfile import PlyData, PlyElement
import open3d as o3d

sys.path.append('learning')
sys.path.append('partition')
import spg
import graphnet
import pointnet
import metrics
import provider
import s3dis_dataset
import custom_dataset


# ### 2.2.Loading model and Weights

# In[7]:


def load_weights(model_path,model_config,db_info):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    
    checkpoint['args'].model_config = model_config 
    cloud_embedder = pointnet.CloudEmbedder(checkpoint['args'])
    model = create_model(checkpoint['args'], db_info) #use original arguments, architecture can't change    
    model.load_state_dict(checkpoint['state_dict'])
    return model, cloud_embedder, checkpoint['args']


# In[8]:


def create_model(args, dbinfo):
    """ Creates model """
    model = nn.Module()

    nfeat = args.ptn_widths[1][-1]
    model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)

    model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()]))) 
    if args.cuda: 
        model.cuda()
    return model 


# In[9]:


"""
MODEL_PATH = 'results/s3dis/bw/cv1/model.pth.tar'
model_config = 'gru_10_0,f_13'
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
pc_attribs = 'xyzelspvXYZ'
dbinfo = HelixDataset().get_info(edge_attribs,pc_attribs)
"""
#model,cloud_embedder, args = load_weights(MODEL_PATH,model_config,dbinfo)


# ### 2.3.Run Inferences

# In[12]:


def predict(args, root):
    args.ROOT_PATH = root
    HelixDataset().preprocess_pointclouds(args.ROOT_PATH)
    create_dataset = HelixDataset().get_datasets

    collected = defaultdict(list)
    for ss in range(args.test_multisamp_n):
        eval_data = create_dataset(args,ss)[1]
        loader = torch.utils.data.DataLoader(eval_data, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=8)
        for bidx, (targets, GIs, clouds_data) in enumerate(loader):
            model.ecc.set_info(GIs, args.cuda)
            label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
            data = clouds_data
            embeddings = cloud_embedder.run(model, *clouds_data)
            outputs = model.ecc(embeddings)
            fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
            collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))
    predictions = {}
    with h5py.File(os.path.join(args.ROOT_PATH, 'predictions.h5'), 'w') as hf:
        for fname,output in collected.items():
            o_cpu, t_cpu, tvec_cpu = list(zip(*output))
            o_cpu = np.mean(np.stack(o_cpu,0),0)
            prediction = np.argmax(o_cpu,axis=-1)
            predictions[fname] = prediction
            hf.create_dataset(name=fname, data=prediction) #(0-based classes)
    return predictions


# In[13]:


#predictions = predict(args, root)


# ### 2.4.Outputs Segmented Point Cloud

# In[15]:


def visualise(root_path, filename, predictions):
    n_labels = 13
    
    folder = os.path.split(filename)[0] + '/'
    file_name = os.path.split(filename)[1]
    
    #---load the values------------------------------------------------------------
    fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
    spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
    ply_folder = os.path.join(root_path,'clouds',folder)
    ply_file   = os.path.join(ply_folder,file_name)

    if not os.path.isdir(ply_folder ):
        os.mkdir(ply_folder)
    if (not os.path.isfile(fea_file)) :
        raise ValueError("%s does not exist and is needed" % fea_file)
    
    geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

    if not os.path.isfile(spg_file):    
        raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
    else:
        graph_spg, components, in_component = provider.read_spg(spg_file)
        
    pred_red  = predictions[filename]        
    if (len(pred_red) != len(components)):
        raise ValueError("It looks like the spg is not adapted to the result file") 
    pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
    
    print("writing the prediction file...")
    provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  args.dataset)


# In[16]:


#visualise(root, file, predictions)


# # **Regrouping in a Class**

# In[3]:


class PointCloudSegmentation(object):
    """
    Collection of functions used to segment a point cloud
    """
    
    def __init__(self, MODEL_PATH, model_config, edge_attribs, pc_attribs):
        self._model_path = MODEL_PATH
        self._model_config = model_config
        self._edge_attribs = edge_attribs
        self._pc_attribs = pc_attribs
        self._dbinfo = HelixDataset().get_info(edge_attribs,pc_attribs)
        self._model = None
        self._cloud_embedder = None
        self._args = None
        
    
    def process(self, input_pcl, visualize = True):
        """ take a raw point cloud as input and output a segmented point cloud in a .ply file"""
        root, file = self._partition(input_pcl)
        print("=================\n   "+ 'Running Inferences' +"\n=================")
        predictions = self._predict(root, file)
        print("=================\n   "+ 'Outputing Segmented Point Cloud' +"\n=================")
        if visualize :
            self._visualise(root, file , predictions)
        return predictions
    
    
    def load_model(self):
        """ load the weiths of the model """
        print("=================\n   "+ 'Preparing Model' +"\n=================")
        print("=> loading checkpoint '{}'".format(self._model_path))
        checkpoint = torch.load(self._model_path)

        checkpoint['args'].model_config = self._model_config 
        cloud_embedder = pointnet.CloudEmbedder(checkpoint['args'])
        model = self._create_model(checkpoint['args'], self._dbinfo) #use original arguments, architecture can't change    
        model.load_state_dict(checkpoint['state_dict'])
        self._model = model
        self._cloud_embedder = cloud_embedder
        self._args = checkpoint['args']
        return 
        
        
    def _create_model(self, args, dbinfo):
        """ Creates the model """
        model = nn.Module()
        nfeat = args.ptn_widths[1][-1]
        model.ecc = graphnet.GraphNetwork(args.model_config, nfeat, [dbinfo['edge_feats']] + args.fnet_widths, args.fnet_orthoinit, args.fnet_llbias,args.fnet_bnidx, args.edge_mem_limit)
        model.ptn = pointnet.PointNet(args.ptn_widths[0], args.ptn_widths[1], args.ptn_widths_stn[0], args.ptn_widths_stn[1], dbinfo['node_feats'], args.ptn_nfeat_stn, prelast_do=args.ptn_prelast_do)
        print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()]))) 
        if args.cuda: 
            model.cuda()
        return model 
    
    
    def _partition(self, path_to_pcl, k_nn_geof = 45, k_nn_adj = 10, lambda_edge_weight = 1., reg_strength = 0.03, d_se_max = 0, voxel_width = 0.03, ver_batch = 0, overwrite = 0 ):
            """ Large-scale Point Cloud Segmentation with Superpoint Graphs
            Input : - path_to_pcl : path to the raw point cloud .ply file.
                    - k_nn_geof : number of neighbors for the geometric features, type=int
                    - k_nn_adj : adjacency structure for the minimal partition, type=int
                    - lambda_edge_weight : parameter determine the edge weight for minimal part, type=float
                    - reg_strength : regularization strength for the minimal partition, type=float
                    - d_se_max : max length of super edges, type=float
                    - voxel_width : voxel size when subsampling (in m), type=float
                    - ver_batch : Batch size for reading large files, 0 do disable batch loading, type=int
                    - overwrite : Wether to read existing files or overwrite them, type=int
            """

            root, file =  os.path.dirname(os.path.dirname(os.path.split(os.path.abspath(path_to_pcl))[0])), os.path.split(os.path.abspath(path_to_pcl))[1]
            root = root + '/'

            helix_data = HelixDataset()
            folder = helix_data.folders[0]
            n_labels = len(helix_data.labels.keys())

            times = [0,0,0] #time for computing: features / partition / spg

            if not os.path.isdir(root + "clouds"):
                os.mkdir(root + "clouds")
            if not os.path.isdir(root + "features"):
                os.mkdir(root + "features")
            if not os.path.isdir(root + "superpoint_graphs"):
                os.mkdir(root + "superpoint_graphs")

            print("=================\n   "+ 'Start Partitioning {}'.format(file)+"\n=================")

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

            if not os.path.isfile(data_folder + file):
                raise ValueError('{} does not exist in {}'.format(file, data_folder))

            file_name   = os.path.splitext(os.path.basename(file))[0]

            data_file   = data_folder      + file_name + helix_data.extension
            cloud_file  = cloud_folder     + file_name
            fea_file    = fea_folder       + file_name + '.h5'
            spg_file    = spg_folder       + file_name + '.h5'

            #--- build the geometric feature file h5 file ---
            if os.path.isfile(fea_file) and not overwrite:
                print("    reading the existing feature file...")
                geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
            else :
                print("    creating the feature file...")
                #--- read the data files and compute the labels---

                xyz = helix_data.read_pointcloud(data_file).astype(dtype='float32')
                if voxel_width > 0:
                    xyz = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
                labels = []
                rgb = []

                start = timer()
                #---compute 10 nn graph-------
                graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)
                #---compute geometric features-------
                geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
                end = timer()
                times[0] = times[0] + end - start
                del target_fea
                write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
            #--compute the partition------
            sys.stdout.flush()
            if os.path.isfile(spg_file) and not overwrite:
                print("    reading the existing superpoint graph file...")
                graph_sp, components, in_component = read_spg(spg_file)
            else:
                print("    computing the superpoint graph...")
                #--- build the spg h5 file --
                features = geof
                geof[:,3] = 2. * geof[:, 3]

                graph_nn["edge_weight"] = np.array(1. / (lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
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
            return root[:-1], folder + file_name
    
    def _predict(self, root, file):
        self._args.ROOT_PATH = root
        file_name = os.path.splitext(os.path.basename(file))[0]+'.h5'
        HelixDataset().preprocess_pointclouds(self._args.ROOT_PATH, single_file = True, filename = file_name)
        create_dataset = HelixDataset().get_datasets

        collected = defaultdict(list)
        for ss in range(self._args.test_multisamp_n):
            eval_data = create_dataset(self._args,ss)[1]
            loader = torch.utils.data.DataLoader(eval_data, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=8)
            for bidx, (targets, GIs, clouds_data) in enumerate(loader):
                self._model.ecc.set_info(GIs, self._args.cuda)
                label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()
                data = clouds_data
                embeddings = self._cloud_embedder.run(self._model, *clouds_data)
                outputs = self._model.ecc(embeddings)
                fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
                collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))
        predictions = {}
        with h5py.File(os.path.join(self._args.ROOT_PATH, os.path.splitext(os.path.basename(file))[0] +'_predictions.h5'), 'w') as hf:
            for fname,output in collected.items():
                o_cpu, t_cpu, tvec_cpu = list(zip(*output))
                o_cpu = np.mean(np.stack(o_cpu,0),0)
                prediction = np.argmax(o_cpu,axis=-1)
                predictions[fname] = prediction
                hf.create_dataset(name=fname, data=prediction) #(0-based classes)
        return predictions  
    
    
    def _visualise(self, root_path, filename,predictions):
        n_labels = 13

        folder = os.path.split(filename)[0] + '/'
        file_name = os.path.split(filename)[1]

        #---load the values------------------------------------------------------------
        fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
        spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
        ply_folder = os.path.join(root_path,'clouds',folder)
        ply_file   = os.path.join(ply_folder,file_name)

        if not os.path.isdir(ply_folder ):
            os.mkdir(ply_folder)
        if (not os.path.isfile(fea_file)) :
            raise ValueError("%s does not exist and is needed" % fea_file)

        geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

        if not os.path.isfile(spg_file):    
            raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
        else:
            graph_spg, components, in_component = provider.read_spg(spg_file)

        pred_red  = predictions[filename]        
        if (len(pred_red) != len(components)):
            raise ValueError("It looks like the spg is not adapted to the result file") 
        pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))

        print("writing the prediction file (i.e Semantic Segmented Point Cloud) in {}...".format(ply_folder))
        provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  self._args.dataset)
        
    


# # **How to use the Class**

# ## Initialize the model

# In[2]:


"""
MODEL_PATH = 'results/s3dis/bw/cv1/model.pth.tar'
model_config = 'gru_10_0,f_13'
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
pc_attribs = 'xyzelspvXYZ'
"""


# In[3]:


#model = PointCloudSegmentation(MODEL_PATH, model_config, edge_attribs, pc_attribs)


# ## Load the Weights

# In[4]:


#model.load_model()


# ## Segment the Point Cloud

# In[5]:


#predictions = model.process('data/TEST/data/test/99DuxtonRd.ply', visualize = True ) #set visualize to True if you want to write out the segmented point cloud.

