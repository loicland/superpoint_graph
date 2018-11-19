
# coding: utf-8

# # Superpoint inference
# This notebook uses a pretrained super point model to do inference on a new data set.

# In[1]:


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


# ## Import model and load weights

# In[2]:


MODEL_PATH = 'results/s3dis/bw/cv1/model.pth.tar'
model_config = 'gru_10_0,f_13'
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
pc_attribs = 'xyzelpsvXYZ'
dbinfo = HelixDataset().get_info(edge_attribs,pc_attribs)


# In[3]:


def load_model(model_path,model_config,db_info):
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    
    checkpoint['args'].model_config = model_config 
    cloud_embedder = pointnet.CloudEmbedder(checkpoint['args'])
    model = create_model(checkpoint['args'], dbinfo) #use original arguments, architecture can't change    
    model.load_state_dict(checkpoint['state_dict'])
    return model, cloud_embedder, checkpoint['args']


# In[4]:


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


# In[5]:


model,cloud_embedder, args = load_model(MODEL_PATH,model_config,dbinfo)


# In[6]:


a = model.eval()


# ## Evaluate on a dataset

# In[7]:


data_to_test = 'helix'
if data_to_test == 's3dis':
    args.ROOT_PATH = 'data/S3DIS'
    create_dataset = s3dis_dataset.get_datasets
elif data_to_test == 'helix':
    args.ROOT_PATH = 'data/helix'
    HelixDataset().preprocess_pointclouds(args.ROOT_PATH)
    create_dataset = HelixDataset().get_datasets


# In[8]:


collected, predictions = defaultdict(list), {}
for ss in range(args.test_multisamp_n):
    eval_data = create_dataset(args,ss)[1]
    loader = torch.utils.data.DataLoader(eval_data, batch_size=1, collate_fn=spg.eccpc_collate, num_workers=8)
    for bidx, (targets, GIs, clouds_data) in enumerate(loader):
        model.ecc.set_info(GIs, args.cuda)
        label_mode_cpu, label_vec_cpu, segm_size_cpu = targets[:,0], targets[:,2:], targets[:,1:].sum(1).float()

        embeddings = cloud_embedder.run(model, *clouds_data)
        outputs = model.ecc(embeddings)

        fname = clouds_data[0][0][:clouds_data[0][0].rfind('.')]
        collected[fname].append((outputs.data.cpu().numpy(), label_mode_cpu.numpy(), label_vec_cpu.numpy()))


with h5py.File(os.path.join(args.ROOT_PATH, 'predictions.h5'), 'w') as hf:
    for fname,output in collected.items():
        o_cpu, t_cpu, tvec_cpu = list(zip(*output))
        o_cpu = np.mean(np.stack(o_cpu,0),0)
        prediction = np.argmax(o_cpu,axis=-1)
        predictions[fname] = prediction
        hf.create_dataset(name=fname, data=prediction) #(0-based classes)


# In[9]:


predictions['test/room_1900']


# ## Visualisation
# Parameters for visualisation:
# * i = input rgb pointcloud
# * g = ground truth,
# * f = geometric features, 
# * p = partition, 
# * r = prediction result
# * e = error
# * s = SPG

# In[13]:


visualise(args.ROOT_PATH,'iprsf','test/room_1900',os.path.join(args.ROOT_PATH,'predictions.h5'))


# In[11]:


def visualise(root_path, output_type,filename,prediction_file):

    rgb_out = 'i' in output_type
    gt_out  = 'g' in output_type
    fea_out = 'f' in output_type
    par_out = 'p' in output_type
    res_out = 'r' in output_type
    err_out = 'e' in output_type
    spg_out = 's' in output_type
    
    n_labels = 13
    
    folder = os.path.split(filename)[0] + '/'
    file_name = os.path.split(filename)[1]
    
    #---load the values------------------------------------------------------------
    fea_file   = os.path.join(root_path,'features',folder,file_name + '.h5')
    spg_file   = os.path.join(root_path,'superpoint_graphs',folder,file_name + '.h5')
    ply_folder = os.path.join(root_path,'clouds',folder)
    ply_file   = os.path.join(ply_folder,file_name)
    res_file   = prediction_file

    if not os.path.isdir(ply_folder ):
        os.mkdir(ply_folder)
    if (not os.path.isfile(fea_file)) :
        raise ValueError("%s does not exist and is needed" % fea_file)
    
    geof, xyz, rgb, graph_nn, labels = provider.read_features(fea_file)

    if (par_out or res_out) and (not os.path.isfile(spg_file)):    
        raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
    else:
        graph_spg, components, in_component = provider.read_spg(spg_file)
        
    if res_out or err_out:
        if not os.path.isfile(res_file):
            raise ValueError("%s does not exist and is needed to output the result ply" % res_file) 
        try:
            pred_red  = np.array(h5py.File(res_file, 'r').get(folder + file_name))        
            if (len(pred_red) != len(components)):
                raise ValueError("It looks like the spg is not adapted to the result file") 
            pred_full = provider.reduced_labels2full(pred_red, components, len(xyz))
        except OSError:
            raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))

            #---write the output clouds----------------------------------------------------
    if rgb_out:
        print("writing the RGB file...")
        provider.write_ply(ply_file + "_rgb.ply", xyz, rgb)

    if gt_out: 
        print("writing the GT file...")
        provider.prediction2ply(ply_file + "_GT.ply", xyz, labels, n_labels, args.dataset)

    if fea_out:
        print("writing the features file...")
        provider.geof2ply(ply_file + "_geof.ply", xyz, geof)

    if par_out:
        print("writing the partition file...")
        provider.partition2ply(ply_file + "_partition.ply", xyz, components)

    if res_out:
        print("writing the prediction file...")
        provider.prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  args.dataset)

    if err_out:
        print("writing the error file...")
        provider.error2ply(ply_file + "_err.ply", xyz, rgb, labels, pred_full+1)

    if spg_out:
        print("writing the SPG file...")
        provider.spg2ply(ply_file + "_spg.ply", graph_spg)

#     if res_out and bool(args.upsample):
#         if args.dataset=='s3dis':
#             data_file   = root + 'data/' + folder + file_name + '/' + file_name + ".txt"
#             xyz_up, rgb_up = read_s3dis_format(data_file, False)
#         elif args.dataset=='sema3d':#really not recommended unless you are very confident in your hardware
#             data_file  = data_folder + file_name + ".txt"
#             xyz_up, rgb_up = read_semantic3d_format(data_file, 0, '', 0, args.ver_batch)
#         elif args.dataset=='custom_dataset':
#             data_file  = data_folder + file_name + ".ply"
#             xyz_up, rgb_up = read_ply(data_file)
#         del rgb_up
#         pred_up = interpolate_labels(xyz_up, xyz, pred_full, args.ver_batch)
#         print("writing the upsampled prediction file...")
#         prediction2ply(ply_file + "_pred_up.ply", xyz_up, pred_up+1, n_labels, args.dataset)


# In[2]:


cloud = o3d.read_point_cloud('data/helix/clouds/test/small_pred.ply')


# In[3]:


visualizer = o3d.JVisualizer()
visualizer.add_geometry(cloud)
visualizer.show()

