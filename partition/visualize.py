"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    
this functions outputs varied ply file to visualize the different steps
"""
import os.path
import numpy as np
import argparse
import sys
sys.path.append("./partition/")
from plyfile import PlyData, PlyElement
from provider import *
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--dataset', default='s3dis', help='dataset name: sema3d|s3dis')
parser.add_argument('--ROOT_PATH', default='/mnt/bigdrive/loic/S3DIS', help='folder containing the ./data folder')
parser.add_argument('--res_file', default='../models/cv1/predictions_val', help='folder containing the results')
parser.add_argument('--supervized_partition',type=int,  default=0)
parser.add_argument('--file_path', default='Area_1/conferenceRoom_1', help='file to output (must include the area / set in its path)')
parser.add_argument('--upsample', default=0, type=int, help='if 1, upsample the prediction to the original cloud (if the files is huge it can take a very long and use a lot of memory - avoid on sema3d)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files')
parser.add_argument('--output_type', default='igfpres', help='which cloud to output: i = input rgb pointcloud \
                    , g = ground truth, f = geometric features, p = partition, r = prediction result \
                    , e = error, s = SPG')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.ROOT_PATH+'/'
rgb_out = 'i' in args.output_type
gt_out  = 'g' in args.output_type
fea_out = 'f' in args.output_type
par_out = 'p' in args.output_type
res_out = 'r' in args.output_type
err_out = 'e' in args.output_type
spg_out = 's' in args.output_type
folder = os.path.split(args.file_path)[0] + '/'
file_name = os.path.split(args.file_path)[1]

if args.dataset == 's3dis':
    n_labels = 13
if args.dataset == 'sema3d':
    n_labels = 8   
if args.dataset == 'vkitti':
    n_labels = 13
if args.dataset == 'custom_dataset':
    n_labels = 10    
#---load the values------------------------------------------------------------
fea_file   = root + "features/"          + folder + file_name + '.h5'
if not os.path.isfile(fea_file) or args.supervized_partition:
    fea_file   = root + "features_supervision/"          + folder + file_name + '.h5'
spg_file   = root + "superpoint_graphs/" + folder + file_name + '.h5'
ply_folder = root + "clouds/"            + folder 
ply_file   = ply_folder                  + file_name
res_file   = args.res_file + '.h5'

if not os.path.isdir(root + "clouds/"):
    os.mkdir(root + "clouds/" )
if not os.path.isdir(ply_folder ):
    os.mkdir(ply_folder)
if (not os.path.isfile(fea_file)) :
    raise ValueError("%s does not exist and is needed" % fea_file)
    
geof, xyz, rgb, graph_nn, labels = read_features(fea_file)

if (par_out or res_out) and (not os.path.isfile(spg_file)):    
    raise ValueError("%s does not exist and is needed to output the partition  or result ply" % spg_file) 
else:
    graph_spg, components, in_component = read_spg(spg_file)
if res_out or err_out:
    if not os.path.isfile(res_file):
        raise ValueError("%s does not exist and is needed to output the result ply" % res_file) 
    try:
        pred_red  = np.array(h5py.File(res_file, 'r').get(folder + file_name))        
        if (len(pred_red) != len(components)):
            raise ValueError("It looks like the spg is not adapted to the result file") 
        pred_full = reduced_labels2full(pred_red, components, len(xyz))
    except OSError:
        raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))
#---write the output clouds----------------------------------------------------
if rgb_out:
    print("writing the RGB file...")
    write_ply(ply_file + "_rgb.ply", xyz, rgb)
    
if gt_out: 
    print("writing the GT file...")
    prediction2ply(ply_file + "_GT.ply", xyz, labels, n_labels, args.dataset)
    
if fea_out:
    print("writing the features file...")
    geof2ply(ply_file + "_geof.ply", xyz, geof)
    
if par_out:
    print("writing the partition file...")
    partition2ply(ply_file + "_partition.ply", xyz, components)
    
if res_out and not bool(args.upsample):
    print("writing the prediction file...")
    prediction2ply(ply_file + "_pred.ply", xyz, pred_full+1, n_labels,  args.dataset)
    
if err_out:
    print("writing the error file...")
    error2ply(ply_file + "_err.ply", xyz, rgb, labels, pred_full+1)
    
if spg_out:
    print("writing the SPG file...")
    spg2ply(ply_file + "_spg.ply", graph_spg)
    
if res_out and bool(args.upsample):
    if args.dataset=='s3dis':
        data_file   = root + 'data/' + folder + file_name + '/' + file_name + ".txt"
        xyz_up, rgb_up = read_s3dis_format(data_file, False)
    elif args.dataset=='sema3d':#really not recommended unless you are very confident in your hardware
        data_file  = data_folder + file_name + ".txt"
        xyz_up, rgb_up = read_semantic3d_format(data_file, 0, '', 0, args.ver_batch)
    elif args.dataset=='custom_dataset':
        data_file  = data_folder + file_name + ".ply"
        xyz_up, rgb_up = read_ply(data_file)
    del rgb_up
    pred_up = interpolate_labels(xyz_up, xyz, pred_full, args.ver_batch)
    print("writing the upsampled prediction file...")
    prediction2ply(ply_file + "_pred_up.ply", xyz_up, pred_up+1, n_labels, args.dataset)
    
