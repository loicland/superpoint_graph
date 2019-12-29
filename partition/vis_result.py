import numpy as np
from open3d import *
import argparse

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--viz_file', '-f', default='', help='ply file path')
args = parser.parse_args()
pcd = read_point_cloud(args.viz_file) # Read the point cloud
draw_geometries([pcd]) # Visualize the point cloud
