
# coding: utf-8

# # Notebook for generating S3DIS like data from .ply

# In[4]:


import open3d as o3d
import numpy as np


# In[22]:


infile = 'data/helix2/data/test/99DuxtonRd.ply'
outfile = 'data/helix2/data/test/99DuxtonRd.txt'


# In[5]:


cloud = o3d.read_point_cloud(infile)
pts = np.asarray(cloud.points)
pts = pts - np.min(pts,axis=0,keepdims=True)


# In[12]:


colors = (np.asarray(cloud.colors)*255).astype('uint8')


# In[20]:


all_data = np.concatenate([pts,colors],axis = 1)


# In[33]:


with open(outfile,'w') as fp:
    for row in all_data:
        fp.write('%.3f %.3f %.3f %d %d %d\n' % tuple(row))


# ## Now doing the reverse

# In[11]:


infile = 'data/S3DIS/data/Area_1/conferenceRoom_1/conferenceRoom_1.txt'
outfile = 'data/helix/data/test/s3disconferenceRoom_1.ply'


# In[6]:


format_str = '%.3f %.3f %.3f %d %d %d\n'
pts = []
with open(infile,'r') as fp:
    lines = fp.readlines()
    for line in lines:
        row = line.split(' ')
        pts.append([float(row[0]),float(row[1]),float(row[2])])


# In[7]:


np.asarray(pts)


# In[8]:


cloud = o3d.PointCloud()
cloud.points = o3d.Vector3dVector(pts)


# In[12]:


o3d.write_point_cloud(outfile,cloud)

