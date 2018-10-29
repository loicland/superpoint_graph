import open3d as o3d

class HelixDataset:

    def __init__(self):
        self.name = "helix_v1"
        self.folders = ["test"]
        self.extension = ".ply"
        self.labels = {
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
            'clutter': 13
        }
    
    def read_pointcloud(self,filename):
        cloud = o3d.read_point_cloud(filename)
        return np.asarray(cloud.points)
