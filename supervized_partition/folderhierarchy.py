import os
import sys

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))

class FolderHierachy:
    SPG_FOLDER = "superpoint_graphs"
    EMBEDDINGS_FOLDER = "embeddings"
    SCALAR_FOLDER = "scalars"
    MODEL_FILE = "model.pth.tar"

    def __init__(self,outputdir,dataset_name,root_dir,cv_fold):
        self._root = root_dir
        if dataset_name=='s3dis':
            self._outputdir = os.path.join(outputdir,'cv' + str(cv_fold))
            self._folders = ["Area_1/", "Area_2/", "Area_3/", "Area_4/", "Area_5/", "Area_6/"]
        elif dataset_name=='sema3d':
            self._outputdir = os.path.join(outputdir,'best')
            self._folders  = ["train/", "test_reduced/", "test_full/"]
        elif dataset_name=='vkitti':
            self._outputdir = os.path.join(outputdir, 'cv' + str(cv_fold))
            self._folders  = ["01/", "02/", "03/", "04/", "05/", "06/"]
        
        if not os.path.exists(self._outputdir):
            os.makedirs(self._outputdir)

        self._spg_folder = self._create_folder(self.SPG_FOLDER)
        self._emb_folder = self._create_folder(self.EMBEDDINGS_FOLDER)
        self._scalars = self._create_folder(self.SCALAR_FOLDER)   

    @property
    def outputdir(self): return self._outputdir

    @property
    def emb_folder(self): return self._emb_folder
    
    @property
    def spg_folder(self): return self._spg_folder

    @property
    def scalars(self): return self._scalars

    @property
    def model_path(self): return os.path.join(self._outputdir, self.MODEL_FILE)

    def _create_folder(self,property_name):
        folder = os.path.join(self._root , property_name )
        if not os.path.isdir(folder):
            os.mkdir(folder)
        return folder