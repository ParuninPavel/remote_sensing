import numpy as np
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from tqdm import tqdm
from MapObject import *

class DatasetDstl(object):
    def __init__(self, imgs_1=None, imgs_2=None, imgs_3=None, pandas_grid_sizes=None, pandas_polygons=None):
        self.imgs_1 = imgs_1
        self.imgs_2 = imgs_2
        self.imgs_3 = imgs_3
        self.pandas_grid_sizes = pandas_grid_sizes
        self.pandas_polygons = pandas_polygons
        
    def generate_DS_for_fly_generator(self, dataset_folder, destination_folder, test_id=None):
        self.imgs_1 = dataset_folder+'three_band/'
        self.pandas_polygons = pd.read_csv(dataset_folder + 'train_wkt_v4.csv')
        self.pandas_grid_sizes = pd.read_csv(dataset_folder + 'grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
        id_list = self.pandas_polygons.ImageId.unique()
        if test_id is not None:
            id_list = self._fly_create_test(test_id, id_list, destination_folder)
        print('Test set was created.')
        self._fly_create_train(id_list, destination_folder)
        print('Train set was created.')
        
    def _fly_create_test(self, test_id, id_list, destination_folder):
        test_dir = destination_folder+'test/'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if test_id == -1:
            test_id = id_list[0]
        if test_id in id_list:
            index = np.argwhere(id_list==test_id)
            id_list = np.delete(id_list, index)
            
            map_obj = self.id2MapObject(self.imgs_1, test_id)
        
            if not os.path.exists(test_dir+'images'):
                os.makedirs(test_dir+'images')
            tiff.imsave(test_dir+'images/'+test_id+'.tif', map_obj.img)
            
            
            
            if not os.path.exists(test_dir+'masks'):
                os.makedirs(test_dir+'masks')
            tiff.imsave(test_dir+'masks/'+test_id+'.tif', map_obj.mask) 
        else: 
            print('There is no mask for this ID.')
        return id_list
    
    def _fly_create_train(self, id_list, destination_folder):
        for id in tqdm(id_list):
            train_dir = destination_folder+'train/'
            if not os.path.exists(train_dir+'images/'):
                os.makedirs(train_dir+'images/')
            if not os.path.exists(train_dir+'masks/'):
                os.makedirs(train_dir+'masks/')            
            
            map_obj = self.id2MapObject(self.imgs_1, id)
            
            tiff.imsave(train_dir+'images/'+id+'.tif', map_obj.img)
            tiff.imsave(train_dir+'masks/'+id+'.tif', map_obj.mask)
                     
    def id2MapObject(self, folder_path, img_id):
        img = tiff.imread(folder_path+img_id+'.tif')
        img = np.rollaxis(img, 0,3)
        H,W,_ = img.shape
        Xmax, Ymax = self.pandas_grid_sizes[self.pandas_grid_sizes.ImageId == img_id].iloc[0,1:3]
        W1 = 1.0 * W * W / (W + 1)
        H1 = 1.0 * H * H / (H + 1)
        xf = W1 / Xmax
        yf = H1 / Ymax
        
        masks_dict = {}
        for cls in self.pandas_polygons.ClassType.unique():
            mask_wkt = self.pandas_polygons[self.pandas_polygons.ImageId == img_id]
            mask_wkt = mask_wkt[mask_wkt.ClassType == cls]
            mask_wkt = mask_wkt.MultipolygonWKT.values[0]
            mask_raw = wkt_loads(mask_wkt)
            mask_raw = shapely.affinity.affine_transform(mask_raw, [xf,0,0,yf,0,0])
            masks_dict[cls] = mask_raw
        
        map_obj = MapObject(img,masks_dict)
        map_obj.poly2mask()
        return map_obj
    
    
    def generate_patches():
        raise NotImplementedError
    
    def statistics():
        raise NotImplementedError
        
    def plot_statistics():
        raise NotImplementedError