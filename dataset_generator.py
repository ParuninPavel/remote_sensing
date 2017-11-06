class DatasetPreparation(object):
    '''This class creates masks from polygons and generates patches dataset.'''
    def __init__(self, file_w_poly, grid_sizes, tif_folder_1 = 'Dataset/DSTL/three_band/', tif_folder_2 = None,
                 dataset_folder = 'Dataset/DSTL/'):
        self.dataset_folder = dataset_folder
        self.patch_folder = dataset_folder+'patches/'
        self.polygons = file_w_poly
        self.grid_sizes = grid_sizes
        self.tif_folder_1 = tif_folder_1
        self.tif_folder_2 = tif_folder_2
    
    def _convert_coordinates_to_raster(coords, img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        Xmax, Ymax = xymax
        H, W = img_size
        W1 = 1.0 * W * W / (W + 1)
        H1 = 1.0 * H * H / (H + 1)
        xf = W1 / Xmax
        yf = H1 / Ymax
        coords[:, 1] *= yf
        coords[:, 0] *= xf
        coords_int = np.round(coords).astype(np.int32)
        return coords_int


    def _get_xmax_ymin(self, grid_sizes_panda, imageId):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
        return (xmax, ymin)


    def _get_polygon_list(self, wkt_list_pandas, imageId, cType):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
        multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
        polygonList = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygonList = wkt_loads(multipoly_def.values[0])
        return polygonList


    def _get_and_convert_contours(self, polygonList, raster_img_size, xymax):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        perim_list = []
        interior_list = []
        if polygonList is None:
            return None
        for k in range(len(polygonList)):
            poly = polygonList[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
                interior_list.append(interior_c)
        return perim_list, interior_list


    def _plot_mask_from_contours(self, raster_img_size, contours, class_value=1):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        img_mask = np.zeros(raster_img_size, np.uint8)
        if contours is None:
            return img_mask
        perim_list, interior_list = contours
        cv2.fillPoly(img_mask, perim_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask


    def generate_mask_for_image_and_class(self, raster_size, imageId, class_type):#grid_sizes_panda=GS, wkt_list_pandas=DF):
        # __author__ = visoft
        # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        grid_sizes_panda = self.grid_sizes
        wkt_list_pandas = self.polygons
        xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
        polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
        contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
        mask = _plot_mask_from_contours(raster_size, contours, 1)
        return mask
    
    def _create_test(self, test_id, id_list):
        test_dir = self.patch_folder+'test/'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        if test_id == -1:
            test_id = id_list[0]
        if test_id in id_list:
            #id_list.remove(test_id)
            index = np.argwhere(id_list==test_id)
            id_list = np.delete(id_list, index)
            
            img = tiff.imread(self.tif_folder_1+test_id+'.tif')
            
            if not os.path.exists(test_dir+'images'):
                os.makedirs(test_dir+'images')
            tiff.imsave(test_dir+'images/'+test_id+'.tif', img)
            
            mask_multi =[]
            for cls in self.polygons.ClassType.unique():
                _, h, w = img.shape 
                mask = generate_mask_for_image_and_class((h, w), test_id, cls)
                mask_multi.append(mask)
            mask_multi = np.array(mask_multi)
            if not os.path.exists(test_dir+'masks'):
                os.makedirs(test_dir+'masks')
            tiff.imsave(test_dir+'masks/'+test_id+'.tif', mask_multi) 
        else: 
            print('There is no mask for this ID.')
        return id_list
    
#    new version of dataset generatoe 
#    def _generate_patch(self, img, mask, x, y, size=(256,256), aug=True):
#         _, h, w = img.shape
#         h_1 = x
#         h_2 = x+size[0]
#         w_1 = y
#         w_2 = y+size[1]   
#         # Augmentation
#         if aug =
#         h_1 = np.clip (h_1, 0, h-size[0])
#         h_2 = np.clip (h_2, 0+size[0], h)
#         w_1 = np.clip (w_1, 0, w-size[1])
#         w_2 = np.clip (w_2, 0+size[1], w)
#         patch_img = img[:,h_1:h_2,w_1:w_2]
#         patch_mask = mask[:,h_1:h_2,w_1:w_2]
#         return patch_img, patch_mask
    
#     def static_patch_generator(self, id_list, size = (256,256), aug = True, random = 0):
#         index = 0
#         for id in tqdm(id_list):
#             train_dir = self.patch_folder+'train/'
#             if not os.path.exists(train_dir+'images/'):
#                 os.makedirs(train_dir+'images/')
#             if not os.path.exists(train_dir+'masks/'):
#                 os.makedirs(train_dir+'masks/')            
#             img = tiff.imread(self.tif_folder_1+id+'.tif')
#             _, h, w = img.shape
#             mask_multi =[]
            
#             for cls in self.polygons.ClassType.unique():
#                 mask = generate_mask_for_image_and_class((h, w), id, cls)
#                 mask_multi.append(mask)
#             mask_multi = np.array(mask_multi)
#             if random == 0:
#                 for i in range(h//size[0]):
#                     for j in range (w//size[1]):
#                         patch_img, patch_mask = self._generate_patch(img, mask_multi, size, i*size[0], j*size[1], aug)
#                         tiff.imsave(train_dir+'images/'+str(index)+'.tif', patch_img)
#                         tiff.imsave(train_dir+'masks/'+str(index)+'.tif', patch_mask)
#                         index +=1
#             else:
                
        
    def generate_patches_from_mask(self, img, mask, size, aug=True):
        _, h, w = img.shape
        
            
        for i in range(h//size[0]):
            for j in range (w//size[1]):
                h_1 = i*size[0] 
                h_2 = (i+1)*size[0]
                w_1 = j*size[1]
                w_2 = (j+1)*size[1]   
                if aug:
                    d_h = int(random.uniform(-1, 1)*size[0]*0.15 )
                    d_w = int(random.uniform(-1, 1)*size[1]*0.15)   
                    h_1+=d_h
                    h_2+=d_h
                    w_1+=d_w
                    w_2+=d_w
                h_1 = np.clip (h_1, 0, h-size[0])
                h_2 = np.clip (h_2, 0+size[0], h)
                w_1 = np.clip (w_1, 0, w-size[1])
                w_2 = np.clip (w_2, 0+size[1], w)
                patch_img = img[:,h_1:h_2,w_1:w_2]
                patch_mask = mask[:,h_1:h_2,w_1:w_2]
                
                if aug :
                    if random.uniform(0, 1) > 0.5:
                        patch_img = patch_img[:,::-1]
                        patch_mask = patch_mask[:,::-1]
                    if random.uniform(0, 1) > 0.5:
                        patch_img = patch_img[:,:, ::-1]
                        patch_mask = patch_mask[:,:, ::-1]
                
                yield patch_img, patch_mask
            # TODO: create augmentation and random crop       
        
    def _create_train(self, id_list, size = (256,256) , aug =True):
        index = 0
        for id in tqdm(id_list):
            train_dir = self.patch_folder+'train/'
            if not os.path.exists(train_dir+'images/'):
                os.makedirs(train_dir+'images/')
            if not os.path.exists(train_dir+'masks/'):
                os.makedirs(train_dir+'masks/')            
            img = tiff.imread(self.tif_folder_1+id+'.tif')
            
            mask_multi =[]
            for cls in self.polygons.ClassType.unique():
                _, h, w = img.shape 
                mask = generate_mask_for_image_and_class((h, w), id, cls)
                mask_multi.append(mask)
            mask_multi = np.array(mask_multi)
            for aug_step in range(aug*4+1):
                for patch_img, patch_mask in self.generate_patches_from_mask(img, mask_multi, size):
                    tiff.imsave(train_dir+'images/'+str(index)+'.tif', patch_img)
                    tiff.imsave(train_dir+'masks/'+str(index)+'.tif', patch_mask)
                    index +=1  
            print(index)
            
    def _create_val_from_train(self, part = 0.1):
        val_dir = self.patch_folder+'val/'
        if not os.path.exists(val_dir+'images/'):
            os.makedirs(val_dir+'images/')
        if not os.path.exists(val_dir+'masks/'):
            os.makedirs(val_dir+'masks/')  
        
        files = os.listdir(self.patch_folder+'train/images/')
        files = np.array(files)
        msk = np.random.rand(len(files)) < part
        val_names = files[msk]
        for name in val_names:
            shutil.move(self.patch_folder+'train/images/' + name, val_dir+'images/'+name)
            shutil.move(self.patch_folder+'train/masks/' + name, val_dir+'masks/'+name)
        return len(val_names)
        
    def generate_patches_from_poly(self, size = (256,256), aug=True, test_id= None):
        '''Function for patches generation from tif and polygons. Saves results on hard drive.'''
        id_list = self.polygons.ImageId.unique()
        if test_id is not None:
            id_list = self._create_test(test_id, id_list)
        print('Test set was created.')
        self._create_train(id_list, aug=aug)
        print('Train set was created.')
        self._create_val_from_train()
        print('Validation set was created')
