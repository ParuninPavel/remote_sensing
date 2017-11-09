class MapObject (object):
    __colors = [[255, 0, 0],
                [255, 143, 143],
                [64, 64, 64],
                [112, 64, 0],
                [3, 59, 0],
                [10, 207, 0],
                [0, 11, 135],
                [0, 21, 252],
                [247, 235, 2],
                [128, 121, 4]]
    def __init__(self, img, polygons = None, poly_grids = None, mask = None, predicted_mask = None, RGB_mask = None):
        '''Creates object that includes img (heigth, width, channels), polygons - dictionary {class:shapely MultipolygonWKT}, corresponding mask (heigth, width, num_classes)'''
        self.img = img
        self.polygons = polygons
        self.mask = mask
        self.predicted_mask = predicted_mask
        self.RGB_mask = None
        self.predicted_RGB_mask = None

    def _plot_mask_from_wkt(self, raster_img_size, geometry, class_value=1):
        img_mask = np.zeros(raster_img_size, np.uint8)
        if geometry is None:
            return img_mask
        perim_list = []
        interior_list = []
        for k in range(len(geometry)):
            poly = geometry[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = np.round(perim).astype(np.int32)
            perim_list.append(perim_c)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = np.round(interior).astype(np.int32)
                interior_list.append(interior_c)

        cv2.fillPoly(img_mask, perim_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask

    def poly2mask(self):
        mask_multi =[]
        h, w, _ = self.img.shape
        for cls in list(self.polygons.keys()):
            mask = self._plot_mask_from_wkt((h, w), self.polygons[cls])
            mask_multi.append(mask)
        mask_multi = np.array(mask_multi)
        mask_multi = np.rollaxis(mask_multi, 0,3)
        self.mask = mask_multi
        return mask_multi
        
    def mask2poly():
        raise NotImplementedError
        
    def get_random_patches(self, quantity, patch_size=(256,256), scale=1, aug=True, random_shear=False):
        # Scaling image with scaling factor (Useful for different classes)
        h,w,_ = self.img.shape
        img = cv2.resize(self.img, (int(h*scale), int(w*scale)))
        mask = cv2.resize(self.mask, (int(h*scale), int(w*scale)))
        h,w,_ = img.shape
        
        # Change size of patch - for augmentation bigger patches
        size = patch_size
        if aug:
            size = (patch_size[0]+patch_size[1],patch_size[0]+patch_size[1])
        assert h>size[0] and w>size[1], 'Patch size should be less than 0,5*image_size if aug=True or image_size if aug=False'
        print(h>size[0] and w>size[1])
        # Deriving of random coordinates
        coords = []
        for i in range (quantity):
            coord_h = random.randint(0, h-size[0])
            coord_w = random.randint(0, w-size[1])
            coords.append((coord_h, coord_w))
        
        # Deriving of raw patches
        raw_img_patches, raw_mask_patches = self._crop(coords, size, img, mask, aug, random_shear)
        
        # Creating final patches with crop of image center with size of patch_size
        img_patches = []
        mask_patches = []
        for raw_img_patch in raw_img_patches:
            h = (raw_img_patch.shape[0] - patch_size[0])//2
            w = (raw_img_patch.shape[1] - patch_size[1])//2
            img_patches.append(raw_img_patch[h:h+patch_size[0], w:w+patch_size[1]])
            
        for raw_mask_patch in raw_mask_patches:
            h = (raw_mask_patch.shape[0] - patch_size[0])//2
            w = (raw_mask_patch.shape[1] - patch_size[1])//2
            mask_patches.append(raw_mask_patch[h:h+patch_size[0], w:w+patch_size[1]]) 
        return img_patches, mask_patches
        
    def get_ordered_patches(self, patch_size=(256,256), scale=1, aug=True, random_shear=True, shear=0.15, padding=False, no_mask = False):
        h,w,_ = self.img.shape
        img = cv2.resize(self.img, (int(h*scale), int(w*scale)))
        if not no_mask:
            mask = cv2.resize(self.mask, (int(h*scale), int(w*scale)))
        print(img.shape)
        shear_int = (int(patch_size[0]*shear),int(patch_size[1]*shear))
        
        # Create reflective padding for aquiring integer number of patches
        if padding:
            h,w,_ = img.shape
            pad_h_1 = (patch_size[0]-shear_int[0]-(h-patch_size[0])%(patch_size[0]-shear_int[0]))//2
            pad_h_2 = (patch_size[0]-shear_int[0]-(h-patch_size[0])%(patch_size[0]-shear_int[0]))//2+(patch_size[0]-shear_int[0]-(h-patch_size[0])%(patch_size[0]-shear_int[0]))%2
            pad_w_1 = (patch_size[1]-shear_int[1]-(w-patch_size[1])%(patch_size[1]-shear_int[1]))//2
            pad_w_2 = (patch_size[1]-shear_int[1]-(w-patch_size[1])%(patch_size[1]-shear_int[1]))//2+(patch_size[1]-shear_int[1]-(w-patch_size[1])%(patch_size[1]-shear_int[1]))%2
            print(pad_h_1,pad_h_2,pad_w_1,pad_w_2)
            img = np.pad(img,((pad_h_1,pad_h_2),(pad_w_1,pad_w_2),(0,0)), 'reflect')
            if not no_mask:
                mask = np.pad(mask,((pad_h_1,pad_h_2),(pad_w_1,pad_w_2),(0,0)), 'reflect')
        
        # Get coordinates of patches
        h,w,_ = img.shape
        coords = []
        n_rows = (h-patch_size[0])//(patch_size[0]-shear_int[0])+1
        n_cols = (w-patch_size[1])//(patch_size[1]-shear_int[1])+1
        for i in range(n_rows):
            for j in range(n_cols):
                coord_h = i*(patch_size[0]-shear_int[0])
                coord_w = j*(patch_size[1]-shear_int[1])
                coords.append((coord_h, coord_w))
        print(img.shape)
        
        # Pad image and mask for geting bigger picture on augmentation step
        if aug:
            img = np.pad(img,((patch_size[1]//2,patch_size[1]//2),(patch_size[0]//2,patch_size[0]//2),(0,0)), 'reflect')
            if not no_mask:
                mask = np.pad(mask,((patch_size[1]//2,patch_size[1]//2),(patch_size[0]//2,patch_size[0]//2),(0,0)), 'reflect')
        # Get patches
        if aug:
            if not no_mask:
                raw_img_patches, raw_mask_patches = self._crop(coords, (patch_size[0]+patch_size[1],patch_size[0]+patch_size[1]),
                                     img, mask=mask, random_shear=random_shear, aug=aug)
            else:
                raw_img_patches, raw_mask_patches = self._crop(coords, (patch_size[0]+patch_size[1],patch_size[0]+patch_size[1]),
                                     img, random_shear=random_shear, aug=aug)
        else:
            if not no_mask:
                raw_img_patches, raw_mask_patches = self._crop(coords, (patch_size[0],patch_size[1]),
                                     img, mask=mask, random_shear=random_shear, aug=aug)
            else:
                raw_img_patches, raw_mask_patches = self._crop(coords, (patch_size[0],patch_size[1]),
                                     img, random_shear=random_shear, aug=aug)
        print(len(coords)) 
        print('____', mask.shape)
        
        # Deriving centers of images and masks with the size=patch_size
        img_patches = []
        mask_patches = []
        for raw_img_patch in raw_img_patches:
            h = (raw_img_patch.shape[0] - patch_size[0])//2
            w = (raw_img_patch.shape[1] - patch_size[1])//2
            img_patches.append(raw_img_patch[h:h+patch_size[0], w:w+patch_size[1]])
            
        for raw_mask_patch in raw_mask_patches:
            h = (raw_mask_patch.shape[0] - patch_size[0])//2
            w = (raw_mask_patch.shape[1] - patch_size[1])//2
            mask_patches.append(raw_mask_patch[h:h+patch_size[0], w:w+patch_size[1]]) 
        return img_patches, mask_patches
    
    def _augmentation(self, img, mask = None, rotation = True, mirror = True, zoom = True):
        img_out = img
        mask_out = mask
        # Mirror augmentation
        if mirror == True:
            if random.uniform(0, 1) > 0.5:
                img_out = img_out[::-1]
                if mask_out is not None:
                    mask_out = mask_out[::-1]
            if random.uniform(0, 1) > 0.5:
                img_out = img_out[:, ::-1]
                if mask_out is not None:
                    mask_out = mask_out[:, ::-1]
        # Rotation augmentaion
        if rotation == True:
            n = random.randint(0,3)
            img_out = np.rot90(img_out, n)
            if mask_out is not None:
                mask_out = np.rot90(mask_out, n)
        # Scale augmentaion
        if zoom == True:
            resize_factor = random.uniform(0.8, 1.2)
            h_img, w_img, _ = img_out.shape
            img_out = cv2.resize(img_out, (int(h_img*resize_factor),int(w_img*resize_factor)))
            if mask_out is not None:
                h_msk, w_msk, _ = mask_out.shape 
                mask_out = cv2.resize(mask_out, (int(h_msk*resize_factor), int(w_msk*resize_factor)))
            
        return img_out, mask_out
        
    def _crop(self, coords, size, img, mask=None, aug = True, random_shear = True):
        '''Function for creating crops from image with coordinates'''
        img_patches = []
        mask_patches = []
        h, w, _ = img.shape
        print(img.shape)
        for coord in coords:
            h_1 = coord[0] 
            h_2 = h_1+size[0]
            w_1 = coord[1]
            w_2 = w_1+size[1] 
            if random_shear:
                d_h = int(random.uniform(-0.1, 0.1)*size[0] )
                d_w = int(random.uniform(-0.1, 0.1)*size[1])   
                h_1+=d_h
                h_2+=d_h
                w_1+=d_w
                w_2+=d_w
            h_1 = np.clip (h_1, 0, h-size[0])
            h_2 = np.clip (h_2, 0+size[0], h)
            w_1 = np.clip (w_1, 0, w-size[1])
            w_2 = np.clip (w_2, 0+size[1], w)
            img_patch = img[h_1:h_2,w_1:w_2]
            
            #print(img_patch.shape)
            
            if mask is not None:
                mask_patch = mask[h_1:h_2,w_1:w_2]
                if aug:
                    img_patch, mask_patch = self._augmentation(img_patch, mask_patch)
                mask_patches.append(mask_patch)
            else:
                if aug:
                    img_patch, _ = self._augmentation(img_patch)
            img_patches.append(img_patch)
            
        return img_patches, mask_patches
        
    def plot_RGB_mask(self):
        h, w, n_cls = self.mask.shape
        cls_pixels = []
        for cls in range (n_cls):
            cls_pixels.append(np.sum(self.mask[:,:,cls]))
        indices = np.linspace(0, n_cls-1, n_cls).astype(int)
        indices_sorted = [x for _,x in sorted(zip(cls_pixels,indices), reverse = True)]
        RGB_mask = np.ones((h,w,3), dtype='uint8')*255
        for ind in indices_sorted:
            RGB_mask[self.mask[:,:,ind].astype(bool)] = np.array(self.__colors[ind])
        self.RGB_mask = RGB_mask
        return RGB_mask
    
    def plot_predicted_RGB_mask(self):
        h, w, n_cls = self.predicted_mask.shape
        cls_pixels = []
        for cls in range (n_cls):
            cls_pixels.append(np.sum(self.predicted_mask[:,:,cls]))
        indices = np.linspace(0, n_cls-1, n_cls).astype(int)
        indices_sorted = [x for _,x in sorted(zip(cls_pixels,indices), reverse = True)]
        RGB_mask = np.ones((h,w,3), dtype='uint8')*255
        for ind in indices_sorted:
            RGB_mask[self.predicted_mask[:,:,ind].astype(bool)] = np.array(self.__colors[ind])
        self.RGB_predicted_mask = RGB_mask
        return RGB_mask
        
    def stich_mask_patches():
        raise NotImplementedError
        
    def show(self):
        if self.img is not None:
            tiff.imshow(self.img)
            plt.show()
        if self.RGB_mask is not None:
            tiff.imshow(self.RGB_mask)
            plt.show()
    
    def compare_masks(self):
        if self.predicted_RGB_mask is not None:
            tiff.imshow(self.predicted_RGB_mask)
            plt.show()
            tiff.imshow(self.RGB_mask)
            plt.show()
        else:
            print('There is not predicted mask!')