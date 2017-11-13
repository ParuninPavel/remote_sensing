import numpy as np
import tifffile as tiff



class BatchGeneratorStatic(object):
    
    def __init__(self, batch_size = 32, dim_x = 256, dim_y = 256, dim_z = 3, mask_z = [1], shuffle = True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.mask_z = mask_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, X, y):
        # Infinite loop
        while 1:
        # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(X)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_X_temp = [X[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                list_y_temp = [y[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # Generate data
                X_out, y_out = self.__data_generation(list_X_temp, list_y_temp)

                yield X_out, y_out

    def __get_exploration_order(self, X):
        
        # Find exploration order
        indexes = np.arange(len(X))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, list_X_temp, list_y_temp):
        # X : (n_samples, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size, self.dim_x, self.dim_y, len(self.mask_z)))

        # Generate data
        for i in range(len(list_X_temp)):
            # Store volume
            img = tiff.imread(list_X_temp[i])
            img = img/1
            img = np.rollaxis(img, 0, 3)
            #print(img.shape)
            X[i, :, :, :] = img

            # Store class
            mask = tiff.imread(list_y_temp[i])
            mask = mask/1
            
            mask = np.rollaxis(mask, 0, 3)
            #print(mask.shape)
            for n, channel in enumerate(self.mask_z):
                y[i, :, :, n] = mask[:, :, channel]
        return X, y
    def debug(self, X, y):
        indexes = self.__get_exploration_order(X)
        imax = int(len(indexes)/self.batch_size)
        
        #for i in range(1):
        # Find list of IDs
        for i in range(1):
            list_X_temp = [X[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
            list_y_temp = [y[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
            # Generate data
            X_out, y_out = self.__data_generation(list_X_temp, list_y_temp)
        print(X_out.shape)
        print(y_out.shape)