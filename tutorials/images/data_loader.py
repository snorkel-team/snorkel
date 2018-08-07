import json
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import skimage.io as io


class DataLoader(object):
    def __init__(self, data_path='/data/'):
        # fix SSL certificate issues when loading images via HTTPS
        import ssl; ssl._create_default_https_context = ssl._create_unverified_context

        current_dir = os.getcwd()
        self.data_path = current_dir + data_path

        def load_train_attr(self):
            self.train_mscoco = np.load(self.data_path + 'train_mscoco.npy')
            self.train_vg = np.load(self.data_path + 'train_vg.npy')
            self.train_vg_idx = np.load(self.data_path + 'train_vg_idx.npy')
            self.train_ground = np.load(self.data_path + 'train_ground.npy')

            self.train_object_names = np.load(self.data_path + 'train_object_names.npy')
            self.train_object_x = np.load(self.data_path + 'train_object_x.npy')
            self.train_object_y = np.load(self.data_path + 'train_object_y.npy')
            self.train_object_height = np.load(self.data_path + 'train_object_height.npy')
            self.train_object_width = np.load(self.data_path + 'train_object_width.npy')

        def load_val_attr(self):
            self.val_mscoco = np.load(self.data_path + 'val_mscoco.npy')
            self.val_vg = np.load(self.data_path + 'val_vg.npy')
            self.val_vg_idx = np.load(self.data_path + 'val_vg_idx.npy')
            self.val_ground = np.load(self.data_path + 'val_ground.npy')

            self.val_object_names = np.load(self.data_path + 'val_object_names.npy')
            self.val_object_x = np.load(self.data_path + 'val_object_x.npy')
            self.val_object_y = np.load(self.data_path + 'val_object_y.npy')
            self.val_object_height = np.load(self.data_path + 'val_object_height.npy')
            self.val_object_width = np.load(self.data_path + 'val_object_width.npy')

        load_train_attr(self)
        self.train_num = np.shape(self.train_object_names)[0]
        load_val_attr(self)
        self.val_num = np.shape(self.val_object_names)[0]
        
        with open(self.data_path + 'image_data.json') as json_data:
            self.data = json.load(json_data)

    
    def show_examples(self, annotated=False, label=-1):

        def show_image(vg_idx):
            image_url = self.data[vg_idx]['url']
            I = io.imread(image_url)
            plt.axis('off')
            plt.imshow(I)
    
        def show_image_annotated(vg_idx, idx):
            image_url = self.data[vg_idx]['url']
            I = io.imread(image_url)    
            plt.axis('off')
            plt.imshow(I)
            ax = plt.gca()
            
            for i in range(np.shape(self.val_object_y[idx])[0]):
                ax.add_patch(Rectangle((self.val_object_x[idx][i], self.val_object_y[idx][i]),
                                        self.val_object_width[idx][i],
                                        self.val_object_height[idx][i],
                                        fill=False,
                                        edgecolor='cyan',
                                        linewidth=1))

        split_idx = np.where(self.val_ground == label)[0]
        idx_list = np.random.choice(split_idx,3)

        plt.figure(figsize=(15,3))
        for j,i in enumerate(idx_list):
            plt.subplot(1,3,j+1)
            if annotated:
                show_image_annotated(int(self.val_vg_idx[i]), i)
            else:
                show_image(int(self.val_vg_idx[i]))
        plt.suptitle('Query Examples')

