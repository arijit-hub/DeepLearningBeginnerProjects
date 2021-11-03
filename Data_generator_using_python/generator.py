import os
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        #TODO: implement constructor

        ## Checking if the file path of the images is given correctly and setting it as a class variable ##
        assert (type(file_path) == str)
        self.file_path = file_path

        ## Getting the names of the images in the file path ##
        self.img_list = os.listdir(file_path)

        ## Checking if the label path of the images is given correctly and setting it as a class variable ##
        assert (type(label_path) == str)
        self.label_path = label_path

        ## Intaking the json file and converting into python dictionary ##
        with open(label_path) as file:
            self.label_dict = json.load(file)

        ## Setting the class names for the respective integer values ##
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        ## Checking if the batch size is given correctly and setting it as a class variable ##
        assert (type(batch_size) == int)
        self.batch_size = batch_size

        ## Checking if the image size is given correctly and setting it as a class variable ##
        assert (type(image_size) == list)
        self.H, self.W, self.C = image_size

        ## The dataset length is nothing but the total number of images ##
        self.dataset_length = len(self.img_list)


        ## Setting the boolean flags for data manipulation and augmentation ##
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        ## Making a numpy array of the total image list ##
        self.total_indices = np.array(list(range(self.dataset_length)))

        ## Setting the number of batches in an epoch ##
        self.number_of_batches = int(math.ceil(self.dataset_length / self.batch_size))

        ## Initializing the current batch number and current epoch number ##
        self.current_batch_number = 0
        self.epoch_number = 0

        ## Adding in image indices from the beginning if the last batch has less number of images ...##
        ## ... than the batch size ##
        if (self.dataset_length % self.batch_size != 0):
            remainder_imgs = self.dataset_length % self.batch_size

            self.total_indices = np.append(self.total_indices, list(range(self.batch_size - remainder_imgs)))

        ## Setting out the batches by splitting the total indices into batch-sized samples ##
        self.batch_idxs = np.split(self.total_indices, self.number_of_batches)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        ## Checking if the current batch matches with the total number of batches, ... ##
        ## ... suggesting it to have passed the last batch ##
        ## If the conditions is True, we need to set the current batch to 0 and also increase.. ##
        ## the epoch number ##
        if self.current_batch_number == self.number_of_batches:
            self.current_batch_number = 0
            self.epoch_number = self.epoch_number + 1

        ## If shuffle flag is true, we need to shuffle the batch indices after each epoch ##
        if (self.shuffle == True and self.current_batch_number == 0):
            self.batch_idxs = np.random.choice(self.total_indices,
                                               (self.number_of_batches, self.batch_size), replace=False)

        ## Getting the current batch indices ##
        current_batch_idx = self.batch_idxs[self.current_batch_number]

        ## Setting empty array for images and labels ##
        imgs = np.array([])
        labels = np.array([])

        ## getting the batch of images and labels ##
        for i, idx in enumerate(current_batch_idx):
            ## Getting the image name ##
            img_name = self.img_list[idx]

            ## Extracting the number of the image from the name ##
            ## The number coincides with the json file's depiction of key and gives out the value as the label ##
            img_number = img_name[-6:-4]

            ## Loading the image ##
            img = np.load(os.path.join(self.file_path, img_name))

            ## Applying data augmentation if the flags are True ##
            if (self.rotation or self.mirroring) == 1:
                img = self.augment(img)

            ## Resizing the image ##
            img = resize(img, (self.H, self.W, self.C))[np.newaxis, ...]

            ## Getting the label integer value ##
            label = self.label_dict[img_number]

            ## If its the first image of the batch it should be added to imgs array as it is ##
            ## If its not the first one, then it should be concatenated with the other images... ##
            ## ... based on axis 0. ##
            if i == 0:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img))
            labels = np.append(labels, label)

        self.current_batch_number = self.current_batch_number + 1

        return imgs, labels.astype(int)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        ## random mirroring ##
        if self.mirroring == True:

            flag_mirror = int(np.random.randint(2, size=1))

            if flag_mirror == 1:
                img = np.fliplr(img)

        ## random rotation ##
        if self.rotation == True:

            flag_rotation = int(np.random.randint(4, size=1))

            if flag_rotation == 1:
                img = np.rot90(img)

            elif flag_rotation == 2:
                img = np.rot90(img, 2)

            elif flag_rotation == 3:
                img = np.rot90(img, 3)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_number

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        assert (type(int_label) == int)
        return self.class_dict[int_label]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        imgs, labels = self.next()

        grid_num = math.ceil(math.sqrt(len(imgs)))

        for i, (img, label) in enumerate(zip(imgs, labels)):
            plt.subplot(grid_num, grid_num, i + 1)
            plt.imshow(img)
            plt.title(self.class_name(int(label)))
            plt.xticks([])
            plt.yticks([])

        plt.show()


# a = ImageGenerator('exercise_data',
#                    'Labels.json',
#                    4 , [64 , 64 , 3] , rotation = True , shuffle = True)
#
# a.show()