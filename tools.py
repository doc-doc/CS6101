import numpy as np
import scipy.misc as smc

class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for feed to NN.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def data_augmentation(self, im, size=[224,224]):
        """

        :param im:
        :return:
        """
        im = smc.imresize(im, size)
        return im


    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """
        im = self.data_augmentation(im)
        if len(im.shape) < 3:
            im = np.tile(im[:,:,np.newaxis],(1,1,3))
        elif im.shape[2] > 3:
            im = im[:,:,0:3]
        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        #im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)
