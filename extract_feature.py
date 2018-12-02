# ====================================================
# @Time    : 2018/9/27 17:42
# @Author  : Xiao Junbin
# @Email   : xiaojunbin@u.nus.edu
# @File    : extract_feature.py
# ====================================================

from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model,load_model
import os
import os.path as osp
from argparse import ArgumentParser
from keras.utils import plot_model
import keras.backend as K


def set_gpu_id(gpu_id):
    
    # use cpu only if gpu_id is -1
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


# extract features from each photo in the directory
def extract_features(directory, filename=''):
    # load the model
    #model = VGG16()
    model = load_model('models/VGG16_iaprtc12_new_50.h5')
    # re-structure the model
    #model.layers.pop()
    #model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # summarize
    print(model.summary())
    #plot_model(model, to_file='models/my_small_vgg_net.png')
     
    # extract features from each photo
    features = dict()

    if filename == '':
        names = listdir(directory)
    else:
        with open(filename, 'r') as f:
            names = f.readlines()
        names = [n.rstrip('\n')+'.jpg' for n in names]
    image_num = len(names)
    
    input_tensor = model._feed_inputs+[K.learning_phase()]
    output_tensor = [model.layers[-4].output]
    func = K.function(inputs=input_tensor, outputs=output_tensor)

    for i,name in enumerate(names):
    	# load an image from file
        image_name = directory + '/' + name
        image = load_img(image_name, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    	# prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = func([image,0])[0]
        #feature = model.predict(image, verbose=0)
        #print(feature.shape)
        #return
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        if i % 100 == 0:
            print('{}<{}:{}'.format(i, image_num, name))
    return features



def extract_image_feature(filename):
    '''
    extract the vgg fc7 feature for an image
    :param filename:
    :return:
    '''
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)

    return feature



    
def main(args):
    # extract features from all images
    directory = 'datasets/iaprtc12/images'
    filename = 'datasets/iaprtc12/'+args.dataset+'_list.txt'
    feature_file = 'datas/'+args.dataset+'_features.pkl'

    set_gpu_id(args.gpu_id)

    features = extract_features(directory, filename)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open(feature_file, 'wb'))


if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=-1, required=False, help='default value -1 means using cpu')
    parser.add_argument('--dataset', dest='dataset', type=str, default='train', required=False, help='train or test')
    args = parser.parse_args()
    main(args)
