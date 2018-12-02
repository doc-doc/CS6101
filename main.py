# ====================================================
# @Time    : 2018/9/28 22:27
# @Author  : Xiao Junbin
# @Email   : xiaojunbin@u.nus.edu
# @File    : main.py
# ====================================================
from caption import *
from extract_feature import extract_image_feature
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path as osp
from argparse import ArgumentParser
import numpy as np

def set_gpu_id(gpu_id):
    '''
    set gpu id
    :param gpu_id:
    '''
    gpu = ''
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def train(train_file, feature_file, desc_file):
    '''
    Train the caption model
    :param train_file:
    :param val_file:
    :param model_path:
    :return:
    '''
    train = load_set(train_file)
    print('Dataset: {}'.format(len(train)))

    train_descriptions = load_clean_descriptions(desc_file, train)
    print('Descriptions: train = {}'.format(len(train_descriptions)))

    train_features = load_photo_features(feature_file, train)
    print('Photos: train = {}'.format(len(train_features)))

    #tokenizer = create_tokenizer(train_descriptions)
    with open('datas/tokenizer.pkl','rb') as ftk:
        tokenizer = load(ftk)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary size: {}'.format(vocab_size))

    max_desc_len = get_max_length(train_descriptions)
    print('maximum description length: {}'.format(max_desc_len))

    # generator = data_generator(train_descriptions,train_features, tokenizer, max_desc_len, vocab_size)
    # inputs, outputs = next(generator)
    # print(inputs[0].shape)
    # print(inputs[1].shape)
    # print(outputs.shape)

    #X1train, X2train, ytrain = create_sequences(tokenizer, max_desc_len, train_descriptions, train_features, vocab_size )
    model = create_model(vocab_size, 1024, max_desc_len)

    epochs = 60
    steps = len(train_descriptions)
    for i in range(epochs):
        generator = data_generator(train_descriptions, train_features, tokenizer, max_desc_len, vocab_size)
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        model.save('models/model_CNN_BiLSTM_'+str(i)+'.h5')


def run_train():
    
    root = 'datasets/iaprtc12/'
    train_file = root + 'train_list.txt'
    feature_file = 'datas/train_features.pkl'
    desc_file = root + 'descriptions.txt'
    train(train_file, feature_file, desc_file)


def predict():

    with open('datas/tokenizer.pkl','rb') as f:
        tokenizer = load(f)

    test_file = 'datasets/iaprtc12/test_list.txt'
    test = load_set(test_file)

    test_features = load_photo_features('datas/test_features.pkl', test)
    test_num = len(test_features)
    
    vocab_file = 'datasets/iaprtc12/vocab.txt'
    with open(vocab_file, 'r') as f:
        vocab = f.readlines()
    vocab = [v.rstrip('\n') for v in vocab]
    
    dic = dict()
    for i in range(len(vocab)):
        dic[vocab[i]] = i
    
    model_file = 'models/model_CNN_BiLSTM1024_49.h5'
    model = load_model(model_file)
    fp = open('datas/test_Bi_beam3.txt','w')
    i = 0
    
    for image in test:
        feature = test_features[image]
        pred = ['0']*291
        sequence = []
        if i % 100 == 0:
            print(i, test_num, image)
        desc = generate_desc(model, tokenizer, feature, 25)
        '''
        prob = generate_prob(model, tokenizer, feature, 25)
        probs.append(prob)
        i += 1
        continue
        '''
        #if i % 1 == 0:
        #    print(i, test_num, image, desc)
        
        # use bean search with beam_size = 2
        #desc = generate_desc_beam_search(model, tokenizer, feature, 25, 3)
        #ignore the startseq and endseq flag
        desc = desc.split()[1:-1]
        #delete the redundant tags
        desc = list(set(desc))
        for word in desc:
            sequence.append(dic[word])
        for s in sequence:
            pred[s] = '1'
        line = ' '.join(pred)
        fp.writelines(line+'\n')
        i += 1
    fp.close()
    #np.savetxt('datas/rnn_prediction.txt', np.array(probs), fmt='%.4f', delimiter=' ')


def run_evaluate(groundtruth_file, prediction_file):
    '''
    evaluate the predictions for test images
    :param groundtruth_file:
    :param prediction_file:
    :return:
    '''
    groundtruth = np.loadtxt(groundtruth_file, dtype=int, delimiter=' ')
    predictions = np.loadtxt(prediction_file, dtype=int, delimiter=' ')
    CP, CR, CF = evaluate(groundtruth, predictions)
    groundtruth = np.reshape(groundtruth, [groundtruth.size, 1])
    predictions = np.reshape(predictions, [predictions.size, 1])
    OP, OR, OF = evaluate(groundtruth, predictions)
    print('CP\tCR\tCF\tOP\tOR\tOF')
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(CP, CR, CF, OP, OR, OF))

	
	
def demo(args):
    
    
    with open('datas/tokenizer.pkl','rb') as f:
        tokenizer = load(f)
    model_file = 'models/model_CNN_BiLSTM1024_49.h5'
    model = load_model(model_file)
    
    image_path = 'datasets/iaprtc12/images/30/' + args.image_name
    if not osp.exists(image_path):
        print('File {} is not existing.'.format(image_path))
        return
    
    feature = extract_image_feature(image_path)
    
    '''
    train = load_set('datasets/iaprtc12/train_list.txt')
    train_features = load_photo_features('datas/train_features.pkl', train)
    image_id = args.image_name.split('.')[0]
    image_id = '00/'+image_id
    feature = train_features[image_id]
    '''
    descs = dict()
    with open('datasets/iaprtc12/descriptions.txt', 'r') as f:
        test_descs = f.readlines()
    for record in test_descs:
        record = record.rstrip('\n').split()
        image_id, desc = record[0],' '.join(record[1:])
        descs[image_id] = desc
    image_id = '30/' + args.image_name.split('.')[0]

    desc = generate_desc(model, tokenizer, feature, 25)
    desc = desc.split()[1:-1] #ignore the startseq and endseq flag
    desc = list(set(desc))
    desc = ' '.join(desc)
    image = Image.open(image_path)
    plt.figure(1)
    plt.imshow(image)
    plt.title('PD: '+desc)
    plt.figure(2)
    plt.imshow(image)
    plt.title('GT: '+descs[image_id])
    plt.show()
    
def combine(res_rnn, res_cnn):
    
    cnn_res = np.loadtxt(res_cnn, dtype=float, delimiter=' ')
    rnn_res = np.loadtxt(res_rnn, dtype=int, delimiter=' ')
    
    cnn_res = cnn_res > 0.25
    rnn_res = rnn_res > 0
    cb_res = np.logical_or(cnn_res, rnn_res)
    np.savetxt('datas/cnn_rnn_predictions.txt', cb_res, fmt='%d', delimiter=' ')


def main():
    
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=-1, required=False, help='input -1 to use cpu')
    parser.add_argument('--image_name', dest='image_name', type=str, default='26.jpg', required=False, help='image name')
    args = parser.parse_args()
    set_gpu_id(args.gpu_id)
    #run_train()
    #run_evaluate('datasets/iaprtc12/test_anno.txt','datas/test_greedy.txt')
    #predict()
    combine('datas/test_Bi_beam2.txt', 'datas/cnn_predictions.txt')
    run_evaluate('datasets/iaprtc12/test_anno.txt','datas/cnn_rnn_predictions.txt')
    #demo(args)

if __name__=="__main__":
    main()
