# ====================================================
# @Time    : 2018/9/27 19:56
# @Author  : Xiao Junbin
# @Email   : xiaojunbin@u.nus.edu
# @File    : caption.py
# ====================================================
import sys
import numpy as np
from pickle import load,dump
from preprocess import load_doc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu


def load_set(filename):
    '''
    load a pre-defined list of photo identifiers
    :param filename:
    :return:
    '''
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        #identifier = line.split('.')[0]
        dataset.append(line)
    return dataset


def load_clean_descriptions(filename, dataset):
    '''
    load clean description into memory
    :param filename:
    :param dataset:
    :return:
    '''
    doc = load_doc(filename)
    descriptions = dict()
    i = 1
    for line in doc.split('\n'):
        tokens = line.split()
        #print(i, tokens)
        if len(tokens)== 0:break
        image_id,image_desc = tokens[0],tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq '+ ' '.join(image_desc)+ ' endseq'
            descriptions[image_id].append(desc)
        i += 1
    return descriptions

def load_photo_features(filename, dataset):
    '''
    Load image features
    :param filename:
    :param dataset
    :return:
    '''
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    '''
    convert descriptions to list
    :param descriptions:
    :return:
    '''
    all_desc = list()
    for key in descriptions:
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    '''
    fit a tokenizer given caption descriptions
    :param decriptions:
    :return:
    '''
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def get_max_length(descriptions):
    '''
    calculate the length of the description with the most words
    :param descriptions:
    :return:
    '''
    lines = to_lines(descriptions)
    max_len = 1
    for desc in lines:
        clen = len(desc.split())
        if  clen > max_len:
            max_len = clen
    return max_len

def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    '''
    X1,		X2 (text sequence), 						y (word)
    photo	startseq, 							little
    photo	startseq, little,						girl
    photo	startseq, little, girl, 					running
    photo	startseq, little, girl, running, 			        in
    photo	startseq, little, girl, running, in, 		                field
    photo	startseq, little, girl, running, in, field, endseq
    create sequences of images, input sequences and output words for an image
    :param tokenizer:
    :param max_length:
    :param descriptions:
    :param photos:
    :return:
    '''
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            #encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            #split the sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                #encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


def create_sequence_batch(tokenizer, max_length, desc_list, photo, vocab_size):
    '''
    Create sequences of images, input sequences and output words for an image
    :param tokenizer:
    :param max_length:
    :param desc_list:
    :return:
    '''
    X1,X2,y = [],[],[]
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1),np.array(X2),np.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    '''
    Yield data for model.fit generator
    :param descriptions:
    :param photos:
    :param tokenizer:
    :param max_length:
    :return:
    '''
    while True:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequence_batch(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]


def create_model(vocab_size, hiddden_layer_size, max_length):
    '''
    define the caption model
    :param vocab_size:
    :param max_length:
    :return:
    '''
    #Feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(hiddden_layer_size, activation='relu')(fe1)

    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, hiddden_layer_size, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(hiddden_layer_size)(se2)
    se4 = LSTM(hiddden_layer_size,go_backwards=True)(se2)

    # decoder model
    decoder1 = add([fe2, se3, se4])
    decoder2 = Dense(hiddden_layer_size, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


def index2word(index, tokenizer):
    '''
    map index to word
    :param index:
    :param tokenizer:
    :return:
    '''
    for word, ind in tokenizer.word_index.items():
        if ind == index:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    '''
    generate descriptions for the given photo by using the trained model
    :param model:
    :param tokenizer:
    :param photo:
    :param max_length:
    :return:
    '''
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index2word(yhat, tokenizer)
        if word is None:break
        in_text += ' '+word
        if word == 'endseq':break
    #del_red = list(set(in_text.split()))
    #in_text = ' '.join(del_red)
    return in_text

	
def generate_prob(model, tokenizer, photo, max_length):
    '''
    generate descriptions for the given photo by using the trained model
    :param model:
    :param tokenizer:
    :param photo:
    :param max_length:
    :return:
    '''
    in_text = 'startseq'
    prob = np.zeros(291)     
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0).squeeze()
        prob = np.maximum(prob, yhat[2:])
        yhat = np.argmax(yhat)
        word = index2word(yhat, tokenizer)
        if word is None:break
        in_text += ' '+word
        if word == 'endseq':break
    return prob


def generate_desc_beam_search(model, tokenizer, photo, max_length, beam_size):
    '''
    use beam search to generate descriptions
    :param model:
    :param tokenizer:
    :param photo:
    :param max_length:
    :param beam_size
    :return:
    '''
    in_text = 'startseq'
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    text_input = pad_sequences([sequence], maxlen=max_length)[0]
    k_beam = [(0, text_input)]
    flag = 0
    for l in range(max_length):
        if flag>=beam_size:break
        #len_1 = len(all_k_beams)
        all_k_beams = []
        #flag = 0
        for prob, sent_predict in k_beam:
            cur_ind = sent_predict[-1]
            cur_word = index2word(cur_ind, tokenizer)
            if cur_word == 'endseq':
                flag += 1
                continue
            predicted = model.predict([photo, np.array([sent_predict])], verbose=0)[0]
            tmp = - predicted
            sorted_predict = tmp.argsort()
            i = 0
            len_1 = len(all_k_beams)
            while len(all_k_beams) == len_1:
                #if all the candidates are unsatisfactory,slide the beam window
                candidates = sorted_predict[i*beam_size:(i+1)*beam_size][::-1]
                all_k_beams += [
                    (
                        prob+np.log(predicted[next_wid]),list(sent_predict[1:])+[next_wid]
                    )
                    for next_wid in candidates if next_wid != cur_ind
                ]
                i += 1

        if flag < beam_size :
            k_beam = sorted(all_k_beams)[-beam_size:][::-1]
    final_seq = k_beam[0][1]
    desc = [index2word(id, tokenizer) for id in final_seq if id != 0]
    desc = ' '.join(desc)
    return desc
	

def evaluate(groundtruth, predictions):
    '''
    evaluate the predictions with macro/micro precision/recall and F1 metrics
    :param groundtruth:
    :param predictions:
    :return:
    '''
    eps = sys.float_info.epsilon
    tp = groundtruth & predictions
    tp_num = np.sum(tp, axis = 0)
    grd_num = np.sum(groundtruth, axis = 0)
    pred_num = np.sum(predictions, axis = 0)
    precision = tp_num / (pred_num+eps)
    recall = tp_num /(grd_num+eps)
    F = 2*precision*recall/(precision+recall+eps)
    CP, CR, CF = np.mean(precision), np.mean(recall), np.mean(F)
    return CP,CR,CF
