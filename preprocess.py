# ====================================================
# @Time    : 2018/10/1 11:25
# @Author  : Xiao Junbin
# @Email   : xiaojunbin@u.nus.edu
# @File    : preprocess.py
# ====================================================

import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
#import panda
import csv

def load_doc(filename):
    '''
    load the file
    :param filename:
    :return:
    '''
    fp = open(filename, 'r')
    text = fp.read()
    fp.close()
    return text


def generate_descriptions(anno_file_name, vocab_file_name):
    '''
    generate descriptions according to rare first rule
    :param anno_file_name:
    :param vocab_file_name:
    :return:
    '''
    with open(anno_file_name, 'r') as f:
        annos = f.readlines()
    with open(vocab_file_name, 'r') as f:
        vocabs = f.readlines()
    word_count = dict()
    vocabs = [v.rstrip('\n') for v in vocabs]
    for word in vocabs:
        word_count[word] = 0
    indexes = []
    for anno in annos:
        anno = anno.split(' ')
        inds = []
        for i,a in enumerate(anno):
            if a == '1':
                inds.append(i)
        indexes.append(inds)
        for id in inds:
            word_count[vocabs[id]] += 1
    sorted_words  = sorted(word_count.items(), key=lambda A:A[-1], reverse=False)
    #rearrange the word order according to the sorted_words order
    descriptions = []
    for inds in indexes:
        desc = []
        for word, count in sorted_words:
            for id in inds:
                if vocabs[id] == word:
                    desc.append(word)
        descriptions.append(desc)

    return descriptions


def map_image_descriptions(datasplit, descriptions, outfile):
    '''
    create the image descriptions
    :param image_names:
    :param annotations:
    :param vocab:
    :return:
    '''
    with open(datasplit, 'r') as f:
        image_names = f.readlines()
    fout = open(outfile, 'w')
    for i,name in enumerate(image_names):
        name = name.rstrip('\n')
        tmp = name
        for words in descriptions[i]:
            tmp += ' '+words
        tmp += '\n'
        fout.writelines(tmp)

    fout.close()

def show(root, descriptions, image_name):
    '''
    visualize the image and annotations
    :param image_name:
    :param annotation_file:
    :param vocab_file:
    :return:
    '''
    root = root+'/images/'
    for name_desc in descriptions:
        row = name_desc.split()
        name, desc = row[0], row[1:]
        desc = ' '.join(desc)
        if name != image_name:
            continue
        name = osp.join(root, name+'.jpg')
        if not osp.exists(name):
            print('File {} is not existed'.format(name))
            return
        image = Image.open(name)
        plt.imshow(image)
        plt.title(desc)
        plt.show()
        return

def load_desc(desc_file):
    '''
    load the description file
    :param desc_file:
    :return:
    '''
    with open(desc_file, 'r') as f:
        descs = f.readlines()
    descs = [d.rstrip('\n') for d in descs]

    return descs


#def desciption_to_csv(description, csv_file):
#    pass


def main():
    root = 'datasets/iaprtc12/'
    anno_file_name = root + 'test_anno.txt'
    vocab_file_name = root + 'vocab.txt'
    train_file = root + 'test_list.txt'
    desc_file = root + 'test_descriptions.txt'

    descriptions = generate_descriptions(anno_file_name, vocab_file_name)
    #descriptions = load_desc(desc_file)

    #show(root, descriptions, '11/11438')
    map_image_descriptions(train_file, descriptions, desc_file)



if __name__== "__main__":
    main()
