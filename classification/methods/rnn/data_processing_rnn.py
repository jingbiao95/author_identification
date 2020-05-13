#encoding:utf-8
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import re
import jieba
import os
from  text_classification.settings import MEDIA_ROOT

def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        word.extend(jieba.lcut(blk))
                contents.append(word)
            except:
                pass
    return labels, contents

def built_vocab_vector(dataSet, filenames, voc_size = 10000):
    '''
    去停用词，得到前9999个词，获取对应的词 以及 词向量
    :param filenames:
    :param voc_size:
    :return:
    '''
    stopword_file = os.path.join(MEDIA_ROOT,os.path.normpath('data/cnn/stopwords.txt'))  # 中文停用词 共用

    stopword = open(stopword_file, 'r', encoding='utf-8')
    stop = [key.strip(' \n') for key in stopword]

    all_data = []
    j = 1
    embeddings = np.zeros([10000, 100])
    categories = []  # 类别

    for filename in filenames:
        labels, content = read_file(filename)  #在这里就开始分词了（read_file)
        if labels not in categories:
            categories.append(labels)

        for eachline in content:
            line =[]
            for i in range(len(eachline)):
                if str(eachline[i]) not in stop:#去停用词
                    line.append(eachline[i])
            all_data.extend(line)

    counter = Counter(all_data)
    count_paris = counter.most_common(voc_size-1)
    word, _ = list(zip(*count_paris))
    # f_file = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnn/vector_word.txt'))  #产生训练集的词向量表文件
    f_file = os.path.join(MEDIA_ROOT, 'data','vector_word.txt')#  词向量
    dataSet.vector_word_filename = f_file
    f = codecs.open(f_file, 'r', encoding='utf-8')
    # vocab_word_file = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnn/vocab_word.txt'))
    vocab_word_file = os.path.join(MEDIA_ROOT, 'data',str(dataSet.id)+'.vocab_word.txt')
    dataSet.vocab_filename = vocab_word_file
    vocab_word = open(vocab_word_file, 'w', encoding='utf-8')
    for ealine in f:
        item = ealine.split(' ')
        key = item[0]
        vec = np.array(item[1:], dtype='float32')
        if key in word:
            embeddings[j] = np.array(vec)
            vocab_word.write(key.strip('\r') + '\n')
            j += 1
    # np_file = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnn/vector_word.npz'))
    np_file = os.path.join(MEDIA_ROOT, 'data',str(dataSet.id)+'.vector_word.npz')
    dataSet.vector_word_npz = np_file
    np.savez_compressed(np_file, embeddings=embeddings)

    return categories


def get_wordid(filename):
    key = open(filename, 'r', encoding='utf-8')

    wordid = {}
    wordid['<PAD>'] = 0
    j = 1
    for w in key:
        w = w.strip('\n')
        w = w.strip('\r')
        wordid[w] = j
        j += 1
    return wordid



def read_category(categories):

    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def process(filename, word_to_id, cat_to_id, max_length=250):
    labels, contents = read_file(filename)
    data_id, label_id = [], []

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad

def process_text(text_data,word_to_id, cat_to_id, max_length=250):
    '''对一行文本进行处理'''
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    try:
        text_data = text_data.rstrip()

        blocks = re_han.split(text_data)
        word = []
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        contents.append(word)
    except:
        pass

    data_id, label_id = [], []
    data_id.append([word_to_id[x] for x in contents[0] if x in word_to_id])
    data_id.append([0])

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')

    return x_pad

def get_word2vec(filename):
    with np.load(filename) as data:
        return data['embeddings']


def batch_iter(x, y,  batch_size = 64):
    data_len = len(x)
    num_batch = int((data_len - 1)/batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    '''
    np.arange(4) = [0,1,2,3]
    np.random.permutation([1, 4, 9, 12, 15]) = [15,  1,  9,  4, 12]
    '''
    x_shuff = x[indices]
    y_shuff = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i+1) * batch_size, data_len)
        yield x_shuff[start_id:end_id], y_shuff[start_id:end_id]

def sequence(x_batch):
    seq_len = []
    for line in x_batch:
        length = np.sum(np.sign(line))
        seq_len.append(length)

    return seq_len


