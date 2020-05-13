# -*- coding: utf-8 -*-
import os
from text_classification.settings import  MEDIA_ROOT, CHECKPOINTS
from classification.methods.cnn import data_processing
class Parameters(object):

    def __init__(self):
        self.embedding_size = 100  # dimension of word embedding
        self.vocab_size = 10000  # number of vocabulary
        self.pre_trianing = None  # use vector_char trained by word2vec

        self.seq_length = 600  # max length of sentence
        self.num_classes = 10  # number of labels

        self.num_filters = 128  # number of convolution kernel
        self.kernel_size = [2, 3, 4]  # size of convolution kernel

        self.keep_prob = 0.5  # droppout
        self.lr = 0.001  # learning rate
        self.lr_decay = 0.9  # learning rate decay
        self.clip = 5.0  # gradient clipping threshold

        self.num_epochs = 5  # epochs
        self.batch_size = 64  # batch_size

        self.train_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.train.txt'))  # train data
        self.test_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.test.txt'))  # test data
        self.val_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.val.txt'))  # validation data

        self.vocab_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/1.vocab_word.txt'))  # vocabulary
        self.vector_word_filename = os.path.join(MEDIA_ROOT,
                                            os.path.normpath('data/vector_word.txt'))  # vector_word trained by word2vec  写死不动
        self.vector_word_npz = os.path.join(MEDIA_ROOT,
                                       os.path.normpath('data/1.vector_word.npz'))  # save vector_word to numpy file


    def builtPM(self,dataSet,parameters):

        self.num_filters = int(parameters["num_filters"])
        self.keep_prob = float(parameters["keep_prob"])
        self.lr = float(parameters["lr"])
        # self.lr_decay =float( parameters["lr_decay"])
        # self.clip = float(parameters["clip"])
        self.num_epochs = int(parameters["num_epochs"])
        self.batch_size = int(parameters["batch_size"])

        self.train_filename , self.test_filename, self.val_filename = self.getFilePath(dataSet)

        self.vector_word_npz = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vector_word_npz))
        self.vocab_filename = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vocab_filename))
        self.vector_word_filename = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vector_word_filename))
        self.pre_trianing = data_processing.get_word2vec(self.vector_word_npz)
    def getFilePath(self,dataSet):
        '''
        构建train 训练数据, test 测试数据, val 验证数据 的文件路径
        '''
        train_path = os.path.join(MEDIA_ROOT,os.path.normpath( dataSet.train.path))  # 训练数据集
        test_path = os.path.join(MEDIA_ROOT,os.path.normpath( dataSet.test.path))
        val_path = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.val.path))
        return train_path, test_path, val_path
