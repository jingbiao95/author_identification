# encoding:utf-8
import os
from text_classification.settings import MEDIA_ROOT
from classification.methods.rnn import data_processing_rnn  # RNN 依赖

class Parameters(object):
    """RNN配置参数"""

    def __init__(self):
        # 模型参数
        self.embedding_dim = 100  # 词向量维度
        self.num_classes = 10  # 类别数
        self.vocab_size = 10000  # 词汇表达小
        self.pre_trianing = None  # use vector_char trained by word2vec
        self.seq_length = 250
        self.num_layers = 2  # 隐藏层层数
        self.hidden_dim = 100  # 隐藏层神经元

        self.keep_prob = 0.5  # dropout保留比例
        self.learning_rate = 0.001  # 学习率
        self.clip = 5.0
        self.lr_decay = 0.9  # learning rate decay
        self.batch_size = 64  # 每批训练大小
        self.num_epochs = 3  # 总迭代轮次

        self.train_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.train.txt'))  # train data
        self.test_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.test.txt'))  # test data
        self.val_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/cnews.val.txt'))  # validation data

        self.vocab_filename = os.path.join(MEDIA_ROOT, os.path.normpath('data/1.vocab_word.txt'))  # vocabulary
        self.vector_word_filename = os.path.join(MEDIA_ROOT, os.path.normpath(
            'data/vector_word.txt'))  # vector_word trained by word2vec  写死不动
        self.vector_word_npz = os.path.join(MEDIA_ROOT, os.path.normpath(
            'data/1.vector_word.npz'))  # save vector_word to numpy file

    def builtPM(self,dataSet,parameters):
        self.num_layers = int(parameters["num_layers"])
        self.hidden_dim = int(parameters["hidden_dim"])
        self.keep_prob = float(parameters["keep_prob"])
        self.learning_rate = float(parameters["learning_rate"])
        self.num_epochs = int(parameters["num_epochs"])
        self.batch_size = int(parameters["batch_size"])

        self.train_filename , self.test_filename, self.val_filename = self.getFilePath(dataSet)

        self.vector_word_npz = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vector_word_npz))
        self.vocab_filename = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vocab_filename))
        self.vector_word_filename = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.vector_word_filename))
        self.pre_trianing = data_processing_rnn.get_word2vec(self.vector_word_npz)

    def getFilePath(self,dataSet):
        '''
        构建train 训练数据, test 测试数据, val 验证数据 的文件路径
        '''
        train_path = os.path.join(MEDIA_ROOT,os.path.normpath( dataSet.train.path))  # 训练数据集
        test_path = os.path.join(MEDIA_ROOT,os.path.normpath( dataSet.test.path))
        val_path = os.path.join(MEDIA_ROOT, os.path.normpath(dataSet.val.path))
        return train_path, test_path, val_path
