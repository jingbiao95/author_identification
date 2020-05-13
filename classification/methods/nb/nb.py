#!/usr/local/bin/python
# -*- coding : utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer  # tf-idf
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB,ComplementNB
from sklearn import metrics  # 计算分类精度：
from scipy import sparse
# import re
# reString = "[a-zA-z0-9]+"  #用来手动分词所用
# re.findall(reString,text)


class NB():
    """
    决策树
    """
    def __init__(self, path,type="BernoulliNB"):
        self.type = type
        xtrain_tfv, xvalid_tfv, ytrain = self.prepare_train_data(path)
        self.train(xtrain_tfv, xvalid_tfv, ytrain,type)  # 获取决策树模型


    def prepare_train_data(self, path):
        # 1准备训练数据
        train_data = pd.read_csv(path)
        # 2分词
        self.lbl_enc = preprocessing.LabelEncoder()  # 类别标签化
        y = self.lbl_enc.fit_transform(train_data.labels.values)  # 标签化
        xtrain, xvalid, ytrain, yvalid = train_test_split(train_data.text.values, y, stratify=y,
                                                          random_state=42, test_size=0.1, shuffle=True)
        self.yvalid =yvalid
        self.tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english')

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        self.tfv.fit(list(xtrain)+list(xvalid))  # tf-idf 训练器

        xtrain_tfv = self.tfv.transform(xtrain)  # 训练数据的词频
        xvalid_tfv = self.tfv.transform(xvalid)
        # print(type(train_data))
        return xtrain_tfv, xvalid_tfv,ytrain

    def train(self, xtrain_tfv, xvalid_tfv, ytrain,type):
        # 训练数据
        if type =="BernoulliNB":
            clf = BernoulliNB()
        elif type == "GaussianNB":
            clf = GaussianNB()
            xtrain_tfv  =xtrain_tfv.toarray()
            xvalid_tfv = xvalid_tfv.toarray()
        elif type =="MultinomialNB":
            clf = MultinomialNB()
        elif type == "ComplementNB":
            clf = ComplementNB()
        #csr_matrix  Gaus
        clf.fit(xtrain_tfv, ytrain)
        self.valid_predictions = clf.predict(xvalid_tfv)
        self.clf = clf

    def __prepare_predict_data(self, path):
        # 1准备预测数据
        predict_data = pd.read_csv(path)
        # 2 对预测数据进行tf-idf转换


        predict_tfv = self.tfv.transform(predict_data.text.values)
        if self.type == "GaussianNB":
            predict_tfv = predict_tfv.toarray()
        return predict_tfv

    def predict(self, path):
        # 获取预测数据
        predict_data = self.__prepare_predict_data(path)

        # 预测类别
        predict = self.clf.predict(predict_data)
        return predict

    def get_real(self, predict):
        # 矩阵转换为对应字符
        return self.lbl_enc.inverse_transform(predict)

    def metrics(self, actual, predict):
        """

        :param actual: 一维向量
        :param predict:一维向量
        :return:
        """

        return metrics.accuracy_score(actual, predict),\
                metrics.precision_score(actual, predict, average='weighted'),\
                metrics.recall_score(actual, predict, average='weighted'),\
                metrics.f1_score(actual, predict, average='weighted')

def predict_text(model,text_data):

    xvalid_tfv = model.tfv.transform([text_data])
    if model.type == "GaussianNB":
        xvalid_tfv = xvalid_tfv.toarray()
    predictions = model.clf.predict(xvalid_tfv)
    predictions_label = model.lbl_enc.inverse_transform(predictions)[0]
    return  predictions_label

def example():
    nb = NB(r"E:\text_classification\static\media\data\train.csv","MultinomialNB")
    predict = nb.predict(r"E:\text_classification\static\media\data\train.csv")
    test_data = pd.read_csv(r"E:\text_classification\static\media\data\train.csv")
    predict = predict_text(nb, test_data.text.values[0])
    real = test_data.labels.values[0]
    print(predict)
    print(real)
    print(metrics.accuracy_score(nb.yvalid, nb.valid_predictions))

if __name__ == '__main__':
    example()
