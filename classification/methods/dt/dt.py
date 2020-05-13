#!/usr/local/bin/python
# -*- coding : utf-8 -*-
import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
from text_classification.settings import TENSORBOARD_DIR,STATICFILES_DIRS
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition,  metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# import re
# reString = "[a-zA-z0-9]+"  #用来手动分词所用
# re.findall(reString,text)


class DT():
    """
    决策树
    """
    def __init__(self, trpath, tepath,criterion,max_depth,max_features):
        xtrain_svd_scl, xvalid_svd_scl, ytrain = self.getSource(trpath, tepath) #数据预处理
        self.train(xtrain_svd_scl,xvalid_svd_scl,ytrain,criterion,int(max_depth),int(max_features))  # 获取决策树模型

    ## 获取文件数据
    def getSource(self,trpath,tepath):
        '''
        数据预处理
        :param trpath:
        :param tepath:
        :return:
        '''
        train = pd.read_csv(trpath)
        # test = pd.read_csv(tepath)
        # stop_words = stopwords.words('english')

        self.lbl_enc = preprocessing.LabelEncoder()  # y的标签
        self.y = self.lbl_enc.fit_transform(train.labels.values)
        xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, self.y, stratify=self.y,
                                                          random_state=42, test_size=0.1, shuffle=True)
        self.yvalid = yvalid

        self.tfv = TfidfVectorizer(min_df=3, max_features=None,
                                   strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                   ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                   stop_words='english')
        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        self.tfv.fit(list(xtrain) + list(xvalid))
        xtrain_tfv = self.tfv.transform(xtrain)
        xvalid_tfv = self.tfv.transform(xvalid)

        self.svd = decomposition.TruncatedSVD(n_components=120)
        self.svd.fit(xtrain_tfv)
        xtrain_svd = self.svd.transform(xtrain_tfv)
        xvalid_svd = self.svd.transform(xvalid_tfv)

        # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
        self.scl = preprocessing.StandardScaler()
        self.scl.fit(xtrain_svd)
        xtrain_svd_scl = self.scl.transform(xtrain_svd)
        xvalid_svd_scl = self.scl.transform(xvalid_svd)

        return xtrain_svd_scl,xvalid_svd_scl,ytrain


    def train(self, xtrain_svd_scl,xvalid_svd_scl,ytrain,criterion,max_depth=10,max_features=100):
        # 训练数据
        self.clf = tree.DecisionTreeClassifier(criterion = criterion,max_depth=max_depth,max_features=max_features)  # 决策树
        self.clf.fit(xtrain_svd_scl, ytrain)
        self.valid_predictions = self.clf.predict(xvalid_svd_scl)


    def visualization(self,dataSetid,criterion,max_depth,max_features):
        feature_names = list(self.tfv.vocabulary_.keys())
        class_names = list(self.lbl_enc.classes_)
        # dot_data = StringIO()
        # tree.export_graphviz(self.clf, out_file=dot_data, feature_names=feature_names,
        #                      class_names=class_names, filled=True, rounded=True,
        #                      special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        dot_data = tree.export_graphviz(self.clf,out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_jpg(os.path.join(STATICFILES_DIRS[0],"dt",str(dataSetid)+criterion+max_depth+max_features+"dt.jpg"))
        # print('Visible tree plot saved as pdf.')
def getImageUrl(dataSetid,criterion,max_depth,max_features):
    return "/static/dt/"+str(dataSetid)+str(criterion)+str(max_depth)+str(max_features)+"dt.jpg"

def predict_text(model,text_data):

    xvalid_tfv = model.tfv.transform([text_data])
    xvalid_svd = model.svd.transform(xvalid_tfv)
    xvalid_svd_scl = model.scl.transform(xvalid_svd)
    predictions = model.clf.predict(xvalid_svd_scl)
    predictions_label = model.lbl_enc.inverse_transform(predictions)
    return  predictions_label


if __name__ == '__main__':
    decision = DT(r"E:\text_classification\static\media\data\train.csv")
    predict = decision.predict(r"E:\text_classification\static\media\data\train.csv")
    test_data = pd.read_csv(r"E:\text_classification\static\media\data\train.csv")
    real = decision.lbl_enc.transform(test_data.labels.values)
    decision.metrics(real, predict)

