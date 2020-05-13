import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
# from keras.layers.recurrent import LSTM, GRU
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.embeddings import Embedding
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
# from keras.preprocessing import sequence, text
# from keras.callbacks import EarlyStopping
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import os


class LR():
    def __init__(self, trpath, tepath, penalty, solver, max_iter):
        xtrain, xvalid, ytrain = self.getSource(trpath, tepath)
        self.train(xtrain, xvalid, ytrain, penalty, solver, int(max_iter))  # 获取决策树模型

    # 获取文件数据
    def getSource(self, trpath, tepath):
        train = pd.read_csv(trpath)
        # test = pd.read_csv(tepath)
        # stop_words = stopwords.words('english')

        self.lbl_enc = preprocessing.LabelEncoder()  # y的标签
        self.y = self.lbl_enc.fit_transform(train.labels.values)  # 已经标签化
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

        return xtrain_svd_scl, xvalid_svd_scl, ytrain

    def train(self, xtrain_svd_scl, xvalid_svd_scl, ytrain, penalty, solver, max_iter):
        self.clf = LogisticRegression(C=1.0, max_iter=max_iter, solver=solver, penalty=penalty)
        self.clf.fit(xtrain_svd_scl, ytrain)
        self.valid_predictions = self.clf.predict(xvalid_svd_scl)


def predict_text(model, text_data):
    '''
    预测单独文本
    :param model:
    :param text_data:
    :return:
    '''
    xvalid_tfv = model.tfv.transform([text_data])
    xvalid_svd = model.svd.transform(xvalid_tfv)
    xvalid_svd_scl = model.scl.transform(xvalid_svd)
    predictions = model.clf.predict(xvalid_svd_scl)
    predictions_label = model.lbl_enc.inverse_transform(predictions)[0]
    return predictions_label


if __name__ == '__main__':
    model = LR(r"E:\text_classification\static\media\data\train.csv")
    test_data = pd.read_csv(r"E:\text_classification\static\media\data\train.csv")
    predict = predict_text(model, test_data.text.values[0])
    real = test_data.labels.values[0]
    print(predict)
    print(real)
    print(metrics.accuracy_score(model.yvalid, model.valid_predictions))
