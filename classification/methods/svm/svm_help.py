import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import os
from sklearn import metrics
import pickle
from  text_classification.settings import TENSORBOARD_DIR

trpath = r"E:\text_classification\static\media\data\train.csv"
tepath = " "
train = pd.read_csv(trpath)
# test = pd.read_csv(tepath)
stop_words = stopwords.words('english')

lbl_enc = preprocessing.LabelEncoder()  # y的标签
y = lbl_enc.fit_transform(train.author.values)

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, stratify=y,
                                                  random_state=42, test_size=0.1, shuffle=True)


tfv = TfidfVectorizer(min_df=3, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)


svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)


# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)

kernel  = "linear" # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
clf = SVC(C=1.0, probability=True,kernel=kernel)  # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)

# Fitting TF-IDF to both training and test sets (semi-supervised learning)

xvalid_tfv = tfv.transform(xvalid)

xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
xvalid_svd_scl = scl.transform(xvalid_svd)

yvalid_predit = clf.predict(xvalid_svd_scl)

acc =metrics.accuracy_score(yvalid,yvalid_predit)
print(acc)