import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, decomposition
import pandas as pd
import numpy as np
from sklearn.svm import SVC

def multiclass_logloss(actual, predicted, eps=1e-15):
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

if __name__ == '__main__':

    path =r"E:\text_classification\static\media\data\train.csv"
    # 1准备训练数据
    train_data = pd.read_csv(path)

    # 2分词
    lbl_enc = preprocessing.LabelEncoder()  # 类别标签化
    y = lbl_enc.fit_transform(train_data.author.values)  # 标签化
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                          token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train_data.text.values, y, stratify=y, random_state=42, test_size=0.1,
                                                      shuffle=True)
    tfv.fit(list(xtrain))  # tf-idf 训练器

    xtrain_tfv = tfv.transform(xtrain)  # 训练数据的词频
    xvalid_tfv  = tfv.transform(xvalid)  # 测试数据的词频



    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), stop_words='english')

    # Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
    ctv.fit(list(xtrain) + list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xvalid)



    # Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)
    xvalid_svd = svd.transform(xvalid_tfv)

    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)


    # Fitting a simple xgb on tf-idf
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_tfv.tocsc(), ytrain)
    predictions = clf.predict_proba(xvalid_tfv.tocsc())

    print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

    print("accscore:",accuracy_score(yvalid,predictions))
    # Fitting a simple xgb on tf-idf
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_ctv.tocsc(), ytrain)
    predictions = clf.predict_proba(xvalid_ctv.tocsc())

    print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print("accscore:",accuracy_score(yvalid,predictions))



    # Fitting a simple xgb on tf-idf svd features
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain_svd, ytrain)
    predictions = clf.predict_proba(xvalid_svd)

    print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print("accscore:",accuracy_score(yvalid,predictions))

