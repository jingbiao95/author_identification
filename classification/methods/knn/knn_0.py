import numpy as np
import operator as opt
import pandas as pd
import nltk
import sklearn


from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
# import plotly.graph_objs as go

def normData(dataSet):
    maxVals = dataSet.max(axis=0)
    minVals = dataSet.min(axis=0)
    ranges = maxVals - minVals
    retData = (dataSet - minVals) / ranges
    return retData, ranges, minVals

def kNN(dataSet, labels, testData, k):
    distSquareMat = (dataSet - testData) ** 2 # 计算差值的平方
    distSquareSums = distSquareMat.sum(axis=1) # 求每一行的差值平方和
    distances = distSquareSums ** 0.5 # 开根号，得出每个样本到测试点的距离
    sortedIndices = distances.argsort() # 排序，得到排序后的下标
    indices = sortedIndices[:k] # 取最小的k个
    labelCount = {} # 存储每个label的出现次数
    for i in indices:
        label = labels[i]
        labelCount[label] = labelCount.get(label, 0) + 1 # 次数加一get方法为：查找字典中有无label如果有返回对应值，else返回第二个参数，此处为0
    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True) # 对label出现的次数从大到小进行排序
    return sortedCount[0][0] # 返回出现次数最大的label
    #计算机在计算的时候根本不会考虑到label是a还是b，他们只会按照下标来处理，之所以输出a，是因为计算机在计算的时候
    #排序过后得到距离最近的下标是0，这个0下标对应的label是a所以才输出a的，label根本不会参与计算。

if __name__ == "__main__":
    dataSet = np.array([[2, 3], [6, 8], [7, 10]])
    normDataSet, ranges, minVals = normData(dataSet)
    labels = ['EAP', 'HPL','MWS']
    testData = np.array([3.9, 5.5])
    normTestData = (testData - minVals) / ranges
    result = kNN(normDataSet, labels, normTestData, 1)


    #读取训练数据
    train = pd.read_csv("H:/Project/python/Kaggle/Spooky Author identification/en_train.csv")
    test = pd.read_csv("H:/Project/python/Kaggle/Spooky Author identification/en_test.csv")
    train.head()
    # print(train.head())
    # print(train.shape)#see how large the training data is.输出test的行列数

    # Storing the first text element as a string把句子分成一个个单词
    first_text = train.text.values[0]#读取第一句话
    first_text_test = test.text.values[0]
    # print(first_text)
    # print("=" * 90)
    # print(first_text.split(" "))
    first_text_list = nltk.word_tokenize(first_text)#去掉单词上带的逗号句号之类的。
    first_text_List_test = nltk.word_tokenize(first_text_test)
    # print(first_text_list)

    #查看nltk自带的stopwords，一些无实际含义，但是经常出现的词。
    stopwords = nltk.corpus.stopwords.words('english')
    # len(stopwords)
    # print(stopwords)

    #从文章中自带的句子中滤除stopwords
    first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stopwords]
    first_text_list_cleaned_test = [word for word in first_text_List_test if word.lower() not in stopwords]
    print(first_text_list_cleaned)
    # print("=" * 90)
    # print("Length of original list: {0} words\n"
    #       "Length of list after stopwords removal: {1} words"
    #       .format(len(first_text_list), len(first_text_list_cleaned)))

    #提取不同时态下，相同单词的不同形式的词干，就是背英语单词的时候，那个词根词干记忆法那个。
    # stemmer = nltk.stem.PorterStemmer()
    # 三个的输出都是run
    # print("The stemmed form of running is: {}".format(stemmer.stem("running")))
    # print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
    # print("The stemmed form of run is: {}".format(stemmer.stem("run")))
    # print("The stemmed form of leaves is: {}".format(stemmer.stem("leaves")))#输出为leav

    #另一种词干转换方式，感觉比上一种好，因为上一种对于leaves的处理结果为leav但是这一种为leaf
    # lemm = WordNetLemmatizer()
    # print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))#输出为leaf
    lemn = WordNetLemmatizer()
    for word in first_text_list_cleaned:
        lemn.lemmatize(word)#把每句话的单词提取出来
    for word in first_text_list_cleaned_test:
        lemn.lemmatize(word)
        # print(lemn.lemmatize(word))#把每句话中的单词提取出来
    #利用词袋的方法把文本转换成数字。
    # Defining our sentence
    # sentence = ["I love to eat Burgers",
    #             "I love to eat Fries"]
    vectorizer_train = CountVectorizer(min_df=0)
    vectorizer_test = CountVectorizer(min_df=0)
    # sentence_transform = vectorizer.fit_transform(sentence)
    # print(sentence_transform)

    # print("########################")
    sentence_transform = vectorizer_train.fit_transform(first_text_list_cleaned)
    sentence_transform_test = vectorizer_train.fit_transform(first_text_list_cleaned_test)
    print(sentence_transform)
    print(sentence_transform_test)
    print("The features are:\n {}".format(vectorizer_train.get_feature_names()))

    print("\nThe vectorized array looks like:\n {}".format(sentence_transform.toarray()))

    result = kNN(sentence_transform.toarray(), labels, "".format(sentence_transform_test.toarray()), 1)


