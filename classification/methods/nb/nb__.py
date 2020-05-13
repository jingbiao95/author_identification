import numpy as np
import pandas as pd
from sklearn import preprocessing
import re


def read_data():
    train_data = pd.read_csv("../data/en_train.csv")
    test_data = pd.read_csv("../data/en_test.csv")
    sample_data = pd.read_csv("../data/sample_submission.csv")
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train_data.author.values)
    print(y)


regString = '[a-zA-Z0-9]+'  # 正则表达式


class NaiveBaye():

    def __init__(self, train_fileName):
        # 1 读取数据
        self.train_data = pd.read_csv(train_fileName)
        self.wordTable = self.getWordTable()  # 词表
        self.authorVec, self.pp = self.getPre()  # 获得先验概率

    def getWordTable(self):  # 总的词表
        wordTable = set([])
        for text in self.train_data["text"]:
            text = re.findall(regString, text)  # 利用正则表达式对句子分词
            wordTable = wordTable | set(text)
        return list(wordTable)

    def getPre(self):
        """
        获取先验概率

        :return:
        """
        # 表的形式如：{"":[],"":[],"":[]}
        authors = self.train_data["author"].unique()
        author_wordTable = np.ones((len(authors), len(self.wordTable))) # 3 * N

        author_article = [1]*len(authors)  # 统计次数      # 1 * 3
        # for author in authors:
        #     author_wordTable[authors.index(author)] = [1] * len(self.wordTable)  # 所有作者的单词初始设为1
        #     author_article[authors.index(author)] = 2  # 作者的文章个数初始设为 2
        # 构建先验概率


        for text, author in zip(self.train_data["text"], self.train_data["author"]):

            author_article[authors.index(author)] += 1
            text = re.split(regString, text)  # 利用正则表达式对句子分词
            for word in text:
                author_wordTable[authors.index(author)][self.wordTable.index(word)] += 1
        authorVec = np.log(author_wordTable / author_article)  #  3 * N
        pp = np.sum(author_article / len(self.train_data["text"]))  # 3 * 1
        return authorVec, pp

    def predict(self, data, pVec, nVec, pp):
        # data 是一个字符串：0
        dlist = re.split(regString, data)
        # 构建数据的词表：
        dvec = np.zeros((1,len(self.wordTable))) # 1 * N
        for word in dlist:
            if word in self.wordTable:
                dvec[self.wordTable.index(word)] += 1
        # 将author_wordTable与devc相乘
        authorVec, pp = self.authorVec, self.pp
        predcit = np.sum(np.matmul(authorVec,dvec.transpose()) + np.log(pp)) #  3 * 1

def ltest():
    na = NaiveBaye("../data/en_train.csv")
    print(na.wordTable)


if __name__ == "__main__":
    # read_data()
    ltest()
