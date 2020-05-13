
import os
from text_classification.settings import MEDIA_ROOT, CHECKPOINTS,TENSORBOARD_DIR
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def getFilePath(dataSet):
    '''
    构建train 训练数据, test 测试数据, val 验证数据 的文件路径
    '''
    train_path = os.path.join(MEDIA_ROOT, dataSet.train.path)  # 训练数据集
    test_path = os.path.join(MEDIA_ROOT, dataSet.test.path)
    # val_path = os.path.join(MEDIA_ROOT, dataSet.val.path)
    return train_path, test_path


# 获取文件数据
def getSource(trpath,tepath):

    train = pd.read_csv(trpath)
    test = pd.read_csv(tepath)

    lbl_enc = preprocessing.LabelEncoder()  # y的标签
    y = lbl_enc.fit_transform(train.author.values)

    return train, test
