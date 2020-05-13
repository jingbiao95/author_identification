from classification.methods.dt import dt
from classification.methods.svm import svm
from classification.methods.knn import knn
from classification.methods.rnn import rnn, Parameters_rnn, data_processing_rnn  # RNN 依赖
from classification.methods.cnn import data_processing, Parameters, cnn
from classification.methods.nb import nb
from classification.methods.lr import lr
from classification.methods.xgb import xgb
import pandas as pd
import numpy as np
import os
import pickle
from text_classification.settings import MEDIA_ROOT, CHECKPOINTS,TENSORBOARD_DIR


class classification():


    def author_classification(self, method, dataSet, parameters,textData):
        """
        通用的分类器
        :param method:
        :param dataSet:
        :param parameters:
        :return:
        """
        if method == "dt":
            data = dtMethod(dataSet, parameters,textData)
        elif method == "knn":
            '''k近邻'''
            data = knnMethod(dataSet, parameters,textData)
        elif method == "nb":
            '''朴素贝叶斯'''
            data = nbMethod(dataSet, parameters,textData)
        elif method == "svm":
            '''svm支持向量机'''
            data = svmMethod(dataSet, parameters,textData)
        elif method=="lr":
            data = lrMethod(dataSet,parameters,textData)
        elif method == "cnn":
            data = cnnMethod(dataSet, parameters,textData)
        elif method == "rnn":
            data = rnnMethod(dataSet, parameters,textData)
        elif method == "xgb":
            data = xgbMethod(dataSet, parameters,textData)

        return data

    def author_classification_predict(self, method, dataSet, textData, parameters):
        """
        测试专用函数
        :param method:
        :param dataSet:
        :param textData:
        :return:
        """

        pm = Parameters_rnn.Parameters()  # 加载参数
        pm.builtPM(dataSet, parameters)
        print("start_predict---------------------------------------------------")
        categories, cat_to_id = data_processing.read_category(dataSet.categories)  # 标签列表化
        wordid = data_processing.get_wordid(pm.vocab_filename)
        # pm.vocab_size = len(wordid)  # 修正词的大小
        # pm.pre_trianing = data_processing.get_word2vec(pm.vector_word_npz)  # 填充词向量

        # model = cnn.TextCnn(pm)
        model = rnn.TextRnn(pm)

        # rnn.train(model, pm, wordid, cat_to_id, dataSet.id)
        pre_label = rnn.val_text(model, textData, pm, wordid, cat_to_id, dataSet.id) # "体育"

        data = {
            "predit_label": categories[pre_label],
            "image_url": rnn.getImageUrl(dataSet.id, pm)}
        return data


def getFilePath(dataSet):
    '''
    构建train 训练数据, test 测试数据, val 验证数据 的文件路径
    '''
    train_path = os.path.join(MEDIA_ROOT, dataSet.train.path)  # 训练数据集
    test_path = os.path.join(MEDIA_ROOT, dataSet.test.path)
    val_path = os.path.join(MEDIA_ROOT, dataSet.val.path)
    return train_path, test_path, val_path

def xgbMethod(dataSet, parameters,textData):
    train_path, test_path, val_path = getFilePath(dataSet)
    
    model_path = os.path.join(TENSORBOARD_DIR, "xgb",
                              "xgb_" + "_dataSet_" + str(dataSet.id)   + "_max_depth_" + parameters["max_depth"] + "_n_estimators_" +
                              parameters["n_estimators"]+"_colsample_bylevel_"+parameters["colsample_bylevel"]+"_subsample_"+parameters["subsample"]+"_nthread_"+parameters["nthread"]+"_learning_rate_"+parameters["learning_rate"])
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # 加载模型
    else:
        model = xgb.XGBOOST(train_path, "",int(parameters["max_depth"]),int( parameters["n_estimators"]), float(parameters["colsample_bylevel"]),
                                 float(parameters["subsample"]),int(parameters["nthread"]),float(parameters["learning_rate"]))
        with open(model_path, "wb") as f:
            pickle.dump(model, f)  # 写模型

    metric = getEvaluation(model.yvalid, model.valid_predictions)
    predict = xgb.predict_text(model, textData)[0]  # 预测数据
    reData = {"text_class": predict, "metric": metric}

def dtMethod(dataSet, parameters,textData):
    # 决策树处理
    # 填装parameters
    train_path, test_path, val_path = getFilePath(dataSet)

    model_path = os.path.join(TENSORBOARD_DIR, "dt",
                              "dt_" + "_dataSet_" + str(dataSet.id) +"_criterion_"+parameters["criterion"]+"_max_depth_"+parameters["max_depth"]+"_max_features_"+parameters["max_features"])
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # 加载模型
    else:
        model = dt.DT(train_path,"",parameters["criterion"],parameters["max_depth"],parameters["max_features"])
        model.visualization(dataSet.id,parameters["criterion"],parameters["max_depth"],parameters["max_features"])
        with open(model_path, "wb") as f:
            pickle.dump(model, f)  # 写模型
    metric = getEvaluation(model.yvalid, model.valid_predictions)
    predict = dt.predict_text(model, textData)[0]# 预测数据
    reData = {"text_class": predict, "metric": metric,"image_url":dt.getImageUrl(dataSet.id,parameters["criterion"],parameters["max_depth"],parameters["max_features"])}

    # real = decision.lbl_enc.transform(test_data.author.values)  # 真实值
    # data = {"accuracy_score": decision.metrics(real, predict)[0], "precision_score": decision.metrics(real, predict)[1],
    #         "recall_score": decision.metrics(real, predict)[2], "f1_score": decision.metrics(real, predict)[3], }
    return reData


def cnnMethod(dataSet, parameters,textData):
    pm = Parameters.Parameters()  # 加载参数
    pm.builtPM(dataSet, parameters) #填充参数
    
    # 读取categories
    categories, cat_to_id = data_processing.read_category(dataSet.categories) #读取数据的类和类ID
    wordid = data_processing_rnn.get_wordid(pm.vocab_filename) # 词ID
    print("--------------------------数据准备结束--------------------------------------------------------------")
    model = cnn.TextCnn(pm)  # 获得cnn模型
    checkpoint_path = os.path.join(CHECKPOINTS, 'text_cnn', os.path.normpath(cnn.make_dir_string(dataSet.id, pm)))
    if not os.path.exists(checkpoint_path):
        # 文件夹不存在-->代表未训练该模型了
        cnn.train(model, pm, wordid, cat_to_id, dataSet.id)  # 保存到本地
    # ------------------训练，预测分界线-----------------------------
    print("start_predict---------------------------------------------------")

    # pre_label, label = cnn.val(model, pm, wordid, cat_to_id, dataSet.id)
    pre_label = cnn.val_text(model, textData, pm, wordid, cat_to_id, dataSet.id)
    # correct = np.equal(pre_label, np.argmax(label, 1))
    # accuracy = np.mean(np.cast['float32'](correct))
    # print('accuracy:', accuracy)
    # print(pre_label[:10])
    # print(np.argmax(label, 1)[:10])
    # data = {"accuracy_score": float(accuracy)}
    data = {
        "predit_label": categories[pre_label],
        "image_url": cnn.getImageUrl(dataSet.id, pm),
    }
    return data


def rnnMethod(dataSet,parameters,textData):
    pm = Parameters_rnn.Parameters()  # 加载参数
    pm.builtPM(dataSet, parameters)
    
    # 读取categories
    categories, cat_to_id = data_processing_rnn.read_category(dataSet.categories)
    wordid = data_processing_rnn.get_wordid(pm.vocab_filename)
    print("----------------------------------------------------------------------------------------")
    model = rnn.TextRnn(pm)  # 获得rnn模型

    if not os.path.exists(os.path.join(CHECKPOINTS, 'text_rnn', os.path.normpath(rnn.make_dir_string(dataSet.id, pm)))):
        # 文件夹不存在-->代表未训练该模型了
        rnn.train(model, pm, pm.vocab_size, cat_to_id, dataSet.id)  # 保存到本地
        # ------------------训练，预测分界线-----------------------------
    print("start_predict---------------------------------------------------")

    # pre_label, label = rnn.val(model, pm, wordid, cat_to_id, dataSet.id)
    # correct = np.equal(pre_label, np.argmax(label, 1))
    # accuracy = np.mean(np.cast['float32'](correct))
    # print('accuracy:', accuracy)
    # print(pre_label[:10])
    # print(np.argmax(label, 1)[:10])
    # data = {"accuracy_score": float(accuracy)}

    pre_label = rnn.val_text(model, textData, pm, wordid, cat_to_id, dataSet.id)
    data = {
        "predit_label": categories[pre_label],
        "image_url": rnn.getImageUrl(dataSet.id, pm)}
    return data

def lrMethod(dataSet,parameters,textData):
    trpath, tepath, worpath = getFilePath(dataSet)
    # 判断是否已经训练好了svm
    model_path = os.path.join(TENSORBOARD_DIR, "lr",
                                  "lr_" + "_dataSet_" + str(dataSet.id)+"_penalty_"+parameters["penalty"]+"_solver_"+parameters["solver"] +"_max_iter_"+parameters["max_iter"])
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # 加载模型
    else:
        model = lr.LR(trpath, tepath, parameters["penalty"],parameters["solver"], parameters["max_iter"])
        with open(model_path, "wb") as f:
            pickle.dump(model, f)  # 写模型
    metric = getEvaluation(model.yvalid, model.valid_predictions)
    predict = svm.predict_text(model, textData)
    reData = {"text_class": predict, "metric": metric}
    return reData

def svmMethod(dataSet, parameters,textData):
    trpath, tepath, worpath = getFilePath(dataSet)
    
    # 判断是否已经训练好了svm
    model_path  = os.path.join(TENSORBOARD_DIR,"svm","svm_"+"_dataSet_"+str(dataSet.id)+"_kernel_function_"+parameters["kernel_function"])
    if os.path.exists(model_path):
        with open(model_path,"rb") as f:
            model =pickle.load(f)  #加载模型
    else:
        model = svm.SVM(trpath, tepath=tepath, kernel=parameters["kernel_function"])
        with open(model_path,"wb") as f:
            pickle.dump(model,f)  #写模型
    metric = getEvaluation(model.yvalid, model.valid_predictions)
    predict = svm.predict_text(model,textData)
    reData = {"text_class":predict,"metric":metric}
    return reData


def knnMethod(dataSet, parameters,textData):
    train_path, test_path, val_path = getFilePath(dataSet)
    
    # 判断是否已经训练好了svm
    model_path = os.path.join(TENSORBOARD_DIR, "knn",
                                  "knn_" + "_dataSet_" + str(dataSet.id) + "_n_neighbors_" + parameters["n_neighbors"]+"_weights_" + parameters["weights"]+"_algorithm_" + parameters["algorithm"]+"_p_" + parameters["p"])
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # 加载模型
    else:
        model = knn.KNN(train_path, test_path, int(parameters["n_neighbors"]), parameters["weights"], parameters["algorithm"],
                            int(parameters["p"]) )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)  # 写模型
    metric = getEvaluation(model.yvalid, model.valid_predictions)
    predict = knn.predict_text(model, textData)
    reData = {"text_class": predict, "metric": metric}
    return  reData


def nbMethod(dataSet, parameters,textData):
    train_path, test_path, val_path = getFilePath(dataSet)
    
    # 判断是否已经训练好了svm
    model_path = os.path.join(TENSORBOARD_DIR, "nb",
                                  "nb_" + "_dataSet_" + str(dataSet.id) + "_type_" + parameters[
                                      "type"])
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # 加载模型
    else:
        model = nb.NB(train_path,parameters["type"])
        with open(model_path, "wb") as f:
            pickle.dump(model, f)  # 写模型
    predict = nb.predict_text(model, textData)
    metric = getEvaluation(model.yvalid, model.valid_predictions)
    reData = {"text_class": predict, "metric": metric}
    return reData

from sklearn.metrics import accuracy_score,average_precision_score,log_loss,classification_report
def getEvaluation(yvalid,predict):
    acc_score = accuracy_score(y_true=yvalid, y_pred=predict)
    # avg_precision_score = average_precision_score(y_true= yvalid,y_pred = predict) #平均准确率
    # log_loss(yvalid, predict) #对数损失（Log-loss）
    report = classification_report(y_true=yvalid, y_pred=predict)#精确率(Precision)
    reData = {"acc_socre":acc_score,"report":report}
    return  reData


clf = classification()