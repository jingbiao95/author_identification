from django.test import TestCase
import  pandas as pd
import matplotlib.pyplot as plt
from classification.models import  DataSet
from classification.action import classification
import os
# Create your tests here.


class ActionTestCase(TestCase):

    def setUp(self):
        DataSet.objects.create(id=1,title='cnews',train="data/cnews.train.txt",test="data/cnews.test.txt",val='data/cnews.val.txt',desc="adf")

    def actionTest(self):
        data = {
                    "method": "cnn",
                    "dataSet": "1",
                    "parameters": {
                        "depth": "5"
                    }
        }
        try:
            method = data["method"]
            id = data["dataSet"]
            parameters = data["parameters"]
            dataSet =DataSet.objects.get(id=id)

            retData = classification().author_classification(method, dataSet, parameters)

            retData = {"success":1,"data":retData}
        except Exception as e:
            print(e)
            retData = {"success":0}

def plot():
    filename = r"F:\text_classification\tensorboard\text_rnn\dataid_1_num_layers_2_hidden_dim_100_keep_prob_0.5_learning_rate_0.001_lr_decay_0.9_clip_5.0_num_epochs_3_batch_size_64\train_info.csv"
    data = pd.read_csv(filename)
    plt.figure(1)
    plt.title("accuracy")
    plt.plot(data["global_step"],data["accuracy"],"o-")
    plt.savefig(os.path.join("",""))
    plt.show()



if __name__ == '__main__':

    # ActionTestCase().actionTest()
    plot()