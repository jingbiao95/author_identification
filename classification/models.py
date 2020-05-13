from django.db import models
import ast
import os
from text_classification.settings import MEDIA_ROOT,BASE_DIR
class ListField(models.TextField):
    __metaclass__ = models.SubfieldBase
    description = "Stores a python list"

    # to_python()   # 把数据库数据转成python数据
    #
    # from_db_value()  # 把数据库数据转成python数据
    #
    # get_pre_value()  # 把python数据压缩准备存入数据库
    #
    # get_db_pre_value()  # 把压缩好的数据转成数据库查询集
    #
    # get_prep_lookup()   # 指定过滤的条件
    #
    # value_to_string()   # 数据序列化
    #
    def __init__(self, *args, **kwargs):
        super(ListField, self).__init__(*args, **kwargs)

    def from_db_value(self,value,expression,connection,content):
        if not value:
            value = []
        if value.strip() == "":
            value = []
        if isinstance(value, list):
            return value

        return ast.literal_eval(value)

    def to_python(self, value):
        if not value:
            value = []

        if isinstance(value, list):
            return value

        return ast.literal_eval(value)

    def get_prep_value(self, value):
        if value is None:
            return value

        return str(value)  # use str(value) in Python 3

    def value_to_string(self, obj):
        value = self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)



# Create your models here.
class DataSet(models.Model):
    '''上传的数据集'''
    title = models.CharField(max_length=20)  # 文件名
    train = models.FileField(upload_to='data')  # 训练文件存储路径
    test = models.FileField(upload_to='data',blank=True) # 测试文件存储路径
    val = models.FileField(upload_to='data',blank=True) # 验证文件存储路径

    categories = ListField(blank=True,null=True)  #用来存放数据集的标签列

    vocab_filename = models.CharField(max_length=200,blank=True)  #有数据处理后存储   文件路径
    vector_word_filename = models.CharField(max_length=200,blank=True)
    vector_word_npz =models.CharField(max_length=200,blank=True)

    desc = models.TextField(null=True)  # 数据集描述
    is_hide = models.BooleanField(default=False)  # 是否可见 对user而言
    is_del = models.BooleanField(default=False)  # 是否删除


class DataSetModel(models.Model):
    '''用来存放DataSet进行过分类后的模型'''
    dataset = models.CharField(max_length=20) # 数据集
    method = models.CharField(max_length=20) # 分类方法
    title = models.CharField(max_length=50) # 模型名称
    model = models.FilePathField() # 模型名  通过超参数构建模型名
    is_del = models.BooleanField(default=False) # 是否删除


