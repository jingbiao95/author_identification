from django.db import models

class User(models.Model):
    username = models.CharField(max_length=20) # 用户名

    password = models.CharField(max_length=20) # 密码
# Create your models here.
