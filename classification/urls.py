
from django.conf.urls import include, url
from classification import views
urlpatterns = [
    url(r'^index$', views.index),  # 主页面
    url(r'^algorithm_select',views.algorithm_select), # 跳转具体算法页面
    url(r'^algorithm_helps', views.algorithm_helps), # 跳转帮助文档
    url(r'^algorithm_submit', views.algorithm_submit), # 提交超参数页面
    url(r'^algorithm_predict', views.algorithm_predict), # 提交超参数页面


]
