
from django.conf.urls import include, url
from user import views
urlpatterns = [
    url(r'^login$', views.login),  # 用户登录界面
    url(r'^login_check$', views.login_check), # 登录确认
]
