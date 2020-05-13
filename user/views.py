from django.shortcuts import render,redirect
from user.models import User


def login_require(view_func):
    '''登录判断装饰器'''
    def wrapper(request,*view_args,**view_kwargs):
        # 判断用户是否登录
        if request.session.has_key("is_login"):
            return redirect('/classification/index')
        else:
            return view_func(request, *view_args, **view_kwargs)
    return wrapper

#登录界面 即首页
@login_require
def login(request):
    # 提供登录界面
    data={}
    return render(request, 'user/login.html', data)


#ajax 请求,确认登录是否成功
def login_check(request):
    # 查询
    uName = request.POST.get("username")
    try:
        user = User.objects.get(username = uName)
    except Exception as e:
        # 没有找到则直接返回
        print(e)
        return redirect("/user/login")
    # print(user.password)
    # print(request.POST.get("pwd"))
    if user.password == request.POST.get("pwd"):
        # 记住用户登录状态
        request.session['is_login'] = True
        return redirect('/classification/index')
    else:
        return redirect("/user/login")

