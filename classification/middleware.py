from django.shortcuts import render,redirect
class userCheckmiddleware(object):
    '''user登录确定 中间件'''
    def process_view(self,request,view_func, *view_args, **view_kwargs):

        # 判断用户是否登录
        if request.session.has_key("is_login"):
            return view_func(request, *view_args, **view_kwargs)
        else:
            return redirect('/user/login')