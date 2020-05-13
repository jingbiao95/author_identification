from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
import json
from  text_classification.settings import BASE_DIR,MEDIA_ROOT
from  classification.action import clf
import  os
from  classification.models import DataSet
import traceback


def login_require(view_func):
    '''登录判断装饰器'''
    def wrapper(request,*view_args,**view_kwargs):
        # 判断用户是否登录
        if request.session.has_key("is_login"):
            return view_func(request, *view_args, **view_kwargs)
        else:
            print("用户未登录")
            return redirect('/user/login')
    return wrapper



@login_require
def index(request):
    '''文本分类平台主页面'''
    data={}
    return render(request, "classification/index.html", data)

@login_require
def algorithm_select(request):
    '''返回具体的算法调参页面'''
    method = request.POST.get("algroithm")
    c = DataSet.objects.all()
    # for s in c:
    #     s
    data={"dataSets":DataSet.objects.all()}
    return render(request,'classification/'+method+'.html',data)

@login_require
def algorithm_helps(request):
    '''返回具体的算法介绍页面'''
    method = request.POST.get("helps")
    return render(request,'helps/'+method+'.html')


@login_require
def algorithm_submit(request):
    '''获取前台发来的信息'''
    data = json.loads(request.body.decode())

    try:
        method = data["method"]
        # id = data["dataSet"]
        textType = data["textType"] #数据类型
        textData = data["text_data"] #预测的文本内容
        parameters = data["parameters"]
        if textType =="zh":
            dataSet =DataSet.objects.get(id=1)
        else:
            dataSet = DataSet.objects.get(id=2)
        retData = clf.author_classification(method, dataSet, parameters,textData)
        retData = {"success:": 1, 'resultData': retData}
    except Exception as e:
        print(e)
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():',traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        retData = {"success":0}

    return HttpResponse(json.dumps(retData, ensure_ascii=False), content_type="application/json, charset=utf-8")
    # return JsonResponse(json.dumps(retData),safe=False)

@login_require
def algorithm_predict(request):
    '''输入文本，进行预测属于那种新闻'''
    '''数据格式：
    '''
    data = json.loads(request.body.decode())
    try:
        method = data["method"]
        # model = data["model"]
        text_data = data["text_data"]
        parameters = data["parameters"]
        id = 1
        dataSet =DataSet.objects.get(id=id)
        retData = clf.author_classification_predict(method, dataSet, text_data,parameters)
        retData = {"success:": 1, 'predict': retData}

    except Exception as e:
        print(e)
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():',traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        retData = {"success":0}

    return HttpResponse(json.dumps(retData,ensure_ascii=False),content_type="application/json, charset=utf-8")
    # return JsonResponse(json.dumps(retData,ensure_ascii=False), safe=False)

