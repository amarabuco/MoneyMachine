from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
import pandas as pd
import numpy as np

# Create your views here.
def index(request):
    return HttpResponse("<h1>Application</h1>")

def aplicacao(request):
    template = loader.get_template('app/index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))

""" def page(request,x):
    page_list = QuestionType.objects.all()
    question_list = Question.objects.filter(question_type=x).order_by('id')
    question_list_id = Question.objects.filter(question_type=x).values_list('id')
    answer_list = Answer.objects.filter(question__in=question_list_id)   
    template = loader.get_template('app/index.html')
    context = {
        'page_list': page_list,
        'question_list': question_list,
        'answer_list': answer_list,
        'question_list_id': question_list_id,
    }
    return HttpResponse(template.render(context, request))
 """

def api2(request):
    stock = pd.read_csv("app\data\VALE3SA.csv")
    x = stock.tail()
    return HttpResponse(x)
    
def api(request,x):
    #stock = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/VALE3.SA?period1=1552682299&period2=1555360699&interval=1d&events=history&crumb=aMJOzZ4One0")
    if (x == "VALE3SA_2014"):
        stock = pd.read_csv("app/data/VALE3SA_2014.csv").to_html()
    else:
        stock = pd.read_csv("app/data/VALE3SA_2000.csv").to_html()
    template = loader.get_template('app/data.html')
    context = {
        'stock' : stock,
    }
    return HttpResponse(template.render(context, request))
    