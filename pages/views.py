from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .models import Question,Answer,QuestionType

# Create your views here.

def index(request):
    return HttpResponse("Projeto Money Machine.")

def desafio(request):
    return page(request,1)

def solucao(request):
    return page(request,2)

def projeto(request):
    return page(request,3)

def aplicacao(request):
    return page(request,4)

def page(request,x):
    page_list = QuestionType.objects.all()
    question_list = Question.objects.filter(question_type=x).order_by('id')
    question_list_id = Question.objects.filter(question_type=x).values_list('id')
    answer_list = Answer.objects.filter(question__in=question_list_id)   
    template = loader.get_template('pages/index.html')
    context = {
        'page_list': page_list,
        'question_list': question_list,
        'answer_list': answer_list,
        'question_list_id': question_list_id,
    }
    return HttpResponse(template.render(context, request))
