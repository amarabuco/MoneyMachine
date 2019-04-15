from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

# Create your views here.
def index(request):
    return HttpResponse("<h1>Application</h1>")

def aplicacao(request):
    return HttpResponse("<h1>$$$ Show me the money, baby $$$</h1>")