from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('desafio', views.desafio, name='desafio'),
    path('Desafio', views.desafio, name='desafio'),
    path('solucao', views.solucao, name='solucao'),
    path('Solução', views.solucao, name='solucao'),
    path('projeto', views.projeto, name='projeto'),
    path('Projeto', views.projeto, name='projeto'),
    path('aplicacao', views.aplicacao, name='aplicacao'),
    path('Aplicação', views.aplicacao, name='aplicacao'),
]
