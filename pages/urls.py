from django.urls import path

from . import views

urlpatterns = [
    path('', views.desafio, name='desafio'),
    path('desafio', views.desafio, name='desafio'),
    path('Desafio', views.desafio, name='desafio'),
    path('solucao', views.solucao, name='solucao'),
    path('Solução', views.solucao, name='solucao'),
    path('projeto', views.projeto, name='projeto'),
    path('Projeto', views.projeto, name='projeto'),   
]
