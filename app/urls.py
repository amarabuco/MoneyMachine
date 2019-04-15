from django.urls import path

from . import views

urlpatterns = [
    path('', views.aplicacao, name='aplicacao'),
    path('index', views.index, name='index'),
    path('aplicacao', views.aplicacao, name='aplicacao'),
    path('Aplicação', views.aplicacao, name='aplicacao'),
]
