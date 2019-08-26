from django.urls import path

from . import views

urlpatterns = [
    path('', views.aplicacao, name='aplicacao'),
    path('index', views.index, name='index'),
    path('aplicacao', views.filtro, name='filtro'),
    path('api', views.api, name='api'),
    path('filtro', views.filtro, name='filtro'),
    path('menu', views.menu, name='menu'),
    path('data', views.data, name='data'),
    path('chart', views.chart, name='chart'), 
    path('analysis', views.analysis, name='analysis'),
    path('training', views.training, name='training'),
    path('forecast', views.forecast, name='forecast'),
    path('Aplicação', views.filtro, name='filtro'),
    #path('api', views.api, name='api'),
    path('<x>/api/', views.api, name='api'),
]
