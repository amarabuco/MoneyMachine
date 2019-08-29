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
    path('candle', views.candle, name='candle'), 
    path('analysis', views.analysis, name='analysis'),
    path('describe', views.describe, name='describe'),
    path('training/<str:model>', views.training, name='training'),
    path('training_log/<str:model>', views.training_log, name='training_log'),
    path('forecast/<str:model>', views.forecast, name='forecast'),
    path('previsao/<str:model>', views.previsao, name='previsao'),
    path('previsao', views.previsao, name='previsao'),
    path('modelo', views.modelo, name='modelo'),
    path('Aplicação', views.filtro, name='filtro'),
    #path('api', views.api, name='api'),
    path('<x>/api/', views.api, name='api'),
]
