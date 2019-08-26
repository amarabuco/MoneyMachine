from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader

#Data Wrangling
import pandas as pd
import numpy as np

#Visualization
import matplotlib
import matplotlib.pyplot as plt

#Machine Learning
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score, r2_score

#python
import pickle
from joblib import dump, load


from .forms import Filtro

# Create your views here.
def index(request):
    return HttpResponse("<h1>Application</h1>")

def aplicacao(request):
    return HttpResponseRedirect('/app/filtro')


def filtro(request):
    
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = Filtro(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            request.session['acao'] = request.POST['acao']

            return HttpResponseRedirect('/app/menu')
            #return render(request, 'app/filtro.html', context )
        

    # if a GET (or any other method) we'll create a blank form
    else:
        form = Filtro()
        
        context = {
            'form': form,
            'acao': 'Nenhuma'
        }

    return render(request, 'app/filtro.html', context )

def menu(request):
    acao = request.session['acao']
    context = {
        'acao': acao
    }
            
    return render(request, 'app/menu.html', context )
    
def api(request):
    stock = pd.read_csv("app/data/bolsa.csv", index_col='Date').head().to_html()
    bolsa = pd.read_csv("app/data/IBOV_2013-2019.csv", index_col='Date')
    
    """
    #stock = pd.read_csv("https://query1.finance.yahoo.com/v7/finance/download/VALE3.SA?period1=1552682299&period2=1555360699&interval=1d&events=history&crumb=aMJOzZ4One0")
    if (x == "VALE3SA_2014"):
        stock = pd.read_csv("app/data/VALE3SA_2014.csv").to_html()
    else:
    stock = pd.read_csv("app/data/VALE3SA_2000.csv").to_html()
    """
    bolsa = bolsa[['Codigo','Open','High', 'Low','Close','Volume']]
    bolsa['Open'] = bolsa[['Open']].apply(lambda x: x.str.replace(',','.')).apply(pd.to_numeric, errors='coerce')
    bolsa['High'] = bolsa[['High']].apply(lambda x: x.str.replace(',','.')).apply(pd.to_numeric, errors='coerce')
    bolsa['Low'] = bolsa[['Low']].apply(lambda x: x.str.replace(',','.')).apply(pd.to_numeric, errors='coerce')
    bolsa['Close'] = bolsa[['Close']].apply(lambda x: x.astype(str)).apply(lambda x: x.str.replace(',','.')).apply(pd.to_numeric, errors='coerce')
    date = pd.Series(bolsa.index)
    date = date.apply(lambda x: pd.to_datetime(x))
    bolsa = bolsa.set_index(date) 
    bolsa.to_csv('app/data/bolsa.csv')    
    """
    template = loader.get_template('app/data.html')
    context = {
        'stock' : stock,
    }
    """
    #return HttpResponse(template.render(context, request))
    return HttpResponse(stock)

def data(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    data = bolsa.get_group(acao).to_html()
    dados = bolsa.get_group(acao)
    #bolsa.index = bolsa.index.strftime('%Y-%m-%d')
    date = pd.Series(dados.index)
    date = date.apply(lambda x: pd.to_datetime(x))
    dados = dados.set_index(date) 
    data = dados.to_html()
    
    context = {
        'title' : 'Dados de mercado',
        'data' : data
    }
    
    return render(request, 'app/data.html', context )
    #return HttpResponse(data)


def chart(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    date = pd.Series(dados.index)
    date = date.apply(lambda x: pd.to_datetime(x))
    close = dados['Close']
    high = dados['High']
    low = dados['Low']
    vol = dados['Volume']

    plt.figure(figsize=(10,10))
    plt.grid(True)
    
    plt.figure(figsize=(10,10))
    plt.xlabel("Data")
    plt.ylabel("Price")
    plt.title(acao)
    plt.plot(close)
    plt.plot(high)
    plt.plot(low)
    plt.set_axis_bgcolor('white')
    plt.grid(True)
    plt.savefig("media/daily.png")
    
    """
    plt.figure(figsize=(10,10))
    plt.xlabel("Data")
    plt.ylabel("Vol")
    plt.title('Volume')
    plt.bar(vol,height=(vol/vol.mean()))
    plt.grid(True)
    plt.savefig("media/daily_vol.png")
    
    plt.subplot(2, 1, 1)
    plt.plot(close)
    plt.subplot(2, 2, 1)
    plt.plot(close)
    #plt.bar(vol,height=(vol/vol.mean()))
    plt.savefig("media/daily.png")
    """
    
    context = {
        'acao': acao
    }
   
    #return HttpResponse(buffer.getvalue(), mimetype="image/png")
    return render(request, 'app/chart.html', context )
    #return HttpResponse(data)


def analysis(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    dados['V-1'] = dados['Volume'].shift(1)
    dados['dH+1'] = dados['High'].shift(-1) - dados['High'] 
    dados['dL+1'] = dados['Low'].shift(-1) - dados['Low']
    dados['dC+1'] = dados['Close'].shift(-1) - dados['Close']
    dados['HxC'] = (dados['High']-dados['Close'])/(dados['High']-dados['Low'])
    dados['LxC'] = (dados['Close']-dados['Low'])/(dados['High']-dados['Low'])
    dados['OxC'] = (dados['Close']/dados['Open'])-1
    dados['V0xV+1'] = (dados['Volume']/dados['Volume'].shift(1))-1
    #analise = dados[['HxC','LxC','OxC','V0xV+1']]
    data = dados.head().to_html()
    
    context = {
        'title' : 'Variáveis analíticas',
        'data' : data
    }
    return render(request, 'app/data.html', context )

def training(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    X = dados[['Open','High', 'Low','Close','Volume']]
    y = dados['High'].shift(-1).fillna(method='pad') 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
    
    #training
    regr = linear_model.BayesianRidge()
    regr.fit(X_train,y_train)
    
    #forecast
    y_pred = regr.predict(X_test)
    real = pd.DataFrame(y_test)
    previsto = pd.DataFrame(y_pred, index=real.index, columns=['previsto'])
    #real.rename(columns={"High": "real"})
    #previsto = previsto.set_index(real.index)
    data = pd.concat([real,previsto],axis=1)
    data['diferenca'] = data['High']-data['previsto']
    erro = np.array(data['diferenca'])
    data = data.to_html()
    #data = previsto.head().to_html()
    
    #metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_pred)
    
    
    #chart
    
    plt.figure(figsize=(5,5))
    plt.xlabel("Data")
    plt.ylabel("High")
    plt.title(acao)
    #plt.plot(y_train)
    plt.grid(True)
    plt.plot(previsto)
    plt.plot(y_test)
    plt.savefig("media/forecast.png")
    
    
    
    plt.figure(figsize=(5,5))
    plt.title('Histograma  - Erro')
    plt.grid(True)
    plt.hist(erro,bins=10)
    plt.savefig("media/hist.png")
    
    #persistence
    dump(regr, 'app/learners/'+acao+'_NB.joblib')
    
    context = {
        'title' : 'Treino',
        'mae': mae,
        'mse': mse,
        'ev': ev,
        'r2': r2,
        'data' : data,
        'acao' : acao
    }
    return render(request, 'app/training.html', context )

def forecast(request):
    acao = request.session['acao']
    regr = load('app/learners/'+acao+'_NB.joblib')
    
    input = pd.DataFrame({'Open':19.30,'High':19.58,'Low':19.24,'Close':19.50,'Volume':3164300}, index=['2019-08-13'])
    y_pred = regr.predict(input)
    data = y_pred[0]
    
    context = {
        'title' : 'Previsão',
        'data' : data,
        'acao' : acao
    }
    
    return render(request, 'app/forecast.html', context )