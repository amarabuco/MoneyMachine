from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.template import loader
from django.utils.safestring import SafeString

#Data Wrangling
import pandas as pd
import numpy as np

#Visualization
import matplotlib
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc

#Machine Learning
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

#python
import pickle
from joblib import dump, load


from .forms import Filtro, Previsao

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
            'title':'Filtro de ação',        
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
        'data' : data,
        'acao' : acao
    }
    
    return render(request, 'app/data.html', context )
    #return HttpResponse(data)


def chart(request):
    acao = request.session['acao'] 
    #bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    bolsa = pd.read_csv("app/data/bolsa.csv").groupby('Codigo')
    dados = bolsa.get_group(acao)
    date = pd.Series(dados.index)
    date = date.apply(lambda x: pd.to_datetime(x))
    open = dados['Open']
    close = dados['Close']
    high = dados['High']
    low = dados['Low']
    vol = dados['Volume']
    ohlc= dados[['Date','Open', 'High', 'Low','Close']].copy()
    #ohlc =ohlc.tail(60).to_html()
    ohlc =ohlc.tail(60)
    """
    f1, ax = plt.subplots(figsize = (10,5))
    candlestick_ohlc(ax, ohlc, width=.6, colorup='white', colordown='black')
    plt.grid(True)
    plt.savefig("media/daily.png")
    """
   
   
    plt.figure(figsize=(10,10))
    plt.xlabel("Data")
    plt.ylabel("Price")
    plt.title(acao)
    plt.plot(close)
    plt.plot(high)
    plt.plot(low)
    plt.grid(True)
    plt.savefig("media/daily.png")
    
    
    plt.figure(figsize=(10,10))
    plt.xlabel("Data")
    plt.ylabel("Vol")
    plt.title('Volume')
    plt.bar(vol,height=(vol/vol.mean()))
    plt.grid(True)
    plt.savefig("media/daily_vol.png")
    
    
    context = {
        'ohlc': ohlc,
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

def describe(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    data = dados.describe().to_html()
    
    context = {
        'title' : 'Descrição',
        'data' : data
    }
    return render(request, 'app/data.html', context )

def training(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    X = dados[['Open','High', 'Low','Close','Volume']]
    y = dados['High'].shift(-1).fillna(method='pad')
    Y = pd.DataFrame({'H+1':dados['High'].shift(-1).fillna(method='pad'),'L+1':dados['Low'].shift(-1).fillna(method='pad')})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
    X_train, X_test, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)
    base = dados.to_html()
    
    #training
    regr = linear_model.BayesianRidge()
    regr.fit(X_train,y_train)    
    
    #trainingmulti
    regr_multi = MultiOutputRegressor(linear_model.BayesianRidge())
    regr_multi.fit(X_train,Ytrain)
    Y_PRED = regr_multi.predict(X_test)
    real = pd.DataFrame(Ytest)
    previsto = pd.DataFrame(Y_PRED, index=Ytest.index, columns=['Alta','Baixa'])
    #real.rename(columns={"High": "real"})
    #previsto = previsto.set_index(real.index)
    data = pd.concat([real,previsto],axis=1)
    #data['diferenca'] = data['High']-data['previsto']
    #erro = np.array(data['diferenca'])
    data = data.to_html()
    #data = previsto.head().to_html()
        
    """
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
    """
    mae = mean_absolute_error(Ytest, Y_PRED)
    mse = mean_squared_error(Ytest, Y_PRED)
    ev = explained_variance_score(Ytest, Y_PRED, multioutput='uniform_average')
    r2 = r2_score(Ytest, Y_PRED)
    
    """
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
    
    """
    #persistence
    dump(regr_multi, 'app/learners/'+acao+'_NB.joblib')
    
    
    context = {
        'title' : 'Treino Regressão',
        'mae': mae,
        'mse': mse,
        'ev': ev,
        'r2': r2,
        'base': base,
        'data' : data,
        'acao' : acao,
        'multi' : Y_PRED[0]
    }
    return render(request, 'app/training.html', context )

def forecast(request):
    acao = request.session['acao']
    """
    regr = load('app/learners/'+acao+'_NB.joblib')
    #abertura = 100
    #open = request.POST['Abertura']
    input = pd.DataFrame({'Open':19.30,'High':19.58,'Low':19.24,'Close':19.50,'Volume':3164300}, index=['2019-08-13'])
    y_pred = regr.predict(input)
    data = y_pred[0]
    title = 'Previsão'
    context = {
        'title' : title,
        'data' : data,
        #'open' : open,
        'acao' : acao
        }
    """ 
    try:
        regr = load('app/learners/'+acao+'_NB.joblib')
        #abertura = 100
        #open = request.POST['Abertura']
        open = request['open']
        input = pd.DataFrame({'Open':19.30,'High':19.58,'Low':19.24,'Close':19.50,'Volume':3164300}, index=['2019-08-13'])
        y_pred = regr.predict(input)
        data = y_pred[0]
        title = 'Previsão'
        context = {
            'title' : title,
            'data' : data,
            'open' : open,
            'acao' : acao
            }
    except:
        title = "É preciso treinar o algoritmo, antes de fazer a previsão"
        data = "-"
        context = {
        'title' : title,
        'data' : data,
        'acao' : acao
        }
     
    
    return render(request, 'app/forecast.html', context )

def training_log(request):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    dados['var'] = dados['Close'].pct_change()*100
    dados['move'] = [1 if x>0.01 else -1 if x<-0.01 else 0 for x in dados['Close'].pct_change()]
    #dados['movecolor'] = [SafeString('<div style=\"color:green\">1</div>') if x>0.01 else SafeString("<div class=\"btn btn-danger\">-1</div>") if x<-0.01 else SafeString("<div class=\"btn btn-default\">0</div>") for x in dados['Close'].pct_change()]
    X = dados[['Open','High', 'Low','Close','Volume']]
    y = dados['move']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
    base = dados.head().to_html()
    
       
    #training
    regr = linear_model.LogisticRegression(random_state=0, solver='saga',multi_class='multinomial')
    #regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    #regr = linear_model.LogisticRegression(random_state=0)
    #regr = LinearSVC(C=1.0)
    #regr = KNeighborsClassifier(n_neighbors=9)
    regr.fit(X_train,y_train)    
    
    #forecast
    y_pred = regr.predict(X_test)
    real = pd.DataFrame(y_test)
    previsto = pd.DataFrame(y_pred, index=real.index, columns=['previsto'])
    real.rename(columns={"move": "real"})
    #previsto = previsto.set_index(real.index)
    
    data = pd.concat([real,previsto],axis=1)
    data['diferenca'] = data['move']-data['previsto']
    erro = np.array(data['diferenca'])
    data = data.to_html()
    prob = pd.DataFrame(regr.predict_proba(X_test), index=real.index).to_html()
    #data = previsto.head().to_html()
    
    
    #metrics
    mae = accuracy_score(y_test, y_pred)
    mse = precision_score(y_test, y_pred, average='weighted')
    r2 = recall_score(y_test, y_pred, average='weighted')
    """
    mae = mean_absolute_error(Ytest, Y_PRED)
    mse = mean_squared_error(Ytest, Y_PRED)
    ev = explained_variance_score(Ytest, Y_PRED, multioutput='uniform_average')
    r2 = r2_score(Ytest, Y_PRED)
    
    """
    #chart
    
    plt.figure(figsize=(5,5))
    plt.xlabel("Data")
    plt.ylabel("High")
    plt.title(acao)
    #plt.plot(y_train)
    #plt.grid(True)
    plt.plot(previsto)
    #plt.plot(y_test,y_test[])
    plt.savefig("media/forecast.png")
    
    
    
    plt.figure(figsize=(5,5))
    plt.title('Histograma  - Erro')
    plt.grid(True)
    plt.hist(erro,bins=10)
    plt.savefig("media/hist.png")
    
    
    #persistence
    dump(regr, 'app/learners/'+acao+'_RL.joblib')
    
    context = {
        'title' : 'Treino Classificação',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'base': base,
        'data' : data,
        'prob': prob,
        'acao' : acao,
        'multi' : '-'
    }
    return render(request, 'app/training_log.html', context )


def forecastNB(acao,open,close,high,low,vol,data):
    try:
        regr = load('app/learners/'+acao+'_NB.joblib')
        input = pd.DataFrame({'Open':open,'High':high,'Low':low,'Close':close,'Volume':vol}, index=['2019-08-13'])
        y_pred = regr.predict(input)
        data = y_pred[0]
        title = 'Previsão'

    except:
        title = "É preciso treinar o algoritmo, antes de fazer a previsão"
        data = "-"
    
    return data

def previsao(request):
    acao = request.session['acao']
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = Previsao(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            data = forecastNB(acao,request.POST['Abertura'],request.POST['Fechamento'],request.POST['Alta'],request.POST['Baixa'],request.POST['Volume'],request.POST['Data'])
            if (data == '-' ):
                title = "É preciso treinar o algoritmo, antes de fazer a previsão"
            else:
                title = 'Previsão'
            context = {
                #'data' : data[0].index,
                'alta' : "{0:.2f}".format(data[0]),
                'baixa' : "{0:.2f}".format(data[1]),
                'amp' :  "{0:.2f}".format((data[0]/data[1]-1)*100),
                'title' : title
            }
            
            return render(request, 'app/forecast.html', context )
            #return render(request, 'app/filtro.html', context )
        

    # if a GET (or any other method) we'll create a blank form
    else:
         
        form = Previsao()
        
        context = {
            'title':'Dados para previsão',
            'form': form,
            'acao': acao
        }

    return render(request, 'app/form.html', context )