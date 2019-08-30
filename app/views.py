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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
##regressão
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
#from sklearn.ensemble import VotingRegressor
##classificação
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

#python
import pickle
from joblib import dump, load


from .forms import *

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

def candle(request):
    acao = request.session['acao'] 
    #bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    bolsa = pd.read_csv("app/data/bolsa.csv").groupby('Codigo')
    dados = bolsa.get_group(acao)
    date = pd.Series(dados.index)
    #date = date.apply(lambda x: pd.to_datetime(x))
    date = dados['Date']
    open = dados['Open']
    close = dados['Close']
    high = dados['High']
    low = dados['Low']
    dados['index'] = dados.index
    #vol = preprocessing.MinMaxScaler().fit_transform(np.array([dados['Volume']]))
    #vol = preprocessing.MinMaxScaler().fit_transform([dados['Date'],dados['Volume']])
    vol = dados['Volume']
    ohlc= np.array(dados[['index','Open', 'High', 'Low','Close']])
    #ohlc =ohlc.tail(60).to_html()
    ohlc =ohlc
    
    
    f1, ax = plt.subplots(figsize = (15,10))
    candlestick_ohlc(ax, ohlc, width=5, colorup='blue', colordown='red')
    plt.grid(True)
    plt.savefig("media/candle.png")
    
    context = {
        'ohlc': ohlc,
        'acao': acao
    }
   
    return render(request, 'app/candle.html', context )

def chart(request):
    acao = request.session['acao'] 
    #bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    bolsa = pd.read_csv("app/data/bolsa.csv").groupby('Codigo')
    dados = bolsa.get_group(acao)
    date = pd.Series(dados.index)
    #date = date.apply(lambda x: pd.to_datetime(x))
    date = dados['Date']
    open = dados['Open']
    close = dados['Close']
    high = dados['High']
    low = dados['Low']
    dados['index'] = dados.index
    #vol = preprocessing.MinMaxScaler().fit_transform(np.array([dados['Volume']]))
    #vol = preprocessing.MinMaxScaler().fit_transform([dados['Date'],dados['Volume']])
    vol = dados['Volume']
    ohlc= np.array(dados[['index','Open', 'High', 'Low','Close']])
    #ohlc =ohlc.tail(60).to_html()
    ohlc =ohlc
    
    
    f1, ax = plt.subplots(figsize = (10,5))
    candlestick_ohlc(ax, ohlc, width=.6, colorup='white', colordown='black')
    plt.grid(True)
    plt.savefig("media/candle.png")
    
   
    plt.figure(figsize=(10,2.5))
    plt.xlabel("Data")
    plt.ylabel("Price")
    plt.title(acao)
    plt.plot(close)
    #plt.plot(high)
    #plt.plot(low)
    #plt.grid(True)
    plt.savefig("media/daily.png")
    
    
    plt.figure(figsize=(10,2.5))
    plt.xlabel("Data")
    plt.ylabel("Vol")
    plt.title('Volume')
    plt.plot(vol)
    #plt.grid(True)
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

def training(request,model):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    X = dados[['Open','High', 'Low','Close','Volume']]
    y = dados['High'].shift(-1).fillna(method='pad')
    Y = pd.DataFrame({'Alta_real':dados['High'].shift(-1).fillna(method='pad'),'Baixa_real':dados['Low'].shift(-1).fillna(method='pad')})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
    X_train, X_test, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)
    base = dados.to_html()
    
    #training
    regr = linear_model.BayesianRidge()
    regr.fit(X_train,y_train)    
    
    #trainingmulti
    if (model == 'adr'):
        modelo = "Automatic Relevance Determination Regression"
        #regr_multi = MultiOutputRegressor(svm.SVR())
        regr_multi = MultiOutputRegressor(linear_model.ARDRegression(compute_score=True))
    elif (model == 'ada'):
        modelo = "Ada Regressor"
        regr_multi = MultiOutputRegressor(AdaBoostRegressor(random_state=0, n_estimators=100))
    elif (model == 'GB'):
        modelo = "GradientBoostingRegressor"
        regr_multi =  MultiOutputRegressor(GradientBoostingRegressor(random_state=1, n_estimators=10))
    else:
        modelo = "LinerRegression com Bayesian Ridge" 
        regr_multi = MultiOutputRegressor(linear_model.BayesianRidge())
        
    """
    # import votingregressor não funciona, precisa atualizar o sklearn
    elif (model == 'VR'):
        modelo = "Voting Regressor com GradientBoostingRegressor, RandomForestRegressor, LinearRegression"
        reg1 =  MultiOutputRegressor(GradientBoostingRegressor(random_state=1, n_estimators=10))
        reg2 =  MultiOutputRegressor(RandomForestRegressor(random_state=1, n_estimators=10))
        reg3 =  MultiOutputRegressor(LinearRegression())
        regr_multi = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    """
    
    regr_multi.fit(X_train,Ytrain)
    Y_PRED = regr_multi.predict(X_test)
    real = pd.DataFrame(Ytest)
    previsto = pd.DataFrame(Y_PRED, index=Ytest.index, columns=['Alta_prevista','Baixa_prevista'])
    #real.rename(columns={"High": "real"})
    #previsto = previsto.set_index(real.index)
    data = pd.concat([real,previsto],axis=1)
    data['diferenca_alta'] = data['Alta_real']-data['Alta_prevista']
    data['diferenca_baixa'] = data['Baixa_real']-data['Baixa_prevista']
    erro = data['diferenca_alta']
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
    """
    
    #metrics
    mae = mean_absolute_error(Ytest, Y_PRED)
    mse = mean_squared_error(Ytest, Y_PRED)
    ev = explained_variance_score(Ytest, Y_PRED, multioutput='uniform_average')
    r2 = r2_score(Ytest, Y_PRED)
    
 
    #chart
    
    plt.figure(figsize=(5,5))
    plt.xlabel("Data")
    plt.ylabel("High")
    plt.title(acao)
    #plt.plot(y_train)
    plt.plot(Ytest['Alta_real'])
    plt.plot(previsto['Alta_prevista'])
    #plt.grid(True)
    plt.savefig("media/forecast_reg.png")
    
    
    
    plt.figure(figsize=(5,5))
    plt.title('Erro Alta (real - prevista)4')
    plt.grid(True)
    plt.hist(erro,bins=5)
    plt.savefig("media/hist_reg.png")
    
    #params
    params = regr.get_params()
    
    #persistence
    if (model == 'VR'):
        dump(regr_multi, 'app/learners/'+acao+'_VR.joblib')
    elif (model == 'GB'):
        dump(regr_multi, 'app/learners/'+acao+'_GB.joblib')
    elif (model == 'adr'):    
        dump(regr_multi, 'app/learners/'+acao+'_ADR.joblib')
    elif (model == 'ada'): 
        dump(regr_multi, 'app/learners/'+acao+'_ADAR.joblib')
    else:
        dump(regr_multi, 'app/learners/'+acao+'_NBR.joblib')
    
    
    context = {
        'title' : 'Treino Regressão',
        'mae': mae,
        'mse': mse,
        'ev': ev,
        'r2': r2,
        'base': base,
        'data' : data,
        'acao' : acao,
        'modelo': modelo,
        'params' : params,
        'multi' : Y_PRED[0]
    }
    return render(request, 'app/training.html', context )

def forecast(request,model):
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
        if (model == 'VR'):
            regr = load('app/learners/'+acao+'_VR.joblib')
        elif (model == 'GB'):
            regr = load('app/learners/'+acao+'_GB.joblib')
        elif (model == 'adr'):    
            regr = load('app/learners/'+acao+'_ADR.joblib')
        elif (model == 'ada'): 
            regr = load('app/learners/'+acao+'_ADAR.joblib')
        elif (model == 'nbr'):
            regr = load('app/learners/'+acao+'_NBR.joblib')
        elif (model == 'knn'):
            regr = load(regr, 'app/learners/'+acao+'_KNN.joblib')
        elif (model == 'svc'):    
            regr = load('app/learners/'+acao+'_SVC.joblib')
        elif (model == 'ada'): 
            regr = load('app/learners/'+acao+'_ADA.joblib')
        elif (model == 'vc'):
            regr = load('app/learners/'+acao+'_VC.joblib')
        elif (model == 'neural'):
            regr = load('app/learners/'+acao+'_neural.joblib')
        elif (model == 'nbc'):
            regr = load('app/learners/'+acao+'_NBC.joblib')
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

def training_log(request,model):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
    dados['var'] = dados['Close'].pct_change()*100
    dados['move'] = [1 if x>0.01 else -1 if x<-0.01 else 0 for x in dados['Close'].pct_change()]
    class_names = ['desce','neutro','sobe',]
    #dados['movecolor'] = [SafeString('<div style=\"color:green\">1</div>') if x>0.01 else SafeString("<div class=\"btn btn-danger\">-1</div>") if x<-0.01 else SafeString("<div class=\"btn btn-default\">0</div>") for x in dados['Close'].pct_change()]
    X = dados[['Open','High', 'Low','Close','Volume']]
    y = dados['move']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
    base = dados.head().to_html()
    
       
    #training
    if (model == 'knn'):
        modelo = "KNN"
        regr = KNeighborsClassifier(n_neighbors=5)
    elif (model == 'svc'):
        modelo = "Support Vector Classifier"    
        regr = svm.SVC(gamma='scale')
    elif (model == 'ada'):
        modelo = "Ada Classifier"    
        regr = AdaBoostClassifier(n_estimators=100)

    elif (model == 'vc'):
        modelo = "Voting Classifier com LogisticRegression, RandomForestClassifier, GaussianNB" 
        clf1 = KNeighborsClassifier(n_neighbors=5)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
        clf3 = GaussianNB()
        regr = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
    elif (model == 'neural'):
        modelo = 'Multi-layer Perceptron classifier'     
        #regr = MLPClassifier(solver='adam', learning_rate='invscaling', activation="tanh", alpha=1e-5, hidden_layer_sizes=(200,5), random_state=0)
        regr = MLPClassifier()
    else:
        modelo = "Gaussian Naive Bayes" 
        regr = GaussianNB()
        #regr = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    regr.fit(X_train,y_train)    
    
    #forecast
    y_pred = regr.predict(X_test)
    real = pd.DataFrame(y_test)
    previsto = pd.DataFrame(y_pred, index=real.index, columns=['previsto'])
    real.rename(columns={"move": "real"})
    #previsto = previsto.set_index(real.index)
    
    data = pd.concat([real,previsto],axis=1)
    data = data.to_html()
    if (model == 'svc' or model == 'vc' ):
        prob = "Não há resultados"
    else:
        prob = pd.DataFrame(regr.predict_proba(X_test), index=real.index, columns=class_names).to_html()
    #data = previsto.head().to_html()
    
    #params
    params = regr.get_params()
    
    #metrics
    mae = accuracy_score(y_test, y_pred)
    mse = precision_score(y_test, y_pred, average='weighted')
    r2 = recall_score(y_test, y_pred, average='weighted')
    

    #chart    
    plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Matriz de confusão - '+acao)
    plt.savefig("media/confusion.png")
    
    plt.figure(figsize=(5,5))
    sober, neutror, descer  = len(y_test[y_test == 1]), len(y_test[y_test == 0]), len(y_test[y_test == -1])
    sobe, neutro, desce = len(y_pred[y_pred == 1]), len(y_pred[y_pred == 0]), len(y_pred[y_pred == -1])
    #grafico = pd.DataFrame({'desce':desce,'neutro':neutro,'sobe':sobe}, index=['classes'], columns=class_names)
    grafico_real = [descer,neutror,sober]
    grafico = [desce,neutro,sobe]
    plt.xlabel("Classe - real x previsto")
    plt.ylabel("Qtd")
    plt.title(acao)
    #plt.plot(y_train)
    plt.bar([4,8,12],grafico_real,width=1, tick_label=class_names, color=['darkred','darkblue','darkgreen'])
    plt.bar([5,9,13],grafico,width=1, tick_label=class_names, color=['red','blue','green'])
    #plt.plot(previsto['Alta_prevista'])
    #plt.grid(True)
    plt.savefig("media/clf_count.png")
    #grafico = grafico.to_html()
    
    
    #persistence
    if (model == 'knn'):
        dump(regr, 'app/learners/'+acao+'_KNN.joblib')
    elif (model == 'svc'):    
        dump(regr, 'app/learners/'+acao+'_SVC.joblib')
    elif (model == 'ada'): 
        dump(regr, 'app/learners/'+acao+'_ADA.joblib')
    elif (model == 'vc'):
        dump(regr, 'app/learners/'+acao+'_VC.joblib')
    elif (model == 'neural'):
        dump(regr, 'app/learners/'+acao+'_neural.joblib')
    else:
        dump(regr, 'app/learners/'+acao+'_NBC.joblib')
    
    context = {
        'title' : 'Treino Classificação',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'base': base,
        'data' : data,
        'prob': prob,
        'acao' : acao,
        'grafico': grafico,
        'modelo': modelo,
        'params' : params,
        'multi' : '-'
    }
    return render(request, 'app/training_log.html', context )


def get_forecast(acao,date,open,close,high,low,vol,model):
    try:
        if (model == 'VR'):
            regr = load('app/learners/'+acao+'_VR.joblib')
        elif (model == 'GB'):
            regr = load('app/learners/'+acao+'_GB.joblib')
        elif (model == 'adr'):    
            regr = load('app/learners/'+acao+'_ADR.joblib')
        elif (model == 'ada'): 
            regr = load('app/learners/'+acao+'_ADAR.joblib')
        elif (model == 'nbr'):
            regr = load('app/learners/'+acao+'_NBR.joblib')
        elif (model == 'knn'):
            regr = load(regr, 'app/learners/'+acao+'_KNN.joblib')
        elif (model == 'svc'):    
            regr = load('app/learners/'+acao+'_SVC.joblib')
        elif (model == 'ada'): 
            regr = load('app/learners/'+acao+'_ADA.joblib')
        elif (model == 'vc'):
            regr = load('app/learners/'+acao+'_VC.joblib')
        elif (model == 'neural'):
            regr = load('app/learners/'+acao+'_neural.joblib')
        elif (model == 'nbc'):
            regr = load('app/learners/'+acao+'_NBC.joblib')
        input = pd.DataFrame({'Open':open,'High':high,'Low':low,'Close':close,'Volume':vol}, index=['2019-08-13'])
        y_pred = regr.predict(input)
        data = y_pred[0]

    except:
        data = "-"
    
    return data

def previsao(request,model):
    acao = request.session['acao']
    regressoes = ['VR','GB','adr','ada','nbr']
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        
        form = Previsao(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            data = get_forecast(acao,request.POST['Data'],request.POST['Abertura'],request.POST['Fechamento'],request.POST['Alta'],request.POST['Baixa'],request.POST['Volume'],model)
            if (data == '-' ):
                return HttpResponse("É preciso treinar o algoritmo, antes de fazer a previsão")
            else:
                title = 'Previsão'
                if (model in regressoes):
                    context = {
                        #'data' : data[0].index,
                        'model': model,
                        'alta' : "{0:.2f}".format(data[0]),
                        'baixa' : "{0:.2f}".format(data[1]),
                        'amp' :  "{0:.2f}".format((data[0]/data[1]-1)*100),
                        'title' : title,
                        'acao': acao,
                        'regressoes': regressoes
                    }
                else:
                    if (data == 1):
                        classe = ' <i class="fas fa-arrow-alt-circle-up"></i> Preço vai subir'
                    if (data == 0):
                        classe = '<i class="fas fa-arrow-alt-circle-right"></i> Preço ficará neutro'
                    else:
                        classe = '<i class="fas fa-arrow-alt-circle-down"></i> Preço vai cair'
                    context = {
                        #'data' : data,
                        'model': model,
                        'classe': classe, 
                        'title' : title,
                        'acao': acao,
                        'regressoes': regressoes
                    }
            
            return render(request, 'app/forecast.html', context )
            #return render(request, 'app/filtro.html', context )
        

    # if a GET (or any other method) we'll create a blank form
    else:
         
        form = Previsao()
        
        context = {
            'title':'Dados para previsão',
            'model': model,
            'form': form,
            'acao': acao
        }

    return render(request, 'app/form.html', context )

def modelo(request):
    acao = request.session['acao']
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = Modelo(request.POST)
        m = request.POST['modelo']
        return HttpResponse(m)

    # if a GET (or any other method) we'll create a blank form
    else:
         
        form = Modelo()
        
        context = {
            'title':'Dados para previsão',
            'form': form,
            'acao': acao
        }

    return render(request, 'app/form.html', context )


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def multir(request,model):
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    lista = ['B3SA3',        'BBDC4',        'BRAP4',        'BRFS3',        'BRKM5',        'BRML3',        'BTOW3',        'CCRO3',
        'CIEL3',        'CMIG4',        'CSAN3',        'CSNA3',        'CYRE3',        'ECOR3',        'EGIE3',        'ELET3',
        'ELET6',        'EMBR3',        'ENBR3',        'EQTL3',        'ESTC3',        'FLRY3',        'GGBR4',        'GOAU4',
        'GOLL4',       'HYPE3',        'IGTA3',        'KROT3',        'ITSA4',       'ITUB4',        'LAME4',        'LREN3',
        'MGLU3',       'MRFG3',        'MRVE3',        'MULT3',        'NATU3',       'PCAR4',        'PETR3',       'PETR4',
        'QUAL3',       'RADL3',        'RENT3',        'SANB11',        'SBSP3',       'TAEE11',        'TIMP3',        'UGPA3',
        'USIM5',        'VALE3',        'VIVT4',        'WEGE3']
    
    resultado = []
    for item in lista:
        bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
        dados = bolsa.get_group(item)
        X = dados[['Open','High', 'Low','Close','Volume']]
        y = pd.DataFrame({'Alta_real':dados['High'].shift(-1).fillna(method='pad'),'Baixa_real':dados['Low'].shift(-1).fillna(method='pad')})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
        if (model == 'adr'):
            modelo = "Automatic Relevance Determination Regression"
            #regr_multi = MultiOutputRegressor(svm.SVR())
            regr_multi = MultiOutputRegressor(linear_model.ARDRegression(compute_score=True))
        elif (model == 'ada'):
            modelo = "Ada Regressor"
            regr_multi = MultiOutputRegressor(AdaBoostRegressor(random_state=0, n_estimators=100))
        elif (model == 'GB'):
            modelo = "GradientBoostingRegressor"
            regr_multi =  MultiOutputRegressor(GradientBoostingRegressor(random_state=1, n_estimators=10))
        else:
            modelo = "LinerRegression com Bayesian Ridge" 
            regr_multi = MultiOutputRegressor(linear_model.BayesianRidge())
        regr_multi = regr_multi.fit(X_train, y_train)
        y_pred = regr_multi.predict(X_test)
        #print(item)
        #print(": ")
        #print(r2_score(y_test, y_pred))
        #print(item,": ", r2_score(y_test, y_pred))
        r = r2_score(y_test, y_pred)
        resultado.append([item,r])
    resultado_geral = pd.DataFrame(resultado).to_html()
    context = {
        'modelo':  modelo,
        'resultado': resultado_geral
    }
    return render(request, 'app/multi.html', context )
        
def multic(request,model):
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    lista = ['B3SA3',        'BBDC4',        'BRAP4',        'BRFS3',        'BRKM5',        'BRML3',        'BTOW3',        'CCRO3',
        'CIEL3',        'CMIG4',        'CSAN3',        'CSNA3',        'CYRE3',        'ECOR3',        'EGIE3',        'ELET3',
        'ELET6',        'EMBR3',        'ENBR3',        'EQTL3',        'ESTC3',        'FLRY3',        'GGBR4',        'GOAU4',
        'GOLL4',       'HYPE3',        'IGTA3',        'KROT3',        'ITSA4',       'ITUB4',        'LAME4',        'LREN3',
        'MGLU3',       'MRFG3',        'MRVE3',        'MULT3',        'NATU3',       'PCAR4',        'PETR3',       'PETR4',
        'QUAL3',       'RADL3',        'RENT3',        'SANB11',        'SBSP3',       'TAEE11',        'TIMP3',        'UGPA3',
        'USIM5',        'VALE3',        'VIVT4',        'WEGE3']
    
    resultado = []
    for item in lista:
        bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
        dados = bolsa.get_group(item)
        dados['var'] = dados['Close'].pct_change()*100
        dados['move'] = [1 if x>0.01 else -1 if x<-0.01 else 0 for x in dados['Close'].pct_change()]
        X = dados[['Open','High', 'Low','Close','Volume']]
        y = dados['move']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
        if (model == 'knn'):
            modelo = "KNN"
            regr = KNeighborsClassifier(n_neighbors=5)
        elif (model == 'svc'):
            modelo = "Support Vector Classifier"    
            regr = svm.SVC(gamma='scale')
        elif (model == 'ada'):
            modelo = "Ada Classifier"    
            #regr = AdaBoostClassifier(n_estimators=100)
            regr = AdaBoostClassifier(n_estimators= 100, base_estimator=DecisionTreeClassifier(max_depth=5),algorithm= 'SAMME', learning_rate= 1)
        elif (model == 'vc'):
            modelo = "Voting Classifier com LogisticRegression, RandomForestClassifier, GaussianNB" 
            clf1 = KNeighborsClassifier(n_neighbors=5)
            clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
            clf3 = GaussianNB()
            regr = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        elif (model == 'neural'):
            modelo = 'Multi-layer Perceptron classifier'     
            #regr = MLPClassifier(solver='adam', learning_rate='invscaling', activation="tanh", alpha=1e-5, hidden_layer_sizes=(200,5), random_state=0)
            regr = MLPClassifier()
        else:
            modelo = "Gaussian Naive Bayes" 
            regr = GaussianNB()
        regr = regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        #print(item)
        #print(": ")
        #print(r2_score(y_test, y_pred))
        #print(item,": ", r2_score(y_test, y_pred))
        r = accuracy_score(y_test, y_pred)
        resultado.append([item,r])
        r = pd.DataFrame(resultado).to_html()
    context = {
        'modelo':  modelo,
        'resultado': r
    }
    return render(request, 'app/multi.html', context )

def gridsearch(request,model):
    acao = request.session['acao'] 
    bolsa = pd.read_csv("app/data/bolsa.csv", index_col='Date').groupby('Codigo')
    dados = bolsa.get_group(acao)
       
    #training
    if (model == 'ada'):
        dados['var'] = dados['Close'].pct_change()*100
        dados['move'] = [1 if x>0.01 else -1 if x<-0.01 else 0 for x in dados['Close'].pct_change()]
        X = dados[['Open','High', 'Low','Close','Volume']]
        y = dados['move']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
        base = dados.head().to_html()
        modelo = "Ada Classifier"    
        
        param_grid = [
            { 'base_estimator': [DecisionTreeClassifier(max_depth=5)], 'algorithm':['SAMME.R','SAMME'], 'learning_rate':[0.5, 1] , 'n_estimators': [75, 100, 125]}
            #{'n_estimators': [50, 100, 150, 200], 'base_estimator': [DecisionTreeClassifier(max_depth=1),DecisionTreeClassifier(max_depth=3),DecisionTreeClassifier(max_depth=5)], 'learning_rate':[0.5, 1, 1.5, 2] , 'algorithm':['SAMME']}
            #{ 'n_estimators': [100], 'base_estimator': [DecisionTreeClassifier(max_depth=5)],'algorithm': 'SAMME', 'learning_rate': 1}
            ]
        
        ada = AdaBoostClassifier(n_estimators=100)
        regr = GridSearchCV(ada, param_grid)
        
        regr.fit(X_train,y_train)    
    
        #forecast
        y_pred = regr.predict(X_test)
        real = pd.DataFrame(y_test)
        previsto = pd.DataFrame(y_pred, index=real.index, columns=['previsto'])
        real.rename(columns={"move": "real"})
        
        data = pd.concat([real,previsto],axis=1)
        data = data.to_html()

        class_names = ['desce','neutro','sobe',]
        prob = pd.DataFrame(regr.predict_proba(X_test), index=real.index, columns=class_names).to_html()
        
        #params
        params = [regr.cv_results_['params'], regr.cv_results_['mean_test_score'], regr.cv_results_['rank_test_score']]
        
        #metrics
        mae = accuracy_score(y_test, y_pred)
        mse = precision_score(y_test, y_pred, average='weighted')
        r2 = recall_score(y_test, y_pred, average='weighted')
        
        context = {
        'title' : 'Treino Classificação',
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'base': base,
        'data' : data,
        'prob': prob,
        'acao' : acao,
        'modelo': modelo,
        'params' : params,
        'multi' : '-'
        }
        return render(request, 'app/training_log.html', context )
    else:
        
        X = dados[['Open','High', 'Low','Close','Volume']]
        y = dados['High'].shift(-1).fillna(method='pad')
        Y = pd.DataFrame({'Alta_real':dados['High'].shift(-1).fillna(method='pad'),'Baixa_real':dados['Low'].shift(-1).fillna(method='pad')})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=0)
        X_train, X_test, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, shuffle=False, random_state=0)
        modelo = "LinerRegression com Bayesian Ridge" 
        regr = MultiOutputRegressor(linear_model.BayesianRidge())
        
        param_grid = [
            {'estimator':[linear_model.BayesianRidge(n_iter= 100,  compute_score=True, fit_intercept=False, alpha_1= 1.e-6),linear_model.BayesianRidge( n_iter= 200, compute_score=True, fit_intercept=False, alpha_1= 1.e-3),
                           linear_model.BayesianRidge(n_iter= 100, alpha_1= 1.e-1),linear_model.BayesianRidge( n_iter= 200, alpha_1= 1.e-9)]}
            #{'estimator':[linear_model.BayesianRidge(n_iter= 100, tol=0.001),linear_model.BayesianRidge( n_iter= 200, tol=0.001)]}
            #{ 'n_iter': [200,300,400,500], 'compute_score':(True, False), 'tol': [0.001,1.e-3]}
            #{ 'n_iter': [200,300,400,500], 'compute_score':['True','False']}
            ]
        
        regr_multi = GridSearchCV(regr, param_grid)
        
        regr_multi.fit(X_train,Ytrain)
        Y_PRED = regr_multi.predict(X_test)
        real = pd.DataFrame(Ytest)
        previsto = pd.DataFrame(Y_PRED, index=Ytest.index, columns=['Alta_prevista','Baixa_prevista'])
        data = pd.concat([real,previsto],axis=1)
        data['diferenca_alta'] = data['Alta_real']-data['Alta_prevista']
        data['diferenca_baixa'] = data['Baixa_real']-data['Baixa_prevista']
        erro = data['diferenca_alta']
        data = data.to_html()
    
        #params
        #params = regr.get_params()
        params = [regr_multi.cv_results_['params'], regr_multi.cv_results_['mean_test_score']]
    
        #metrics
        mae = mean_absolute_error(Ytest, Y_PRED)
        mse = mean_squared_error(Ytest, Y_PRED)
        ev = explained_variance_score(Ytest, Y_PRED, multioutput='uniform_average')
        r2 = r2_score(Ytest, Y_PRED)
 
        context = {
            'title' : 'Treino Regressão',
            'mae': mae,
            'mse': mse,
            'ev': ev,
            'r2': r2,
            #'base': base,
            'data' : data,
            'acao' : acao,
            'modelo': modelo,
            'params' : params,
            'multi' : Y_PRED[0]
        }
        return render(request, 'app/training.html', context )