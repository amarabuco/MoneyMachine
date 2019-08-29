from django import forms

class Modelo(forms.Form):
    modelo = forms.ChoiceField(label='modelo', choices=(
        ('NB','Regressão - Naive Bayes'),('RL','Classificação - Regressão logística'),('KNN','Classificação -  KNN')
    ))
                             

class Previsao(forms.Form):
    Data = forms.DateField()
    Abertura = forms.DecimalField(max_digits=7,decimal_places=2)
    Alta = forms.DecimalField(max_digits=7,decimal_places=2)
    Baixa = forms.DecimalField(max_digits=7,decimal_places=2)
    Fechamento = forms.DecimalField(max_digits=7,decimal_places=2)
    Volume = forms.DecimalField(max_digits=12)

class Filtro(forms.Form):
    acao = forms.ChoiceField(label='Ação', choices=(
        ('B3SA3','B3SA3'),
        ('BBDC4','BBDC4'),
        ('BRAP4','BRAP4'),
        ('BRFS3','BRFS3'),
        ('BRKM5','BRKM5'),
        ('BRML3','BRML3'),
        ('BTOW3','BTOW3'),
        ('CCRO3','CCRO3'),
        ('CIEL3','CIEL3'),
        ('CMIG4','CMIG4'),
        ('CSAN3','CSAN3'),
        ('CSNA3','CSNA3'),
        ('CYRE3','CYRE3'),
        ('ECOR3','ECOR3'),
        ('EGIE3','EGIE3'),
        ('ELET3','ELET3'),
        ('ELET6','ELET6'),
        ('EMBR3','EMBR3'),
        ('ENBR3','ENBR3'),
        ('EQTL3','EQTL3'),
        ('ESTC3','ESTC3'),
        ('FLRY3','FLRY3'),
        ('GGBR4','GGBR4'),
        ('GOAU4','GOAU4'),
        ('GOLL4','GOLL4'),
        ('HYPE3','HYPE3'),
        ('IGTA3','IGTA3'),
        ('KROT3','KROT3'),
        ('ITSA4','ITSA4'),
        ('ITUB4','ITUB4'),
        ('LAME4','LAME4'),
        ('LREN3','LREN3'),
        ('MGLU3','MGLU3'),
        ('MRFG3','MRFG3'),
        ('MRVE3','MRVE3'),
        ('MULT3','MULT3'),
        ('NATU3','NATU3'),
        ('PCAR4','PCAR4'),
        ('PETR3','PETR3'),
        ('PETR4','PETR4'),
        ('QUAL3','QUAL3'),
        ('RADL3','RADL3'),
        ('RENT3','RENT3'),
        ('SANB11','SANB11'),
        ('SBSP3','SBSP3'),
        ('TAEE11','TAEE11'),
        ('TIMP3','TIMP3'),
        ('UGPA3','UGPA3'),
        ('USIM5','USIM5'),
        ('VALE3','VALE3'),
        ('VIVT4','VIVT4'),
        ('WEGE3','WEGE3')
        ))
