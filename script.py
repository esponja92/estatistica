# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------------------
TESTE DAS HIPÓTESES (WIKIPEDIA):

1) Versão aceita (teoria do Iceberg)
2) Teoria da Troca
3) Afundado propositadamente

"Alguns teóricos da conspiração acreditam que o Titanic foi afundado de propósito para eliminar a oposição à criação do Banco da Reserva Federal. Alguns dos homens mais ricos do mundo estavam a bordo do Titanic para sua viagem inaugural, vários dos quais, incluindo John Jacob Astor IV, Benjamin Guggenheim e Isidor Straus, eram supostamente contra à criação de um banco central americano. Todos os três homens morreram durante o naufrágio."
Nesse caso, e partindo do princípio que o plano foi bem sucedido, são informações importantes para a predição da sobrevivência "Pclass", "Sex_cleaned" e "Fare", pois são as características importantes para demarcar os "homens mais ricos do mundo". A performance média deste modelo é dada por:

Media:  76.9088785047
de:  74.5287826871
ate:  79.2889743223

Além disso, este modelo classifica os nomes citados como sobreviventes. Logo, por absurdo, esta teoria é estatísticamente improvável

4) Portas estanques fechadas
5) Teoria das juntas de expansão
6) Incêndio na carvoaria 

------------------------------------------------------------------------------------------
BONS MODELOS:

Este primeiro modelo traz os fatores de maior relevância encontrados para classificar a sobrevivência ou não dos passageiros.
A máxima "mulheres e crianças primeiro" torna o dado "sex" importante, pois ter prioridade na rotina de emergência
confere teoricamente alguma vantagem. Da mesma forma, uma pessoa que estivesse viajando sozinha ou acompanhada também
parece fazer sentido, uma vez que a preocupação com os seus pode trazer desvantagens como lentidão na evasão, ao passo
que uma pessoa independente simplesmente se lançaria sem restrições aos botes salva vidas. Neste modelo, Jack Downson não sobrevive.

1)

used_features =[
    "Sex_cleaned",
    "SibSp",
    "Parch"
]

Media:  78.3836448598
de:  75.9748618092
ate:  80.7924279104


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from scipy.stats import norm
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing dataset
data = pd.read_csv("./train.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                  np.where(data["Embarked"]=="C",1,
                                           np.where(data["Embarked"]=="Q",2,3)
                                          )
                                 )
# Cleaning dataset of NaN
data=data[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis=0, how='any')

used_features =[
    "Pclass",
    "Sex_cleaned",
#    "Age",
#    "SibSp",
#    "Parch",
    "Fare",
    "Embarked_cleaned"
]


# Instantiate the classifier
gnb = GaussianNB()

x = []

print "Calculando as performances do modelo...\n"
for i in range(1000):

    # Split dataset in training and test datasets
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=random.randrange(1000000000))

    # Train classifier
    gnb.fit(
        X_train[used_features].values,
        X_train["Survived"]
    )

    y_pred = gnb.predict(X_test[used_features])

    x.append(float(100*(1-(X_test["Survived"] != y_pred).sum()/float(X_test.shape[0]))))

x.sort()
mean = np.mean(x)
std = np.std(x)
variance = np.square(std)

print "Media: ",mean
print "Desvio padrão: ",std
print "de: ",(mean-std)
print "ate: ",mean+std

plt.plot(x, norm.pdf(x,mean,std))

plt.show()

#testando teoria 3

data = pd.read_csv("./teoria3.csv")

# Convert categorical variable to numeric
data["Sex_cleaned"]=np.where(data["Sex"]=="male",0,1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0,
                                  np.where(data["Embarked"]=="C",1,
                                           np.where(data["Embarked"]=="Q",2,3)
                                          )
                                 )
print "\n---------- TESTE DA TEORIA 3 --------------\n"
preds = gnb.predict(data[used_features])
for i in range(len(data["Name"])):
    if preds[i]:
        print data["Name"][i]," - Sobrevive"
    else:
        print data["Name"][i]," - Não sobrevive"


