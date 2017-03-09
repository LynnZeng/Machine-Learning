
# coding: utf-8

# In[9]:

import csv
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


Xtrn = pd.read_csv("X-trn-200.csv",header=None)
Ytrn = pd.read_csv("Y-trn-200.csv",header=None)
Xtst = pd.read_csv("X-tst-200.csv",header=None)
Ytst = pd.read_csv("Y-tst-200.csv",header=None)
TrnAc_Entr=list(range(2,16))
TstAc_Entr=list(range(2,16))
TrnAc_Gini=list(range(2,16))
TstAc_Gini=list(range(2,16))
for i in range(2,16):
    clf1 = DecisionTreeClassifier(criterion='entropy',max_depth=i,random_state=0)
    clf1 = clf1.fit(Xtrn,Ytrn)
    TrnAc_Entr[i-2]=clf1.score(Xtrn,Ytrn)
    TstAc_Entr[i-2]=clf1.score(Xtst,Ytst)
    clf2 = DecisionTreeClassifier(criterion='gini',max_depth=i,random_state=0)
    clf2 = clf2.fit(Xtrn,Ytrn)
    TrnAc_Gini[i-2]=clf2.score(Xtrn,Ytrn)
    TstAc_Gini[i-2]=clf2.score(Xtst,Ytst)


x=range(2,16)
plt.plot(x,TrnAc_Entr,'b',x,TstAc_Entr,'r')
plt.ylabel('accuracy')
plt.xlabel('max tree depth')
plt.title('(200.csv-Entropy)Blue:traning set,Red:test set')
plt.show()

plt.plot(x,TrnAc_Gini,'b',x,TstAc_Gini,'r')
plt.ylabel('accuracy')
plt.xlabel('max tree depth')
plt.title('(200.csv-Gini)Blue:traning set,Red:test set')
plt.show()

