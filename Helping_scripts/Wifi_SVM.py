import math
import csv
from sklearn.svm import SVC
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
import pandas as pd
from numpy import *


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
a=pd.read_csv('room1.csv')
b=pd.read_csv('room2.csv')
c=pd.read_csv('room3.csv')
d=pd.read_csv('room4.csv')
e=pd.read_csv('room5.csv')

r1=a[['rms','mean']]
r2=b[['rms','mean']]
r3=c[['rms','mean']]
r4=d[['rms','mean']]
r5=e[['rms','mean']]
features=vstack((r1,r2,r3,r4,r5))
#r_train=[r1,r2,r3,r4,r5]
#print r_train[:2]
test_label=c['label']
#test_label.append(d['label'])

#r_test=[r1]
#print features

c1=a['label']
c1.append(b['label'])
c1.append(c['label'])
#c1=vstack((a['label'],b['label'],c['label'],d['label'],e['label']))
c1= np.array(c1)

concat([a.ix[:,['rms','mean']],b.ix[:,['rms','mean']]],axis=1,join='inner')
r1.append(r2)
r1.append(r3)


r1=np.array(r1)


print r1

clf=SVC()
clf.fit(r1,c1)
result=np.array(clf.predict(np.array(r3)))

print result
c=test_label==result
print len(test_label)
print len(result)

######..................................................ACCURACY CALCULATION..............................................######

print "Accuracy svm                    : ",str(sum(c)*100.0/len(result))+" %"


#####..............................PLOT classifier and predicted results with actual value................................#######
plt.plot(test_label,'r',label="accelerometer unit")

plt.plot(result,'g')


plt.show()



