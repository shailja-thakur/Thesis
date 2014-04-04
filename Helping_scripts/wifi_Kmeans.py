from scipy.cluster.vq import *
import pandas as pd
from numpy import *
import matplotlib.pylab as plt
from matplotlib.pyplot import *
import csv
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

c1=[a['rms'],b['rms'],c['rms'],d['rms'],e['rms']]

features=vstack((r1,r2,r3,r4,r5))
print "Features-----", features

centroids,variance=kmeans(features,5)
print "\n Centroids", centroids
print "\n Variance",variance
code,distance=vq(features,centroids)
print "\n Code",code
print "\n Distance",distance

print "Number: A", len(a)
print "Code A", list(code).count(0)

print "Total records", len(a) + len(b) + len(c) + len(d) + len(e)
print "Total CODE:", len(code)
print code[:10]

true_label = a.label.tolist() + b.label.tolist() + c.label.tolist() + d.label.tolist() + e.label.tolist()

print confusion_matrix(true_label, code)

classes=['room1','room2','room3','room4','room5']
cls_report = (classification_report(true_label,
                                       code.tolist(),
                                       labels=['0','1','2','3','4'], target_names=classes))
print cls_report

a=features[code==0]
b=features[code==1]
c=features[code==2]
d=features[code==3]
e=features[code==4]

figure(1)
plt.xlabel("mean")
plt.ylabel("rms")

plt.scatter(a[:,0],a[:,1],c = 'r',label='room1')
plt.scatter(b[:,0],b[:,1],c = 'g',label='room2')
plt.scatter(c[:,0],c[:,1],marker='x',c = 'y',label='room3')
plt.scatter(d[:,0],d[:,1],marker='x',label='room4')
plt.scatter(e[:,0],e[:,1],c = 'black',label='room5')
#plt.scatter(centroids[:,0],centroids[:,1],s = 80,c = 'brown', marker = 's',label='centroid')
legend(loc='upper left')

figure(2)
plt.boxplot(c1)
plt.xticks([1,2,3,4,5],('room1','room2','room3','room4','room5'),rotation=15)
plt.ylabel('RMS -> WIfi signal strength')

plt.show()





