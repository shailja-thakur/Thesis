import csv
import os
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
import pylab
d=pd.read_csv('/home/shailja/littleeye/EnergySense_10th')
print d['Time']

data=d[45:105]
data1=d[17:70]
date1=pd.Series([pd.to_datetime(date1) for date1 in data1['Time']])
date=pd.Series([pd.to_datetime(date) for date in data['Time']])

fig=pylab.figure(1)

ax=plt.subplot(211)

plt.plot(date,data['Power'],'b',label='App Total Power',linewidth=5)
#plt.legend(('App Total Power'),'upper center',shadow=True,fancybox=True)


#plt.legend(('Network Total Power'),'upper center',shadow=True,fancybox=True)
plot(date,data['System'],'red',label='Device Total Power',linewidth=5)

plt.legend(('App Total Power','Device Total Power'),'upper center',shadow=True,fancybox=True)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend

# see text.Text, lines.Line2D, and patches.Rectangle for more info on
# the settable properties of lines, text, and rectangles
frame.set_facecolor('0.80')      # set the frame face color to light gray
plt.setp(ltext, fontsize=20)    # the legend text fontsize
plt.setp(llines, linewidth=5.0)  

plt.tick_params(axis='both', which='major', labelsize=5)
plt.tick_params(axis='both', which='minor', labelsize=5)
#legend()
ylabel("Average Power(mW)",fontsize=20)
title("Power Consumed Without Upload and With Upload",fontsize=25)

subplot(212)
plot(date1,data1['Power'],'b',label='App Total Power',linewidth=5)
plot(date1,data1['DataPower'],'g',label='Network Total Power',linewidth=5)
plot(date1,data1['System'],'red',label='Device Total Power',linewidth=5)
legend()
plt.fill_between(date1,data1['DataPower'],0,color='green')
plt.tick_params(axis='both', which='major', labelsize=20)

plt.legend(('App Total Power','Network Total Power','Device Total Power'),'upper center',shadow=True,fancybox=True)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  # all the text.Text instance in the legend
llines = leg.get_lines()  # all the lines.Line2D instance in the legend
frame  = leg.get_frame()  # the patch.Rectangle instance surrounding the legend

# see text.Text, lines.Line2D, and patches.Rectangle for more info on
# the settable properties of lines, text, and rectangles
frame.set_facecolor('0.80')      # set the frame face color to light gray
plt.setp(ltext, fontsize=20)    # the legend text fontsize
plt.setp(llines, linewidth=5.0)  

ylabel("Average Power (mW)",fontsize=20)
xlabel("Time",fontsize=20)


fig=pylab.figure(2)

plot(date,data['CPU'],'red',linewidth=5)
ylabel("CPU(%)",fontsize=20) 
xlabel("Time",fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
title("CPU(%) usage",fontsize=25)

#legend()
show()


