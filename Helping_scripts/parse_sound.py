import csv
import sys,time


file=sys.argv[1]
s=open('/home/shailja/Parsed_Sound.csv','wb')
f=open(file,'rb')
data1=""
s.write("time"+"value"+"\n")
data=csv.reader(f,delimiter=',')
t=0
for d in data:
	i=1
	print t
	#print len(d)
	while i<len(d):
		
		if d=='\n':
			break
		#print d[i]
		val=i/4000
		data1=data1+str(int(d[0])+float(val))+","+str(d[i])+"\n"
		i+=1
		
	
	t+=1

	if t==90:
		s.write(data1)
		t=0	
		data1=""


s.close()
f.close()


