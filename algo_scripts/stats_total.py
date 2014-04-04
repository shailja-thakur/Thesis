import subprocess
import os
import time
import csv
path = '/home/shailja/shailja.thakur90@gmail.com/app_profiling/'
cpucsv =  path + 'total_cpu.csv'
memcsv = path + 'total_memory.csv'
cpu_writer = csv.writer(open(cpucsv, 'w'))
mem_writer = csv.writer(open(memcsv, 'w'))
cpu_writer.writerow(['timestamp'] + ['ago_time_ms'] + ['cpu_percentage'])
mem_writer.writerow(['timestamp'] + ['mem_MB'])
os.chdir('/home/shailja/android-sdk-linux/platform-tools')

while True:
	cpu_per = ''
	delay = ''
	cpu_cmd = 'adb shell dumpsys cpuinfo'
	mem_cmd = 'adb shell dumpsys meminfo'
	
	c = subprocess.check_output(cpu_cmd.split())
	m = subprocess.check_output(mem_cmd.split())
	print m
	m = m.split('\r\n')
	c = c.split('\r\n')

	for line in c:

		
		if 'CPU usage' in line:
			print line
			delay = str(line.split()[5])[:-2]

			print 'delay ', delay

		if 'TOTAL' in line:

			print line
			cpu_per = str(line.split('%')[0])
			print time.time(), cpu_per
			
			print 'cpu percentage at ',time.time(), 'is', cpu_per

		cpu_writer.writerow([time.time()] + [delay] + [cpu_per] )

	for line in m:
		if 'Dalvik' in line:
			print line
			tot_mem_MB = line.split()[1] 
			print time.time(), tot_mem_MB
			mem_writer.writerow([time.time()] + [tot_mem_MB])

	time.sleep(1)


			

