import numpy as np
import serial
import time

waitTime = 0.1

signalLength = 24
signalTable=[261,330,392,392,330,261,294,330,349,294,330,392,261,261,330,392,392,330,261,330,349,392,247,261]
lengthTable=[1,1,2,1,1,2,1,1,1,1,1,1,2,1,1,2,1,1,2,1,1,1,1,4]

# output formatter
formatter1 = lambda x: "%d" % x
formatter2 = lambda x: "%d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime * 2)))
for data in signalTable:
  s.write(bytes(formatter1(data), 'UTF-8'))
  time.sleep(waitTime)
for data in lengthTable:
  s.write(bytes(formatter2(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")