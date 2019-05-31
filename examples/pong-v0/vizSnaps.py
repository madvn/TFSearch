import numpy as np
import matplotlib.pyplot as plt
import time

plt.figure()
for i in range(100):
	print i
	o = np.load('./envSnaps/envSnap_'+str(i)+'.npy')
	#plt.cla()
	plt.imshow(o)
	plt.show()
	#time.sleep(.5)
