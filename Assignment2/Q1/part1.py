import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from playsound import playsound

def decide_MAFP():
	return 20

def filter_using_MAF(Samples):
	L=decide_MAFP()
	n=len(Samples)
	Modified_Samples=[0 for i in range(n)]
	for i in range(n):
		counter=0
		for j in range(i,max(0,i-L+1)-1,-1):
			Modified_Samples[i]+=Samples[j]
			counter+=1
		Modified_Samples[i]=int(round(Modified_Samples[i]/counter))

	return np.array(Modified_Samples)

a = wavfile.read('test_noise.wav')
Sampling_Rate=a[0]
Samples=a[1]
MSamples=filter_using_MAF(Samples)

####### PLOTTING AND SAVING the sound ##############
time = np.linspace(0., 4, 64000)
#plot 1:
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.subplot(1, 2, 1)
plt.plot(time, Samples)
plt.title("Original Noisy Audio")
plt.ylim(-30000, 30000)
#plot 2:
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.subplot(1, 2, 2)
plt.plot(time, MSamples)
plt.title("Modified Audio")
plt.ylim(-30000, 30000)
plt.savefig('PART1')
plt.show()

###### Saving and Playing sounds ##########
wavfile.write("PART1.wav", Sampling_Rate, MSamples.astype(np.int16))
print('Playing Modified sound ...')
playsound('PART1.wav')

print("\nENDS!\nThe Plot has been saved as `PART1.png` and the Audio as `PART1.wav`. \nTHANK YOU :) ")