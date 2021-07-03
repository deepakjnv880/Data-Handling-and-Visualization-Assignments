import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from playsound import playsound

a = wavfile.read('test_noise.wav')
Sampling_Rate=a[0]
Samples=a[1]

n=len(Samples)
sampling_length = n/Sampling_Rate
Fs = 1.0/sampling_length
ls = range(len(Samples))
freq = np.fft.fftfreq(len(Samples), d = sampling_length)
fft = np.fft.fft(Samples)
for i in range(len(freq)):
	if abs(freq[i]) > 0.05:
		fft[i] = 0.0

back = np.fft.ifft(fft).real

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
plt.plot(time, back)
plt.title("Modified Audio")
plt.ylim(-30000, 30000)
plt.savefig('PART2')
plt.show()

###### Saving and Playing sounds ##########
wavfile.write("PART2.wav", Sampling_Rate, back.astype(np.int16))
print('Playing Modified sound ...')
playsound('PART2.wav')

print("\nENDS!\nThe Plot has been saved as `PART2.png` and the Audio as `PART2.wav`. \nTHANK YOU :) ")