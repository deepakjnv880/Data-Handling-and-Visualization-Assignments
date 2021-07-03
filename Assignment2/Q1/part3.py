import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from playsound import playsound
import math 

Number_of_Frame_for_Noise_Estimation=10
a = wavfile.read('test_noise.wav')
Sampling_Rate=a[0]
Samples=a[1]
Overlapping_Frame_Time=0.02
Overlapping_Frame_Size=int(Overlapping_Frame_Time*Sampling_Rate)
Overlapping_Percentage=50
Number_of_overlapping_Samples=int((Overlapping_Frame_Size*Overlapping_Percentage)/100)
shift=Overlapping_Frame_Size-Number_of_overlapping_Samples
Number_of_Frame=int(len(Samples)/shift)-1
K=0
Window=np.hanning(Overlapping_Frame_Size)
Magnitudes=[]
Phases=[]

for i in range(Number_of_Frame):
	Framed_Signal=Window*Samples[K:K+Overlapping_Frame_Size]
	# Framed_Signal=Samples[K:K+Overlapping_Frame_Size]
	Framed_Signal_DFT=np.fft.fft(Framed_Signal)
	Magnitudes.append(np.abs(Framed_Signal_DFT))
	Phases.append(np.angle(Framed_Signal_DFT))
	K=K+shift


Noise_Mag=np.mean(Magnitudes[:Number_of_Frame_for_Noise_Estimation],axis=0)

Restored_Frames=[]
for i in range(Number_of_Frame):
	temp=[]
	for j in range(len(Magnitudes[i])):
		Magnitudes[i][j]=Magnitudes[i][j]-Noise_Mag[j] if Magnitudes[i][j]-Noise_Mag[j]>=0 else 0
		temp.append(Magnitudes[i][j]*(np.cos(Phases[i][j]) + np.sin(Phases[i][j])*1j))
	Restored_Frames.append(np.fft.ifft(temp).real)

back=list(Restored_Frames[0])
K=shift

for i in range(1,Number_of_Frame):
	for j in range(Number_of_overlapping_Samples):
		back[K+j]=(back[K+j]+Restored_Frames[i][j])/2
	for j in range(Number_of_overlapping_Samples,Overlapping_Frame_Size):
		back.append(Restored_Frames[i][j])
	K=K+shift

back=np.array(back)

####### PLOTTING AND SAVING the sound ##############
time = np.linspace(0., 4, 64000)
#plot 1:
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.subplot(1, 2, 1)
plt.plot(time, Samples)
plt.title("Original Noisy Audio")
plt.ylim(-35000, 35000)
#plot 2:
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.subplot(1, 2, 2)
plt.plot(time, back)
plt.title("Modified Audio")
plt.ylim(-35000, 35000)
plt.savefig('PART3')
plt.show()

#### Saving and Playing sounds ##########
wavfile.write("PART3.wav", Sampling_Rate, back.astype(np.int16))
print('Playing Modified sound ...')
playsound('PART3.wav')

print("\nENDS!\nThe Plot has been saved as `PART3.png` and the Audio as `PART3.wav`. \nTHANK YOU :) ")
