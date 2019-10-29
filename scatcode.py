from __future__ import print_function
import gwpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import argparse
from scipy.signal import find_peaks
from numpy import diff
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from gwpy.time import to_gps
from matplotlib import cm
from gwpy.table import EventTable


channel = 'L1:GDS-CALIB_STRAIN'

# global variables
TCURRENT = int(to_gps('now'))

# argument parser
parser = argparse.ArgumentParser(description= 'This program is used to determine characteristics of a scattering surfaces. Produces a Q Spectrogram of the gravitational wave strain and a 2D plot of Frequency vs Time from the Q Spectrogram data. From the scattering equation, f_max = 2/lambda * (x_scat)(f_scat), the velocity of the scattering surface can be determined. Where f_max is the average maximum frequency of the scattering arches, x_scat is the distance the surface vibrates away from the equilibrium position, and f_scat is the frequency of oscillation of the surface. The velocity of the surfave is (x_scat)*(f_scat). The average time interval between scattering arches and the number of arches in the event are also calculated. Written by Shania A. Nichols August 2019')
parser.add_argument('threshold', type=int,
                   help='Threshold value for max energies. For fast scattering use a value of 15 and for slow scattering use a value of 200. Threshold values may need to be adjusted depending on the specific scattering event.')
parser.add_argument('--gpsstart', type=int, default=TCURRENT-300, required = False,
                    help='GPS start time or datetime of analysis. Default: 5 minutes prior to current time')
parser.add_argument('--duration', type=int, default=60, required = False,
                    help='Duration of of analysis in seconds. Default: 60 seconds')

args = parser.parse_args()

gpsend = args.gpsstart + args.duration

#Timeseries Data
TS = TimeSeries.fetch(channel, args.gpsstart, gpsend)
specgram = TS.spectrogram(2, fftlength=1, overlap=.5) ** (1/2.)
normalised = specgram.ratio('median')

#Plot QSpectrogram
qspecgram = TS.q_transform(qrange=(4, 150), frange=(10,100), outseg=(args.gpsstart, gpsend), fres=.01)
plot = qspecgram.imshow(figsize=[8, 4])
cmap = cm.get_cmap('viridis')
ax = plot.gca()
ax.set_title('Q Transform')
ax.set_xscale('seconds')
ax.set_yscale('log')
ax.set_ylim(10, 100)
ax.set_ylabel('Frequency [Hz]')
ax.set_facecolor(cmap(0))
ax.grid(True, axis='y', which='both')
ax.colorbar(cmap='viridis', vmin=0.5, norm='log', label='Normalized energy')
plot.show(block=False)

map(max,qspecgram.value) #finds max values for qspecgram.values, one max per array which corrolates to a given time
max_energies = list(map(max, qspecgram.value)) #Returns list of max energy values

#returns index for the frequecy that occurs at each time step and the max energy at said time step
frequencies=[] #empty list for frequencies
for i in range(len(max_energies)): 
    inx = np.where(max_energies[i]==qspecgram[i,:])
    freq = qspecgram.frequencies.value[inx]
    frequencies.append(freq[0])

#plot = plt.plot(qspecgram.times.value, frequencies,'.')
#ax.set_yscale('log')

ind, = np.where(np.array(max_energies) > args.threshold)
data = EventTable([qspecgram.times.value[ind], np.array(frequencies)[ind]], names=('time', 'frequency'))

x = np.array(data) #Creates array of the data from the Event table
t = list(column[0] for column in x)
f = list(column[1] for column in x)
f = np.array(f)
t_f = np.vstack((t,f)).T #creates array of time and frequencies	

f_max_ind=scipy.signal.find_peaks(f)[0] #Gives an array of indicies of maximum frequences for each arch
ind=list(f_max_ind)
t_fmax=t_f[ind] #returns array of time and f max for each arch
t = list(column[0] for column in t_fmax)
f_max = list(column[1] for column in t_fmax)

plot = data.scatter('time', 'frequency', figsize=(8, 4))
ax = plot.gca()
ax.set_ylabel('Frequency [Hz]')
plt.plot(t, f_max, 's', color='red')
ax.grid(True)
ax.grid(which="both", axis='y')
ax.set_yscale('log')
ax.set_ylim(10,100)
ax.set_xlim([args.gpsstart,gpsend])
plot.show(block=False)


dt=np.ediff1d(t) #Creates and array that contains the differences in time between consecutive arches
dt= list(dt) #List the difference in time between arches
n = len(dt)
scat_arches= n+1

#Scaterring Equation 
print('f_max = 2/lambda * (x_scat)(f_scat)')
print('Number of Scattering Arches: %f' %(scat_arches))

#Find average time (seconds) between arches
def Average(dt):
	return sum(dt)/n
f_scat =1/(Average(dt)*2) #average frequency of motion [Hz]
dt_std= np.std(dt)

print('Average time interval between scattering arches: %f +/- %f s' %((Average(dt), (dt_std))))

print('f_scat: %f +/- %f Hz' %((f_scat), (dt_std)))

f_max= list(f_max)
x = len(f_max)

#Find average maximum frequency of arches
def Average(f_max):
	return sum(f_max)/x

f_maxstd=np.std(f_max) #Standard deviation of f_max
print('f_max: %f +/- %f Hz' %((Average(f_max), f_maxstd))) #[Hz]

l = 0.000001064

x_scat_std = (((f_maxstd/Average(f_max))**2 + (dt_std/(Average(dt)))**2)**(1/2))/100
x_scat = (l*Average(f_max))/(2*f_scat) #Scatering surface movement distance
print('x_scat: %f +/- %f m' %((x_scat), (x_scat_std)))

v_scat_std = (((x_scat_std/x_scat)**2 + (dt_std/Average(dt))**2)**(1/2))/100 

v_scat= x_scat*f_scat #velocity of scattering surface
print('Velocity of Scattering Surface: %f +/- %f ms-1' %((v_scat), (v_scat_std)))

plot.show()




	

