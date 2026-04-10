import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
import scipy

duration = 2600
startrawsignal = 400
startnoise = 400
record = pd.read_csv("data/csv/PulseTransitTime/s1_sit.csv")
data = record.normECG
samplerate = 500
cutoff_frequency = 40.0
order = 1
coeffs = scipy.signal.butter(order, cutoff_frequency / (0.5 * samplerate), btype="low", analog=False)
data = scipy.signal.lfilter(coeffs[0], coeffs[1], data)
data = data[startrawsignal:startrawsignal+duration]

t = np.arange(0, 5.2, 1 / samplerate)
amplitude_mean = 0.08
amplitude_std = 0.05
max_amplitude = 0.5
random_factors = np.random.exponential(scale=amplitude_mean, size=len(t))
random_factors = np.clip(random_factors, 0, max_amplitude)
noise50hz = 0.5*random_factors * np.sin(2 * np.pi * 60 * t)
data50hz = data + noise50hz
record_name = "data/raw/MIT_NSTDB/bw"
record = wfdb.rdrecord(record_name)
bwdata = record.to_dataframe()
shift = 43000
noisebw = bwdata.noise2[shift+startnoise:shift+startnoise+duration]
noisebw = noisebw.reset_index(drop=True)
databw = data + 1.5 * noisebw
record_name = "data/raw/MIT_NSTDB/ma"
record = wfdb.rdrecord(record_name)
madata = record.to_dataframe()
shift = 19999
noisema = madata.noise2[shift+startnoise:shift+startnoise+duration]
noisema = noisema.reset_index(drop=True)
datama = data + 2.5 * noisema
record_name = "data/raw/MIT_NSTDB/em"
record = wfdb.rdrecord(record_name)
emdata = record.to_dataframe()
delay = 7500
noiseem = emdata.noise2[delay+startnoise:delay+startnoise+duration]
noiseem = noiseem.reset_index(drop=True)
dataem = data + 2 * noiseem
data_a = data
data_b = data50hz
data_c = databw
data_d = datama
data_e = dataem
data_f = data + noise50hz + 1 * noisebw + 2.5 * noisema + 2 * noiseem
yaxis = 2

t = np.linspace(0, 5.2, duration, endpoint=False)

fig, axs = plt.subplots(6, 1, figsize=(10, 12))


axs[0].set_ylim(-yaxis, yaxis)
axs[0].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[0].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[0].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[0].plot(t, data_a, color="black")
axs[0].set_xticks(np.arange(0, 5.25, 0.5))  # Sekundenschritte
axs[0].text(-0.1, 0.5, "a)", transform=axs[0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[0].annotate("PTT Record s1_sit, [:40]Hz-filtered, [-1:1]-normalized", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


axs[1].set_ylim(-yaxis, yaxis)
axs[1].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[1].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[1].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[1].plot(t, data_b, color="black")
axs[1].set_xticks(np.arange(0, 5.25, 0.5))
axs[1].text(-0.1, 0.5, "b)", transform=axs[1].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[1].annotate("60Hz Powerline Interference", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


axs[2].set_ylim(-yaxis, yaxis)
axs[2].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[2].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[2].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[2].plot(t, data_c, color="black")
axs[2].set_xticks(np.arange(0, 5.25, 0.5))
axs[2].text(-0.1, 0.5, "c)", transform=axs[2].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[2].annotate("Baseline Wandering", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


axs[3].set_ylim(-yaxis, yaxis)
axs[3].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[3].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[3].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[3].plot(t, data_d, color="black")
axs[3].set_xticks(np.arange(0, 5.25, 0.5))
axs[3].text(-0.1, 0.5, "d)", transform=axs[3].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[3].annotate("Motion Artifacts", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


axs[4].set_ylim(-yaxis, yaxis)
axs[4].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[4].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[4].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[4].plot(t, data_e, color="black")
axs[4].set_xticks(np.arange(0, 5.25, 0.5))
axs[4].text(-0.1, 0.5, "e)", transform=axs[4].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[4].annotate("Electromagnetic Interference", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


axs[5].set_ylim(-yaxis, yaxis)
axs[5].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[5].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[5].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
axs[5].plot(t, data_f, color="black")
axs[5].set_xticks(np.arange(0, 5.25, 0.5))
axs[5].text(-0.1, 0.5, "f)", transform=axs[5].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[5].annotate("Cumulated Noise", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")
axs[5].set_xlabel("Time", fontweight="bold")

from datetime import datetime, timedelta

# Define the start time of the signal
start_time = datetime.strptime("00:00:00.8", "%H:%M:%S.%f")

# Generate time labels for each second, starting at 00:00:01
time_labels = [start_time + timedelta(seconds=i) for i in range(1, 6)]  # Start at 1 second
formatted_labels = [time.strftime("%H:%M:%S") for time in time_labels]  # Format as HH:MM:SS

# Update the grid and x-axis labels for each subplot
for ax in axs:
    ax.set_xticks(np.arange(0.2, 5.2, 1))  # Tick positions start at 0.2 (corresponds to 00:00:01) and increment by 1 second
    ax.set_xticklabels(formatted_labels)  # Apply formatted labels
    ax.grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange", axis="both")  # Ensure grid is active

plt.subplots_adjust(hspace=0.2)
plt.savefig('plots/noisetypes.png', bbox_inches='tight', dpi=300)
plt.show()