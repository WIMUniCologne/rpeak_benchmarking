import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

samplerate = 400

record = pd.read_csv("data/csv/CPSC/01.csv")
dataa = record.rawECG
peaks = record.Peaks

duration = 8 * samplerate
starta = 396800 # 00:16:32
startb = 400000 # 00:16:40
maximum = np.max(dataa)
minimum = np.min(dataa)
dataa = -1 + 2 * (dataa - minimum) / (maximum - minimum)
aa= dataa[starta:starta+duration]
ab = dataa[startb:startb+duration]
peaksa = peaks[starta:starta+duration]
peaksb = peaks[startb:startb+duration]
dataa = aa * 3.5
datab = ab * 3.5

t = np.linspace(0, int(duration / samplerate), duration, endpoint=False)  # 1000 Punkte in 10 Sekunden
yaxis = 2

fig, axs = plt.subplots(2, 1, figsize=(10, 6))

from datetime import datetime, timedelta

# Define the start time
start_time_a = datetime.strptime("00:16:32", "%H:%M:%S")
start_time_b = datetime.strptime("00:16:40", "%H:%M:%S")

# Generate time labels for the x-axis
time_range_a = [start_time_a + timedelta(seconds=i) for i in range(9)]  # 9 seconds (inclusive range)
time_range_b = [start_time_b + timedelta(seconds=i) for i in range(9)]  # 9 seconds (inclusive range)

# Add labels 'a)' and 'b)' to the subplots
axs[0].text(-0.1, 0.5, "a)", transform=axs[0].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")
axs[1].text(-0.1, 0.5, "b)", transform=axs[1].transAxes, fontsize=12, fontweight="bold", va="center", ha="right")

#axs[0].plot(t[peaksa == 1], dataa[peaksa == 1], "rx")
axs[0].set_ylim(-1.5, 1.5)
axs[0].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[0].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[0].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
#axs[0].set_xticks(np.arange(0, int(duration / samplerate) + 1))
axs[0].set_xticks([i for i in range(len(time_range_a))])  # Match the number of ticks with time_range
axs[0].set_xticklabels([time.strftime("%H:%M:%S") for time in time_range_a])
axs[0].plot(t, dataa, color="black")
axs[0].annotate("CPSC Record 01", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")


#axs[1].plot(t[peaksb == 1], datab[peaksb == 1], "rx")
axs[1].set_ylim(-1.5, 1.5)
axs[1].set_ylabel("Amplitude (mV)", fontweight="bold")
axs[1].axhline(0, color="orange", linestyle="--", linewidth=1.1, alpha=0.7)
axs[1].grid(True, linestyle="--", linewidth=1.1, alpha=0.7, color="orange")
#axs[1].set_xticks(np.arange(0, int(duration / samplerate) + 1))
axs[1].set_xticks([i for i in range(len(time_range_b))])  # Match the number of ticks with time_range
axs[1].set_xticklabels([time.strftime("%H:%M:%S") for time in time_range_b])
axs[1].set_xlabel("Time", fontweight="bold")
axs[1].plot(t, datab, color="black")
axs[1].annotate("CPSC Record 01", xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top")

plt.subplots_adjust(hspace=0.2)
plt.savefig('plots/easyandnoteasy.png', bbox_inches='tight', dpi=300)
plt.show()