import numpy as np
import matplotlib.pyplot as plt

# Parameters for the ECG signal
fs = 1000  # Sampling frequency in Hz
duration = 1.2  # Duration of one heartbeat in seconds

# Create the time axis for one heartbeat
time = np.linspace(0, duration, int(fs * duration), endpoint=False)

def generate_ideal_ecg(time):
    signal = np.zeros_like(time)

    # P wave
    p_wave = 0.1 * np.exp(-((time - 0.2) / 0.04)**2)
    p_peak = time[np.argmax(p_wave)]

    # QRS complex
    q_wave = -0.15 * np.exp(-((time - 0.3) / 0.02)**2)
    q_peak = time[np.argmin(q_wave)]
    
    r_wave = 1.0 * np.exp(-((time - 0.32) / 0.01)**2)
    r_peak = time[np.argmax(r_wave)]
    
    s_wave = -0.25 * np.exp(-((time - 0.34) / 0.02)**2)
    s_peak = time[np.argmin(s_wave)]

    # T wave
    t_wave = 0.30 * np.exp(-((time - 0.6) / 0.1)**2)
    t_peak = time[np.argmax(t_wave)]

    # U wave
    u_wave = 0.03 * np.exp(-((time - 0.95) / 0.08)**2)
    u_peak = time[np.argmax(u_wave)]

    # Sum of all waveform components
    signal += p_wave + q_wave + r_wave + s_wave + t_wave + u_wave

    # Return peak locations
    peaks = {
        "P": (p_peak, signal[np.argmax(p_wave)]),
        "Q": (q_peak-0.01, signal[np.argmin(q_wave)]-0.07),
        "R": (r_peak, signal[np.argmax(r_wave)]),
        "S": (s_peak, signal[np.argmin(s_wave)]-0.07),
        "T": (t_peak, signal[np.argmax(t_wave)]),
        "U": (u_peak, signal[np.argmax(u_wave)])
    }

    return signal, peaks

# Generate an ECG signal for one heartbeat
ekgsignal, peaks = generate_ideal_ecg(time)

# Create the time axis for two consecutive heartbeats
double_time = np.concatenate([time, time + duration])
double_ekg = np.concatenate([ekgsignal, ekgsignal])

# Determine the R peaks for two cardiac cycles
r_peak1 = peaks["R"][0]
r_peak2 = r_peak1 + duration

# Plot the ECG signal twice in a row
plt.figure(figsize=(10, 6))
plt.plot(double_time, double_ekg, color="black", label="EKG-Signal", linewidth=2)

# Label the wave peaks
for label, (t, amp) in peaks.items():
    plt.text(t, amp + 0.02, label, fontsize=12, fontweight="bold", ha="center")
    plt.text(t + duration, amp + 0.02, label, fontsize=12, fontweight="bold", ha="center")

# Mark the RR interval with a double-headed arrow
arrow_y = 0.8  # Arrow height above the signal
plt.annotate("", xy=(r_peak1+0.01, arrow_y), xytext=(r_peak2-0.01, arrow_y),
             arrowprops=dict(arrowstyle="<->", linewidth=2))
plt.text((r_peak1 + r_peak2) / 2, arrow_y + 0.02, "RR-Interval", fontsize=12, fontweight="bold", ha="center")

# Plot labels
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.axis("off")
plt.tight_layout()
plt.savefig('plots/heartcycle.png', bbox_inches='tight', dpi=300)
plt.show()