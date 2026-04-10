import matplotlib.pyplot as plt
import numpy as np
import scipy

data = {
    "Algorithmus": ["Pan", "Hamilton", "Shaik", "Park", "Osman", "Xu", "Nguyen", "Zhai", "Kumari", "Xia", "Zahid", "Laitala", "Han CNN", "Xiang", "Celik"],
    "24 dB": [0.998946012, 0.982418879, 0.999648424, 0.998944282, 0.999296765, 0.99941404, 0.99929693, 0.997427502, 0.999297095, 0.980396844, 0.997894244, 0.996029893, 0.992091184,0.999531287],
    "18 dB": [0.998244177, 0.981243364, 0.998244587, 0.998356808, 0.994743605, 0.998945765, 0.9985932, 0.989788814, 0.998010532, 0.979236277, 0.996029893, 0.990009294, 0.972966807,0.997427502],
    "12 dB": [0.984629608, 0.968237347, 0.98057248, 0.99471769, 0.969384245, 0.991616209, 0.986425339, 0.968547746, 0.978532889, 0.965770461, 0.978093818, 0.956621266, 0.928688078,0.967654069],
    "6 dB": [0.939833795, 0.921137042, 0.93988465, 0.957523566, 0.922733861, 0.945870923, 0.943603543, 0.591863093, 0.915712124, 0.929782082, 0.941740002, 0.902884182, 0.885334441,0.90534188],
    "0 dB": [0.869361203, 0.85693106, 0.883842225, 0.8618058, 0.857483731, 0.805730517, 0.85933897, 0.110575916, 0.853286985, 0.874844545, 0.896867838, 0.846188042, 0.855085691,0.842926418],
    "-6 dB": [0.657349398, 0.770648168, 0.821548822, 0.785305344, 0.721356241, 0.580535003, 0.623389619, 0.182434979, 0.803591702, 0.806718822, 0.832273624, 0.799063336, 0.788920538,0.580995704]
}

data["Elgendi"] = [0.99988278, 0.993008623, 0.969358697, 0.914946504, 0.84886859, 0.796559229]
data["Laitala"] = [0.999648424, 0.989317232, 0.956934554, 0.905870777, 0.830586684, 0.762886598]
data["Han RNN"] = [0.996029893, 0.998127779, 0.984075698, 0.94810847, 0.900509927, 0.834278238]

average_scores = {}
for algo, scores in data.items():
    if algo != "Algorithmus":
        average_scores[algo] = sum(scores) / len(scores)

def interpol(x, y):
    x_smooth = np.linspace(min(x), max(x), 300)
    spline = scipy.interpolate.make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

levels = ["24 dB", "18 dB", "12 dB", "6 dB", "0 dB", "-6 dB"]
x_values = np.arange(len(levels))
average_scores = [average_scores[level] for level in levels]

plt.figure(figsize=(10, 4))

scores_elgendi = data["Elgendi"]
x_smooth_elgendi, y_smooth_elgendi = interpol(x_values, scores_elgendi)
plt.plot(x_smooth_elgendi, y_smooth_elgendi, marker="", color="blue", linestyle="-", linewidth=2, label="Elgendi")

scores_laitala = data["Laitala"]
x_smooth_laitala, y_smooth_laitala = interpol(x_values, scores_laitala)
plt.plot(x_smooth_laitala, y_smooth_laitala, marker="", color="green", linestyle="-", linewidth=2, label="Laitala")

scores_han_rnn = data["Han RNN"]
x_smooth_han_rnn, y_smooth_han_rnn = interpol(x_values, scores_han_rnn)
plt.plot(x_smooth_han_rnn, y_smooth_han_rnn, marker="", color="red", linestyle="-", linewidth=2, label="Han RNN")

x_smooth_avg, y_smooth_avg = interpol(x_values, average_scores)
plt.plot(x_smooth_avg, y_smooth_avg, marker="", color="black", linestyle="--", linewidth=2, label="Average")

plt.xlabel("Signal-to-Noise Ratio", fontweight="bold")
plt.ylabel("F1-Score", fontweight="bold")
plt.xticks(x_values, levels)
plt.grid(True, linestyle="--", linewidth=1.5, alpha=0.7, color="orange")
plt.ylim(0.69, 1.03)
plt.legend()
plt.savefig('plots/noiselevels.png', bbox_inches='tight', dpi=300)
plt.show()
