import matplotlib.pyplot as plt
import numpy as np
import scipy
import benchmarkplatform
import pandas as pd
import algorithms as algos
import resultcomparator

results = {}
for algo in ["elgendi", "laitala", "zhai", "xia", "arteagaFalconi", "zahid", "han_rnn", "han_cnn", "nguyen", "pantompkins", "hamilton", "xu", "shaik", "celik", "xiang", "park", "kumari"]:
    dbname, filepath, filelist = benchmarkplatform.load_database(database="MITNST")
    print(f"Analyzing {dbname} with {algo}:")
    for file in filelist:
        record = pd.read_csv(filepath + file + ".csv")
        data = record.normECG
        peaks = record.Peaks
        samplerate = (int(len(record) / record.Time[len(record)-1]))
        match algo:
            case "elgendi":
                foundpeaks = algos.elgendi(data=data, samplerate=samplerate)
            case "laitala":
                foundpeaks = algos.laitala(modelname="laitalamodell_withf1.h5", data=data, samplerate=samplerate)
            case "zhai":
                foundpeaks = algos.zhai(data=data, samplerate=samplerate)
            case "xia":
                foundpeaks = algos.xia(data=data, samplerate=samplerate)
            case "arteagaFalconi":
                foundpeaks = algos.arteagaFalconi(data=data, samplerate=samplerate)
            case "zahid":
                foundpeaks = algos.zahid(modelname="zahidmodell_withf1.h5", data=data, samplerate=samplerate)
            case "han_rnn":
                foundpeaks = algos.han_rnn(modelname="hanrnnmodell_withf1_2.h5", data=data, samplerate=samplerate)
            case "han_cnn":
                foundpeaks = algos.han_cnn(modelname="hancnnmodell_withf1_2.h5", data=data, samplerate=samplerate)
            case "nguyen":
                foundpeaks = algos.nguyen(data=data, samplerate=samplerate)
            case "pantompkins":
                foundpeaks = algos.pantompkins(data=data, samplerate=samplerate)
            case "hamilton":
                foundpeaks = algos.hamilton(data=data, samplerate=samplerate)
            case "xu":
                foundpeaks = algos.xu(data=data, samplerate=samplerate)
            case "shaik":
                foundpeaks = algos.shaik(data=data, samplerate=samplerate)
            case "celik":
                foundpeaks = algos.celik(modelname="celikmodell_withf1.h5", data=data, samplerate=samplerate)
            case "xiang":
                foundpeaks = algos.xiang(modelname="xiangmodell_withf1_13.h5", data=data, samplerate=samplerate)
            case "park":
                foundpeaks = algos.park(data=data, samplerate=samplerate)
            case "kumari":
                foundpeaks = algos.kumari(data=data, samplerate=samplerate)
        # Determination of the amount of TP, FP and FN
        tp, fp, fn = resultcomparator.determination_tpfpfn(detected=foundpeaks, solution = peaks, samplerate = samplerate)
        precision, sensitivity, accuracy, f1score, der = resultcomparator.overallevaluation(tp,fp,fn)

        if algo not in results:
            results[algo] = [0, 0, 0, 0, 0, 0]
        if file.endswith("_6"):
            results[algo][5] += f1score
        elif file.endswith("00"):
            results[algo][4] += f1score
        elif file.endswith("06"):
            results[algo][3] += f1score
        elif file.endswith("12"):
            results[algo][2] += f1score
        elif file.endswith("18"):
            results[algo][1] += f1score
        elif file.endswith("24"):
            results[algo][0] += f1score
    
    # Average F1-Score for each noise level
    for i in range(6):
        results[algo][i] /= 2


average_scores = [0, 0, 0, 0, 0, 0]

for algo, scores in results.items():
    for i in range(len(average_scores)):
        average_scores[i] += scores[i]

for i in range(len(average_scores)):
    average_scores[i] /= len(results)

def interpol(x, y):
    x_smooth = np.linspace(min(x), max(x), 300)
    spline = scipy.interpolate.make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

levels = ["24 dB", "18 dB", "12 dB", "6 dB", "0 dB", "-6 dB"]
x_values = np.arange(len(levels))

plt.figure(figsize=(10, 4))

scores_elgendi = results["elgendi"]
x_smooth_elgendi, y_smooth_elgendi = interpol(x_values, scores_elgendi)
plt.plot(x_smooth_elgendi, y_smooth_elgendi, marker="", color="blue", linestyle="-", linewidth=2, label="Elgendi")

scores_laitala = results["laitala"]
x_smooth_laitala, y_smooth_laitala = interpol(x_values, scores_laitala)
plt.plot(x_smooth_laitala, y_smooth_laitala, marker="", color="green", linestyle="-", linewidth=2, label="Laitala")

scores_han_rnn = results["han_rnn"]
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
