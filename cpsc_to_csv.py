import numpy as np
import scipy
import pandas as pd
import os

dirname = os.path.dirname(__file__)


"""
Used to save files from the CPSC2020 database as csv files
"""

def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    filepath = os.path.join(dirname, 'data/raw/CPSC/')
    if not os.path.exists(dirname + "/data/csv/CPSC/"):
        os.makedirs(dirname + "/data/csv/CPSC/")
    return filenames, filepath

def create_csv(time, rawecg, normecg, peaks, pathname, filename):
    csvframe = pd.concat([time, rawecg, normecg, peaks], axis=1)
    csvframe.to_csv(dirname + "/data/csv/" + pathname + filename + ".csv", index_label="Index", columns=["Time", "rawECG", "normECG", "Peaks"])

def createcpsccsvfiles():
    samplerateofrecords = 400
    numberofchannels = 1
    filenames, filepath = cpsc()
    recordlengthmin = 0
    recordlengthmax = 0
    totallength = 0
    numberofrecords = 0
    rpeakamount = 0
    for i in filenames:
        numberofrecords += 1
        ecgdata = scipy.io.loadmat(filepath + "A" + i + ".mat")
        ecgdata = ecgdata["ecg"]
        rawECG = np.squeeze(ecgdata)
        peakdata = scipy.io.loadmat(filepath + "RPN_" + i + ".mat")
        peakdata = peakdata["R"]
        peakdata = np.squeeze(peakdata)
        rpeakamount += len(peakdata)
        peaks = np.zeros(len(rawECG))
        normECG = rawECG.copy()
        minimum = np.min(rawECG)
        maximum = np.max(rawECG)
        normECG = -1 + 2 * (normECG - minimum) / (maximum - minimum)
        rawECG = pd.Series(rawECG)
        normECG = pd.Series(normECG)
        peaks = pd.Series(peaks)
        for peakpoint in peakdata:
            peakpoint = int(peakpoint)
            peaks[peakpoint] = 1
        recordlength = len(rawECG)/samplerateofrecords
        if recordlengthmin == 0:
            recordlengthmin = recordlength
        elif recordlengthmin > recordlength:
            recordlengthmin = recordlength
        if recordlengthmax < recordlength:
            recordlengthmax = recordlength
        totallength += recordlength
        rawECG = rawECG.rename("rawECG")
        normECG = normECG.rename("normECG")
        peaks = peaks.rename("Peaks")
        time = pd.Series(np.arange(len(rawECG)) / samplerateofrecords)
        time = time.rename("Time")
        create_csv(time, rawECG, normECG, peaks, "CPSC/", i)

    print(rpeakamount, samplerateofrecords, numberofrecords, totallength, recordlengthmin, recordlengthmax, numberofchannels)
    file_path = os.path.join(dirname, 'data/csv/CPSC/summary.txt')
    with open(file_path, "w") as file:
        file.write(f"rpeakamount = {rpeakamount}\n")
        file.write(f"samplerateofrecords = {samplerateofrecords}\n")
        file.write(f"numberofrecords = {numberofrecords}\n")
        file.write(f"recordlength = {totallength}\n")
        file.write(f"recordlengthmin = {recordlengthmin}\n")
        file.write(f"recordlengthmax = {recordlengthmax}\n")
        file.write(f"numberofchannels = {numberofchannels}\n")


if __name__ == "__main__":
    createcpsccsvfiles()