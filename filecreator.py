from __future__ import annotations

import argparse
from pathlib import Path

import wfdb
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CSV_DIR = BASE_DIR / "data" / "csv"

'''
Creates CSV Evaluation Files (outcomment the corresponding line at the end of the file
Assumption: The folders containing the original database files are stored in the directories
data/raw/MIT_Arrythmia, data/raw/MIT_NSTDB, data/raw/MITLongTerm, 
data/raw/Fantasia, data/raw/PulseTransitTime
'''

def readecgfile(file, annotators, annotatorending):
    """
    Read an ECG file and extract relevant information.

    Args:
        file (str): The path to the ECG file.
        annotators (list): A list of annotators to consider.
        annotatorending (str): The file extension for the annotations.

    Returns:
        tuple: A tuple containing the following:
            - record (DataFrame): The ECG record with additional columns.
            - rpeaks (ndarray): An array indicating the positions of R-peaks.
            - samplerate (int): The sampling rate of the ECG record.
    """
    record_base = RAW_DIR / str(file).lstrip("/")
    record = wfdb.rdrecord(str(record_base))
    samplerate = record.fs
    record = record.to_dataframe()
    record = record.reset_index().rename(columns={"index": "Time"})
    record["Time"] = pd.to_timedelta(record["Time"]).dt.total_seconds()
    annotation = wfdb.rdann(str(record_base), extension=annotatorending)
    rpeakpositions = annotation.sample[np.in1d(annotation.symbol, annotators)]
    rpeaks = np.zeros(len(record), dtype=int)
    rpeaks[rpeakpositions] = 1
    record["Peaks"] = rpeaks
    return record, rpeaks, samplerate


def create_csv(time, rawecg, normecg, peaks, pathname, filename):
    """
    Create a CSV file from the given ECG data.

    Args:
        time (Series): The time values of the ECG data.
        rawecg (Series): The raw ECG data.
        normecg (Series): The normalized ECG data.
        peaks (Series): The peaks data.
        pathname (str): The path to save the CSV file.
        filename (str): The name of the CSV file.
    """
    out_dir = CSV_DIR / str(pathname).strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    csvframe = pd.concat([time, rawecg, normecg, peaks], axis=1)
    out_file = out_dir / f"{filename}.csv"
    csvframe.to_csv(out_file, index_label="Index", columns=["Time", "rawECG", "normECG", "Peaks"])


# MIT Arrhythmia Database
mit_annotators = ["·", "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"]
mit_path = "MIT_Arrhythmia"
mit_annotatorending = "atr"
mit_recordlist = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121", "122",
    "123", "124", "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234"]
mit_ecgrow = 1


# MIT Noise Stress Test
mitnst_annotators = ["·", "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"]
mitnst_path = "MIT_NSTDB"
mitnst_annotatorending = "atr"
mitnst_recordlist = ["118e_6", "118e00", "118e06", "118e12", "118e18", "118e24", "119e_6", "119e00", "119e06", "119e12", "119e18", "119e24"]
mitnst_ecgrow = 1


# MIT Long Term
mitlt_annotators = ["·", "N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"]
mitlt_path = "MITLongTerm"
mitlt_annotatorending = "atr"
mitlt_recordlist = ["14046", "14134", "14149", "14157", "14172", "14184", "15814"]
mitlt_ecgrow = 1

# Fantasia Database
fantasia_annotators = ["N"]
fantasia_path = "Fantasia"
fantasia_annotatorending = "ecg"
fantasia_recordlist = ["f1o01", "f1o02", "f1o03", "f1o04", "f1o05", "f1o06", "f1o07", "f1o08", "f1o09", "f1o10",
    "f1y01", "f1y02", "f1y03", "f1y04", "f1y05", "f1y06", "f1y07", "f1y08", "f1y09", "f1y10",
    "f2o01", "f2o02", "f2o03", "f2o04", "f2o05", "f2o06", "f2o07", "f2o08", "f2o09", "f2o10",
    "f2y01", "f2y02", "f2y03", "f2y04", "f2y05", "f2y06", "f2y07", "f2y08", "f2y09", "f2y10"
]
fantasia_ecgrow = 2

#Pulse Transit Time:
def createfiles_pttdatabase():
    """
    Create CSV files for the Pulse Transit Time database.
    """
    filelist = ["s10_run", "s10_sit", "s10_walk", "s11_run", "s11_sit", "s11_walk", "s12_run",
                "s12_sit", "s12_walk", "s13_run", "s13_sit", "s13_walk", "s14_run", "s14_sit",
                "s14_walk", "s15_run", "s15_sit", "s15_walk", "s16_run", "s16_sit", "s16_walk",
                "s17_run", "s17_sit", "s17_walk", "s18_run", "s18_sit", "s18_walk", "s19_run",
                "s19_sit", "s19_walk", "s1_run", "s1_sit", "s1_walk", "s20_run", "s20_sit",
                "s20_walk", "s21_run", "s21_sit", "s21_walk", "s22_run", "s22_sit", "s22_walk",
                "s2_run", "s2_sit", "s2_walk", "s3_run", "s3_sit", "s3_walk", "s4_run",
                "s4_sit", "s4_walk", "s5_run", "s5_sit", "s5_walk", "s6_run", "s6_sit",
                "s6_walk", "s7_run", "s7_sit", "s7_walk", "s8_run", "s8_sit", "s8_walk",
                "s9_run", "s9_sit", "s9_walk"]
    rpeakamount = 0
    samplerateofrecords = 500
    numberofrecords = len(filelist)
    recordlength = 0
    recordlengthmin = 0
    recordlengthmax = 0
    numberofchannels = 1
    for i in filelist:
        path = "PulseTransitTime"
        csv = pd.read_csv(RAW_DIR / path / f"{i}.csv")
        columnsofinterest = ["time", "ecg", "peaks"]
        data = csv[columnsofinterest]
        time = data.time
        time = pd.to_datetime(time)
        time = (time - time.min()).dt.total_seconds().round(3)
        rawECG = data.ecg
        maximum = np.max(rawECG)
        minimum = np.min(rawECG)
        normECG = -1 + 2 * (rawECG - minimum) / (maximum - minimum)
        peaks = data.peaks
        time.name = "Time"
        rawECG.name = "rawECG"
        normECG.name = "normECG"
        peaks.name = "Peaks"
        create_csv(time, rawECG, normECG, peaks, path, i)
        recordlength = recordlength + len(time)
        if recordlengthmin == 0:
            recordlengthmin = len(time)
        elif len(time) < recordlengthmin:
            recordlengthmin = len(time)
        if recordlengthmax == 0:
            recordlengthmax = len(time)
        elif len(time) > recordlengthmax:
            recordlengthmax = len(time)
        numberofrpeaks = peaks.value_counts().get(1, 0)
        rpeakamount = rpeakamount + numberofrpeaks
    recordlength = recordlength / samplerateofrecords
    recordlengthmin = recordlengthmin / samplerateofrecords
    recordlengthmax = recordlengthmax / samplerateofrecords
    file_path = CSV_DIR / "PulseTransitTime" / "summary.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as file:
        file.write(f"rpeakamount = {rpeakamount}\n")
        file.write(f"samplerateofrecords = {samplerateofrecords}\n")
        file.write(f"numberofrecords = {numberofrecords}\n")
        file.write(f"recordlength = {recordlength}\n")
        file.write(f"recordlengthmin = {recordlengthmin}\n")
        file.write(f"recordlengthmax = {recordlengthmax}\n")
        file.write(f"numberofchannels = {numberofchannels}\n")

def createfiles(recordlist, annotatorlist, ecgrow, path, annotatorending):
    """
    Create CSV files for the given ECG records.

    Args:
        recordlist (list): A list of record names.
        annotatorlist (list): A list of annotators to consider.
        ecgrow (int): The column index of the ECG data in the record.
        path (str): The path to the ECG records.
        annotatorending (str): The file extension for the annotations.
    """
    rpeakamount = 0
    samplerateofrecords = 0
    numberofrecords = len(recordlist)
    recordlength = 0
    recordlengthmin = 0
    recordlengthmax = 0
    numberofchannels = 0
    for i in recordlist:
        record, rpeaks, samplerate = readecgfile(str(Path(path) / i), annotatorlist, annotatorending)
        rawECG = record.iloc[:, ecgrow]
        rawECG = rawECG.fillna(0)
        maximum = np.max(rawECG)
        minimum = np.min(rawECG)
        normalized = -1 + 2 * (rawECG - minimum) / (maximum - minimum)
        normECG = normalized
        rawECG = rawECG.rename("rawECG")
        normECG = normECG.rename("normECG")
        create_csv(record.Time, rawECG, normECG, record.Peaks, path, i)
        recordlength = recordlength + len(record.Time)
        if recordlengthmin == 0:
            recordlengthmin = len(record.Time)
        elif len(record.Time) < recordlengthmin:
            recordlengthmin = len(record.Time)
        if recordlengthmax == 0:
            recordlengthmax = len(record.Time)
        elif len(record.Time) > recordlengthmax:
            recordlengthmax = len(record.Time)
        numberofchannels = record.shape[1]-2
        if samplerateofrecords == 0:
            samplerateofrecords = samplerate
        elif samplerateofrecords != samplerate:
            print("Warning: samplerate does not match!")
        numberofrpeaks = record["Peaks"].value_counts().get(1, 0)
        rpeakamount = rpeakamount + numberofrpeaks
    recordlength = recordlength / samplerateofrecords
    recordlengthmin = recordlengthmin / samplerateofrecords
    recordlengthmax = recordlengthmax / samplerateofrecords
    file_path = CSV_DIR / str(path).strip("/") / "summary.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as file:
        file.write(f"rpeakamount = {rpeakamount}\n")
        file.write(f"samplerateofrecords = {samplerateofrecords}\n")
        file.write(f"numberofrecords = {numberofrecords}\n")
        file.write(f"recordlength = {recordlength}\n")
        file.write(f"recordlengthmin = {recordlengthmin}\n")
        file.write(f"recordlengthmax = {recordlengthmax}\n")
        file.write(f"numberofchannels = {numberofchannels}\n")


def create_database_csvs(database: str) -> None:
    """Create CSV evaluation files for one database.

    `database` uses the same short names as `main.py` / `benchmarkplatform.py`.
    """
    match database:
        case "MITAR":
            createfiles(mit_recordlist, mit_annotators, mit_ecgrow, mit_path, mit_annotatorending)
        case "MITNST":
            createfiles(mitnst_recordlist, mitnst_annotators, mitnst_ecgrow, mitnst_path, mitnst_annotatorending)
        case "MITLT":
            createfiles(mitlt_recordlist, mitlt_annotators, mitlt_ecgrow, mitlt_path, mitlt_annotatorending)
        case "Fantasia":
            createfiles(fantasia_recordlist, fantasia_annotators, fantasia_ecgrow, fantasia_path, fantasia_annotatorending)
        case "PTT":
            createfiles_pttdatabase()
        case _:
            raise ValueError(f"Unknown database: {database}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create CSV evaluation files from raw ECG databases (wfdb / csv)."
    )
    parser.add_argument(
        "--db",
        default="all",
        choices=["all", "MITAR", "MITNST", "MITLT", "Fantasia", "PTT"],
        help="Which database to process (default: all).",
    )
    args = parser.parse_args(argv)

    if args.db == "all":
        for db in ["MITAR", "MITNST", "MITLT", "Fantasia", "PTT"]:
            create_database_csvs(db)
    else:
        create_database_csvs(args.db)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
