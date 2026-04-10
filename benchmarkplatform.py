import pandas as pd
import implemented_algos as algorithm
import resultcomparator as resultcomparator
import csvfilenames as csvfilenames
import os

dirname = os.path.dirname(__file__)


def load_database(database = "MITAR"):

    match database:
        case "MITLT":
            dbname = "MIT Long Term Database"
            filepath = csvfilenames.mitlt_path
            filelist = csvfilenames.mitlt_all

        case "Fantasia":
            dbname = "Fantasia Database"
            filepath = csvfilenames.fantasia_path
            filelist = csvfilenames.fantasia_all

        case "PTT":
            dbname = "Pulse Transit Time Database"
            filepath = csvfilenames.ptt_path
            filelist = csvfilenames.ptt_all

        case "MITAR":
            dbname = "MIT Arrhythmia"
            filepath = csvfilenames.mitar_path
            filelist = csvfilenames.mitar_all

        case "MITNST":
            dbname = "MIT Noise Stress Test Database"
            filepath = csvfilenames.mitnst_path
            filelist = csvfilenames.mitnst_all

        case _:
            dbname = "MIT Arrhythmia"
            filepath = csvfilenames.mitar_path
            filelist = csvfilenames.mitar_all
    
    return dbname, filepath, filelist


# Used for obtaining the Benchmarking Results:

def r_peak_benchmarker(algo="elgendi", database = "MITAR", individualresults = False):
    """
    Used for Result determination: Uses the provided csv-Files obtained from the evaluation databases
    to analyze the detection performance of the specified algorithm on the corresponding database.
    :param algo:                Specifies which algorithm shall be used (see match-strings)
    :param database:            Database the analysis shall be done on
    :param individualresults:   Whether the results for each recording shall be printed or only the overall results
    """
    dbname, filepath, filelist = load_database(database=database)
    print(f"Analyzing {dbname} with {algo}:")
    truepos, falsepos, falseneg = 0, 0, 0
    worst_recording = None
    worst_f1 = 1
    worst_tp, worst_fp, worst_fn = 0, 0, 0
    tp_list, fp_list, fn_list = [], [], []
    for file in filelist:
        record = pd.read_csv(filepath + file + ".csv")
        data = record.normECG
        peaks = record.Peaks
        samplerate = (int(len(record) / record.Time[len(record)-1]))
        if filepath == os.path.join(dirname, 'data/csv/PulseTransitTime/'):
            samplerate = 500
        match algo:
            case "elgendi":
                foundpeaks = algorithm.elgendi(data=data, samplerate=samplerate)
            case "laitala":
                foundpeaks = algorithm.laitala(data=data, samplerate=samplerate)
            case "zhai":
                foundpeaks = algorithm.zhai(data=data, samplerate=samplerate)
            case "xia":
                foundpeaks = algorithm.xia(data=data, samplerate=samplerate)
            case "arteagaFalconi":
                foundpeaks = algorithm.arteagaFalconi(data=data, samplerate=samplerate)
            case "zahid":
                foundpeaks = algorithm.zahid(data=data, samplerate=samplerate)
            case "han_rnn":
                foundpeaks = algorithm.han_rnn(data=data, samplerate=samplerate)
            case "han_cnn":
                foundpeaks = algorithm.han_cnn(data=data, samplerate=samplerate)
            case "nguyen":
                foundpeaks = algorithm.nguyen(data=data, samplerate=samplerate)
            case "pan":
                foundpeaks = algorithm.pantompkins(data=data, samplerate=samplerate)
            case "hamilton":
                foundpeaks = algorithm.hamilton(data=data, samplerate=samplerate)
            case "xu":
                foundpeaks = algorithm.xu(data=data, samplerate=samplerate)
            case "shaik":
                foundpeaks = algorithm.shaik(data=data, samplerate=samplerate)
            case "celik":
                foundpeaks = algorithm.celik(data=data, samplerate=samplerate)
            case "xiang":
                foundpeaks = algorithm.xiang(data=data, samplerate=samplerate)
            case "park":
                foundpeaks = algorithm.park(data=data, samplerate=samplerate)
            case "kumari":
                foundpeaks = algorithm.kumari(data=data, samplerate=samplerate)
        # Determination of the amount of TP, FP and FN
        tp, fp, fn = resultcomparator.determination_tpfpfn(detected=foundpeaks, solution = peaks, samplerate = samplerate)
        truepos += tp
        falsepos += fp
        falseneg += fn
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

        precision, sensitivity, accuracy, f1score, der = resultcomparator.overallevaluation(tp,fp,fn)
        if f1score < worst_f1:
            # save the name, fp, fn and f1 score of the worst performing recording 
            worst_recording = file
            worst_tp = tp
            worst_fp = fp
            worst_fn = fn
            worst_f1 = f1score
        if individualresults:
            with open(os.path.join(dirname, f'results/{database}_results.txt'), 'a') as f:
                f.write(
                    f"Recording: {file} - Peaks: {tp+fn}, FP: {fp}, FN: {fn}, Recall: {sensitivity:.4f}, "
                    f"Precision: {precision:.4f}, Error Rate: {der:.4f}, F1-Score: {f1score:.4f}\n"
                )
    precision, sensitivity, accuracy, f1score, der = resultcomparator.overallevaluation(truepos,falsepos,falseneg)
    if len(tp_list) > 0:
        ci_lower, ci_upper = resultcomparator.bootstrap_f1_ci(tp_list, fp_list, fn_list)
    else:
        ci_lower, ci_upper = 0, 0
    # Save the worst performing recording in a text file as well as the overall results for the corresponding algorithm and database
    with open(os.path.join(dirname, f'results/{database}_results.txt'), 'a') as f:
        if worst_recording is not None:
            worst_precision, worst_recall, _, worst_f1score, worst_der = resultcomparator.overallevaluation(
                worst_tp, worst_fp, worst_fn
            )
            f.write(
                f"Worst performing recording: {worst_recording} - Peaks: {worst_tp+worst_fn}, "
                f"FP: {worst_fp}, FN: {worst_fn}, Recall: {worst_recall:.4f}, Precision: {worst_precision:.4f}, "
                f"Error Rate: {worst_der:.4f}, F1-Score: {worst_f1score:.4f}\n"
            )
        f.write(
            f"Overall results - Peaks: {truepos+falseneg}, FP: {falsepos}, FN: {falseneg}, "
            f"Recall: {sensitivity:.4f}, Precision: {precision:.4f}, Error Rate: {der:.4f}, "
            f"F1-Score: {f1score:.4f} (95% CI: [{ci_lower:.4f}-{ci_upper:.4f}])\n\n"
        )

    return {
        "database": database,
        "algo": algo,
        "tp": truepos,
        "fp": falsepos,
        "fn": falseneg,
        "recall": sensitivity,
        "precision": precision,
        "error_rate": der,
        "f1": f1score,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tp_list": tp_list,
        "fp_list": fp_list,
        "fn_list": fn_list,
        "worst_recording": worst_recording,
        "worst_tp": worst_tp,
        "worst_fp": worst_fp,
        "worst_fn": worst_fn,
    }
