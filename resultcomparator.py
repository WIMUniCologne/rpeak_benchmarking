import numpy as np
np.random.seed(42)

def determination_tpfpfn(detected, solution, samplerate, tolerance = 0.1):
    detected = detected.copy()
    solution = solution.copy()
    errormap = np.zeros(len(detected))
    tolwindow = 2*int(tolerance * samplerate)
    # Check whether two or more detected R-Peaks are directly next to each other
    # If that is the case, the detections are replaced by one detection in the middle
    # (This is the case in raw outputs of DL models, probably no longer necessary due to the detection logic)
    for i in range(0, len(detected)):
        if detected[i] == 1 and i+1 < len(detected):
            if detected[i+1] == 1:
                k = i
                while detected[k] == 1 and k < len(detected):
                    k += 1
                detected[i:k] = [0] * (k - i)
                detected[int((k-i)/2)] = 1
    tp, fp, fn = 0, 0, 0
    for i in range(0, len(detected)):
        if detected[i] == 1:
            lower_index = max(0, i - tolwindow)
            upper_index = min(len(solution), i + tolwindow)
            width = upper_index - lower_index
            #Check whether an annotated R-Peak lies in the tolerance window
            if np.sum(solution[lower_index:upper_index]) > 0:
                tp += 1
            else:
                # For the respective tolerance window, any further detected R-Peak is counted as FP for physiological reasons
                fp += 1
                #errormap[i] = 1
            detected[i] = 0
            solution[lower_index:upper_index] = [0] * width
    #errormap += -solution
    fn += np.sum(solution)
    return tp, fp, fn


def overallevaluation(tp,fp,fn):
    #calculates evaluation metrics
    precision, sensitivity, accuracy, f1score, der = 0,0,0,0,0
    if tp+fp > 0:
        precision = tp/(tp+fp)
    if tp+fn > 0:
        sensitivity = tp/(tp+fn)
        der = (fp + fn)/(tp+fn)
    if tp+fp+fn > 0:
        accuracy = tp/(tp+fp+fn)
    if precision+sensitivity > 0:
        f1score = 2*precision*sensitivity/(precision+sensitivity)
    return precision, sensitivity, accuracy, f1score, der


def compute_f1(tp, fp, fn):
    denom = 2*tp + fp + fn
    return 0 if denom == 0 else 2*tp / denom


def bootstrap_f1_ci(tp_list, fp_list, fn_list, n_iter=10000):
    n = len(tp_list)
    f1_scores = []

    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)

        tp = np.sum(np.array(tp_list)[idx])
        fp = np.sum(np.array(fp_list)[idx])
        fn = np.sum(np.array(fn_list)[idx])

        f1_scores.append(compute_f1(tp, fp, fn))

    lower = np.percentile(f1_scores, 2.5)
    upper = np.percentile(f1_scores, 97.5)

    return lower, upper