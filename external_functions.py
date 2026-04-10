import tensorflow as tf
import numpy as np

class F1Score(tf.keras.metrics.Metric):
    # Model Training was done using this class taken from
    # https://stackoverflow.com/questions/64474463/custom-f1-score-metric-in-tensorflow
    # due to issues with the build in metric during training
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision_result = self.precision.result()
        recall_result = self.recall.result()

        return 2 * (precision_result * recall_result) / (precision_result + recall_result + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


def sedghamiz_thresholding(filtered, initial_peaks, samplerate):
    # Uses the Noise- and Signal- Level calculation from
    # PhysioNet: https://www.physionet.org/content/cpsc2021/1.0.0/python_entry/utils.py
    # Originally a MatLab Implementation from Hooman Sedghamiz
    # https://www.researchgate.net/publication/313673153_Matlab_Implementation_of_Pan_Tompkins_ECG_QRS_detect
    mindistance = 0.272
    peaklocs, peakvalues = initial_peaks, filtered[initial_peaks]
    init_peakamount = len(peakvalues)
    skip, current_rr, mean_rr, searchback, amount_detected_beats1, amount_detected_peaks2, noise_count = 0,0,0,0,0,0,0
    qrsc, qrsi, qrsiraw, qrs_amplitude, nois_c, nois_i, signallevelbuffer1, noiselevelbuffer1, signallevelbuffer2, noiselevelbuffer2, thrsbuffer1, thrsbuffer2 = (
    np.zeros(init_peakamount) for _ in range(12))
    signal_threshold1 = max(filtered[:int(2 * samplerate)]) * 1 / 3
    noise_threshold1 = np.mean(filtered[:int(2 * samplerate)]) * 1 / 2
    signal_level1 = signal_threshold1
    noise_level1 = noise_threshold1
    signal_threshold2 = max(filtered[:int(2 * samplerate)]) * 1 / 3
    noise_threshold2 = np.mean(filtered[:int(2 * samplerate)]) * 1 / 2
    signal_level2 = signal_threshold2
    noise_level2 = noise_threshold2
    for i in range(init_peakamount):
        peak_loc = peaklocs[i]
        min_distance_sample = round(mindistance * samplerate)
        sample_range = 0.150 * samplerate
        valid_peak = 1 <= peak_loc - round(sample_range) and peak_loc <= len(filtered)
        if valid_peak:
            within_min_distance = 0 <= peak_loc - min_distance_sample and peak_loc < len(filtered)
            if within_min_distance:
                idx_max = np.argmax(filtered[peak_loc - min_distance_sample:peak_loc])
                y_i = filtered[peak_loc - min_distance_sample + idx_max]
                x_i = idx_max
            else:
                y_i, x_i = 0, 0
        else:
            if i == 0:
                max_index = np.argmax(filtered[:peak_loc])
                x_i, y_i = max_index, filtered[max_index]
                searchback = 1
            elif peak_loc >= len(filtered):
                y_i, x_i = max(filtered[peak_loc - min_distance_sample:])
        if amount_detected_beats1 >= 9:
            rrdifference = np.diff(qrsi[amount_detected_beats1 - 8:amount_detected_beats1])
            mean_rr = np.mean(rrdifference)
            comp = qrsi[amount_detected_beats1] - qrsi[amount_detected_beats1 - 1]
            if comp <= 0.92 * mean_rr or comp >= 1.16 * mean_rr:
                signal_threshold1 = 0.5 * signal_threshold1
                signal_threshold2 = 0.5 * signal_threshold2
            else:
                current_rr = mean_rr
        if current_rr:
            test_m = current_rr
        elif mean_rr and current_rr == 0:
            test_m = mean_rr
        else:
            test_m = 0
        if test_m:
            if peaklocs[i] - qrsi[amount_detected_beats1] >= round(1.66 * test_m):
                max_index = int(qrsi[amount_detected_beats1] + round(0.200 * samplerate))
                min_index = int(peaklocs[i] - round(0.200 * samplerate))
                if max_index < min_index:
                    max_value_index = np.argmax(filtered[max_index:min_index])
                    pks_temp = filtered[max_index:min_index][max_value_index]
                    locs_temp = max_index + max_value_index
                    locs_temp = qrsi[amount_detected_beats1] + round(0.200 * samplerate) + locs_temp - 1
                    if pks_temp > noise_threshold1:
                        amount_detected_beats1 += 1
                        qrsc[amount_detected_beats1] = pks_temp
                        qrsi[amount_detected_beats1] = locs_temp
                        if locs_temp <= len(filtered):
                            start_index = int(locs_temp - round(mindistance * samplerate))
                            end_index = int(locs_temp)
                            max_value_index = np.argmax(filtered[start_index:end_index])
                            y_i_t = filtered[start_index:end_index][max_value_index]
                            x_i_t = start_index + max_value_index
                        else:
                            start_index = int(locs_temp - round(mindistance * samplerate))
                            end_index = int(locs_temp)
                            slice_data = filtered[start_index:end_index]
                            if len(slice_data) > 0:
                                max_value_index = np.argmax(slice_data)
                                y_i_t = slice_data[max_value_index]
                                x_i_t = start_index + max_value_index
                            else:
                                y_i_t, x_i_t = 0, 0
                        if y_i_t > noise_threshold2:
                            amount_detected_peaks2 += 1
                            qrsiraw[amount_detected_peaks2] = locs_temp - round(mindistance * samplerate) + (x_i_t - 1)
                            qrs_amplitude[amount_detected_peaks2] = y_i_t
                            signal_level2 = 0.25 * y_i_t + 0.75 * signal_level2
                        signal_level1 = 0.25 * pks_temp + 0.75 * signal_level1
        if peakvalues[i] >= signal_threshold1:
            if amount_detected_beats1 >= 3:
                if peaklocs[i] - qrsi[amount_detected_beats1] <= round(0.3600 * samplerate):
                    if peaklocs[i] - round(0.075 * samplerate) >= 0 and peaklocs[i] < len(filtered):
                        slope1 = np.mean(np.diff(filtered[peaklocs[i] - round(0.075 * samplerate):peaklocs[i]]))
                    else:
                        slope1 = 0
                    if int(qrsi[amount_detected_beats1] - round(0.075 * samplerate)) >= 0 and qrsi[amount_detected_beats1] < len(filtered):
                        Slope2 = np.mean(
                            np.diff(filtered[int(qrsi[amount_detected_beats1] - round(0.075 * samplerate)):int(qrsi[amount_detected_beats1])]))
                    else:
                        Slope2 = 0
                    if abs(slope1) <= abs(0.5 * (Slope2)):
                        noise_count += 1
                        nois_c[noise_count] = peakvalues[i]
                        nois_i[noise_count] = peaklocs[i]
                        skip = 1
                        noise_level2 = 0.125 * y_i + 0.875 * noise_level2
                        noise_level1 = 0.125 * peakvalues[i] + 0.875 * noise_level1
                    else:
                        skip = 0
            if skip == 0:
                amount_detected_beats1 += 1
                qrsc[amount_detected_beats1] = peakvalues[i]
                qrsi[amount_detected_beats1] = peaklocs[i]
                if y_i >= signal_threshold2:
                    amount_detected_peaks2 += 1
                    if searchback:
                        qrsiraw[amount_detected_peaks2] = x_i
                    else:
                        qrsiraw[amount_detected_peaks2] = peaklocs[i] - round(mindistance * samplerate) + (x_i - 1)
                    qrs_amplitude[amount_detected_peaks2] = y_i
                    signal_level2 = 0.125 * y_i + 0.875 * signal_level2
                signal_level1 = 0.125 * peakvalues[i] + 0.875 * signal_level1
        elif noise_threshold1 <= peakvalues[i] < signal_threshold1:
            noise_level2 = 0.125 * y_i + 0.875 * noise_level2
            noise_level1 = 0.125 * peakvalues[i] + 0.875 * noise_level1
        elif peakvalues[i] < noise_threshold1:
            noise_count += 1
            nois_c[noise_count] = peakvalues[i]
            nois_i[noise_count] = peaklocs[i]
            noise_level2 = 0.125 * y_i + 0.875 * noise_level2
            noise_level1 = 0.125 * peakvalues[i] + 0.875 * noise_level1
        if noise_level1 != 0 or signal_level1 != 0:
            signal_threshold1 = noise_level1 + 0.25 * abs(signal_level1 - noise_level1)
            noise_threshold1 = 0.5 * signal_threshold1
        if noise_level2 != 0 or signal_level2 != 0:
            signal_threshold2 = noise_level2 + 0.25 * abs(signal_level2 - noise_level2)
            noise_threshold2 = 0.5 * signal_threshold2
        signallevelbuffer1[i] = signal_level1
        noiselevelbuffer1[i] = noise_level1
        thrsbuffer1[i] = signal_threshold1
        signallevelbuffer2[i] = signal_level2
        noiselevelbuffer2[i] = noise_level2
        thrsbuffer2[i] = signal_threshold2
        skip = 0
        searchback = 0
    return qrsi
