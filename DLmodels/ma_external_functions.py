import tensorflow as tf

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

