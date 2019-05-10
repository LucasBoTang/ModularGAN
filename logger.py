import tensorflow as tf


class Logger(object):
    """
    tensorboard logger
    """

    def __init__(self, log_dir):
        """
        initialize summary writer
        """
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """
        add scalar summary
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
