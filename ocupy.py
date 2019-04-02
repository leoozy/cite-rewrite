import tensorflow as tf


if __name__ == "__main__":
    a = tf.zeros([110,110])
    sess = tf.Session()
    while (1):
        b = tf.reduce_sum(a)
        sess.run(b)

