import tensorflow as tf

print("gpu available : ", len(tf.config.experimental.list_physical_devices('GPU')))
