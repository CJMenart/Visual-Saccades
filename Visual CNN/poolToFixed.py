#adapted from Kaiming He's implemention of spp in Tensorflow
#but for using a 'single-level' pyramid to pool down to a fixed feature map
#Chris menart, 10-21-17

import tensorflow as tf

def poolToFixed(inputs, output_size, mode):

    inputs_shape = tf.shape(inputs)
    b = tf.cast(tf.gather(inputs_shape, 0), tf.int32)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)
    f = tf.cast(tf.gather(inputs_shape, 3), tf.int32)
    
    n = output_size    result = []
    
    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))
    
    for row in range(output_size):
        for col in range(output_size):
            start_h = tf.cast(tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            end_h = tf.cast(tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            start_w = tf.cast(tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            end_w = tf.cast(tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(pooling_region, axis=(1, 2))
            result.append(pool_result)
    #print('Pool shape with pool size %d' % output_size)
    #print(result)
    result = tf.concat(result,1)
    print(result.shape.as_list())
    result = tf.reshape(result,[b,output_size,output_size,f])
    print(result.shape.as_list())
    return result