from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from gesture_shuffle_the_h5py import shuffle_batch_seq, trainPart, testPart

train_epoch = 840
display_step = 100
n_classes = 7
batch_num = 7
seq_len = 64
channel = 1

vali_step = 10
total_step = train_epoch*20

testpredmatrix_add = []
test_truelabelmatrix_add = []

# define placeholders
#xs = tf.placeholder(tf.float32, [None, None, 64, 32, None])   # batch_num, seq_len, 64x32, channel
xs = tf.placeholder(tf.float32, [None, None, 32, 32, None])   # batch_num, seq_len, 64x32, channel
ys = tf.placeholder(tf.float32, [None, None, n_classes]) # batch_num, seq_len, 7 classes
conv2_dropout = tf.placeholder(tf.float32)
conv3_dropout = tf.placeholder(tf.float32)
fc4_dropout = tf.placeholder(tf.float32)
fc5_dropout = tf.placeholder(tf.float32)
fc7_dropout = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')

def BiRNN(x, batch_num, n_hidden, output_size):
                     
    # Define a lstm cell with tensorflow
    lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    
    init_state_fw = lstm_cell_fw.zero_state(batch_num, dtype=tf.float32)
    init_state_bw = lstm_cell_bw.zero_state(batch_num, dtype=tf.float32)
        
    output, states = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x,
                                                     initial_state_fw=init_state_fw,
                                                     initial_state_bw=init_state_bw,
                                                     time_major=False)    

    # Linear activation, using rnn inner loop last output
    return tf.reshape(tf.concat(output,1), [-1, n_hidden*2])

x_image = tf.reshape(xs, [-1, 32, 32, channel])
#x_image = tf.reshape(xs, [-1, 2048, channel])
# print(x_image.shape)  # [n_samples, 32, 32, channel]

## conv1 layer ##
W_conv1 = weight_variable([3, 3, channel,32]) # patch 3x3, in size 4, out size 32
b_conv1 = bias_variable([32])
conv1 = conv2d(x_image, W_conv1) + b_conv1

mean1, var1 = tf.nn.moments(W_conv1, [0, 1, 2])
beta1 = tf.Variable(tf.zeros([32]))
gamma1 = tf.Variable(tf.ones([32]))
bn_conv1 = tf.nn.batch_normalization(conv1, mean1, var1, beta1, gamma1, 1e-3)

h_conv1 = tf.nn.relu(bn_conv1) # output size 30x30x32


## conv2 layer ##
W_conv2 = weight_variable([3, 3, 32, 64]) # patch 3x3, in size 32, out size 64
b_conv2 = bias_variable([64])
conv2 = conv2d(h_conv1, W_conv2) + b_conv2

mean2, var2 = tf.nn.moments(W_conv2, [0, 1, 2])
beta2 = tf.Variable(tf.zeros([64]))
gamma2 = tf.Variable(tf.ones([64]))
bn_conv2 = tf.nn.batch_normalization(conv2, mean2, var2, beta2, gamma2, 1e-3)

h_conv2 = tf.nn.relu(bn_conv2) # output size 28x28x64
h_conv2 = tf.nn.dropout(h_conv2, conv2_dropout)


## conv3 layer ##
W_conv3 = weight_variable([3, 3, 64, 128]) # patch 3x3, in size 64, out size 128
b_conv3 = bias_variable([128])
conv3 = conv2d(h_conv2, W_conv3) + b_conv3

mean3, var3 = tf.nn.moments(W_conv3, [0, 1, 2])
beta3 = tf.Variable(tf.zeros([128]))
gamma3 = tf.Variable(tf.ones([128]))
bn_conv3 = tf.nn.batch_normalization(conv3, mean3, var3, beta3, gamma3, 1e-3)

h_conv3 = tf.nn.relu(bn_conv3) # output size 26x26x128
h_conv3 = tf.nn.dropout(h_conv3, conv3_dropout)

# [n_samples, 26, 26, 128] ->> [n_samples, 26*26*128]
h_flat = tf.reshape(h_conv3, [-1, 26*26*128])

## fc4 layer ##
W_fc4 = weight_variable([26*26*128, 512])
b_fc4 = bias_variable([512])
h_fc4 = tf.nn.relu(tf.matmul(h_flat, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, fc4_dropout)

## fc5 layer ##
W_fc5 = weight_variable([512, 512])
b_fc5 = bias_variable([512])
h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)
h_fc5_drop = tf.nn.dropout(h_fc5, fc5_dropout)

h_fc5_reshaped = tf.reshape(h_fc5_drop, [-1, seq_len, 512])
h_lstm6 = BiRNN(h_fc5_reshaped, batch_num, n_hidden=512, output_size=1024)

W_fc7 = weight_variable([1024, n_classes])
b_fc7 = bias_variable([n_classes])
h_fc7 = tf.matmul(h_lstm6, W_fc7) + b_fc7
y_pred = tf.nn.dropout(h_fc7, fc7_dropout)

y_pred = tf.reshape(y_pred, [-1, n_classes])
ys_reshaped = tf.reshape(ys, [-1, n_classes])
                  
# the error between prediction and real data
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys_reshaped, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

update_train = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # to make sure that the train keeps updating
with tf.control_dependencies(update_train):
    train_step = optimizer.minimize(cross_entropy)
    
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(ys_reshaped,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    
    result_txt = open('gesture_bilstm_0522_non-normed_32x32_corrected orientation.txt' ,'w')
    sess.run(init_op)

    for i in range(total_step + 1):
        
        start_time = time.time()
        
        trainData_batch, trainLabel_batch = shuffle_batch_seq(batch_num, seq_len, trainPart)
#        sess.run(train_step, feed_dict={xs:trainData_batch, ys:trainLabel_batch,
#                                        conv2_dropout: 0.4, conv3_dropout: 0.4,
#                                        fc4_dropout: 0.5, fc5_dropout: 0.5, 
#                                        phase_train: True, learning_rate: 1e-3})
        
        if i <= train_epoch*10:
            sess.run(train_step, feed_dict={xs:trainData_batch, ys:trainLabel_batch,
                                        conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
                                        phase_train: True, learning_rate: 1e-3})   
                                        
        elif train_epoch*15 >= i > train_epoch*10:
            sess.run(train_step, feed_dict={xs:trainData_batch, ys:trainLabel_batch,
                                        conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
                                        phase_train: True, learning_rate: 1e-4})
                                        
        elif i > train_epoch*15:
            sess.run(train_step, feed_dict={xs:trainData_batch, ys:trainLabel_batch,
                                        conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
                                        phase_train: True, learning_rate: 1e-5})
        
        train_accuracy = sess.run(accuracy, feed_dict={
                    xs:trainData_batch, ys:trainLabel_batch,
                    conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
#                    state_drop: 1.0,
                    phase_train: False
                    })
    
        train_loss = sess.run(cross_entropy, feed_dict={
                    xs:trainData_batch, ys:trainLabel_batch,
                    conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
#                    state_drop: 1.0,
                    phase_train: False
                    })
    
        print("step %d" % i)
        print("train accuracy: %g, train loss: %g\n" % (train_accuracy, train_loss)) 
        
        if i % vali_step == 0:
            duration = time.time() - start_time
            
            testData_batch, testLabel_batch = shuffle_batch_seq(batch_num, seq_len, testPart)
#            testPart = switch_data_order(batch_num, testPart, testPart_origin)     
            
            vali_accuracy = sess.run(accuracy, feed_dict={
                    xs:testData_batch, ys:testLabel_batch,
                    conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
#                    state_drop: 1.0,
                    phase_train: False
                    })
    
            vali_loss = sess.run(cross_entropy, feed_dict={
                    xs:testData_batch, ys:testLabel_batch,
                    conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
#                    state_drop: 1.0,
                    phase_train: False
                    })  
    
                                 
            print("vali accuracy: %g, vali loss: %g" % (vali_accuracy, vali_loss))
            print("train time elapsed until current step: %.3f sec\n" % duration)
            
        if i > train_epoch*14 :
        
            testpredmatrix = sess.run(tf.argmax(y_pred,1), feed_dict={xs:testData_batch, ys:testLabel_batch,
                        conv2_dropout: 0.6, conv3_dropout: 0.6,
                                        fc4_dropout: 0.5, fc5_dropout: 0.5,
                                        fc7_dropout: 0.5,
                    phase_train: False})
                
            testpredmatrix_add = np.append(testpredmatrix_add, testpredmatrix , axis=0)        
    

            testlabelmatrix = sess.run( ys_reshaped,feed_dict={ys:testLabel_batch})
            
            test_truelabelmatrix = sess.run(tf.argmax(testlabelmatrix,1))
                
            test_truelabelmatrix_add = np.append(test_truelabelmatrix_add, test_truelabelmatrix  , axis=0)
                
                
                
                
            test_confusion_matrix_result =  confusion_matrix(test_truelabelmatrix_add ,testpredmatrix_add ,labels=[0,1,2,3,4,5,6])
            
            matrix = test_confusion_matrix_result.astype(float)
            
            test_matrixresult_normal =  test_confusion_matrix_result / matrix.sum(axis=1)[:, np.newaxis]
            
        result_txt.write("step %d\n" % i)
        result_txt.write("train accuracy: %g, train loss: %g\n" % (train_accuracy, train_loss))
        result_txt.write("vali accuracy: %g, vali loss: %g\n\n" % (vali_accuracy, vali_loss))            
    result_txt.close()
