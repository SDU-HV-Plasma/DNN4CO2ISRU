import time
start = time.time()
import tensorflow.compat.v1 as tf
import tensorflow as tf1
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import csv

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
#Activation_function
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#Training dataset
Traindata = pd.read_csv('Training dataset.CSV',encoding='gbk')

#Pulse rise phase
T0=Traindata[Traindata.Time>=0]
T1=T0[T0.Time<0.3]
X_rise=T1[['Time','Voltage amplitude']]
X_rise_data=np.array(X_rise,dtype='float32')
Y_rise=T1[['Current density']]
Y_rise_data=np.array(Y_rise,dtype='float32')

#Plateau phase
T3=Traindata[Traindata.Time>=0.3]
T4=T3[T3.Time<0.6]
X_plateau=T4[['Time','Voltage amplitude']]
X_plateau_data=np.array(X_plateau,dtype='float32')
Y_plateau=T4[['Current density']]
Y_plateau_data=np.array(Y_plateau,dtype='float32')

#Pulse fall phase
T5=Traindata[Traindata.Time>=0.6]
T6=T5[T5.Time<0.9]
X_fall=T6[['Time','Voltage amplitude']]
X_fall_data=np.array(X_fall,dtype='float32')
Y_fall=T6[['Current density']]
Y_fall_data=np.array(Y_fall,dtype='float32')

#Power-off phase
T7=Traindata[Traindata.Time>=0.9]
T8=T7[T7.Time<=0.95]
X_off=T8[['Time','Voltage amplitude']]
X_off_data=np.array(X_off,dtype='float32')
Y_off=T8[['Current density']]
Y_off_data=np.array(Y_off,dtype='float32')

#Testing dataset
Testdata = pd.read_csv('Testing dataset.CSV',encoding='gbk')

#Pulse rise phase
T0_t=Traindata[Traindata.Time>=0]
T1_t=T0_t[T0_t.Time<0.3]
X_rise_t=T1_t[['Time','Voltage amplitude']]
X_rise_data_t=np.array(X_rise_t,dtype='float32')
Y_rise_t=T1_t[['Current density']]
Y_rise_data_t=np.array(Y_rise_t,dtype='float32')

#Plateau phase
T3_t=Traindata[Traindata.Time>=0.3]
T4_t=T3_t[T3_t.Time<0.6]
X_plateau_t=T4_t[['Time','Voltage amplitude']]
X_plateau_data_t=np.array(X_plateau_t,dtype='float32')
Y_plateau_t=T4_t[['Current density']]
Y_plateau_data_t=np.array(Y_plateau_t,dtype='float32')

#Pulse fall phase
T5_t=Traindata[Traindata.Time>=0.6]
T6_t=T5_t[T5_t.Time<0.9]
X_fall_t=T6_t[['Time','Voltage amplitude']]
X_fall_data_t=np.array(X_fall_t,dtype='float32')
Y_fall_t=T6_t[['Current density']]
Y_fall_data_t=np.array(Y_fall_t,dtype='float32')

#Power-off phase
T7_t=Traindata[Traindata.Time>=0.9]
T8_t=T7_t[T7_t.Time<=0.95]
X_off_t=T8_t[['Time','Voltage amplitude']]
X_off_data_t=np.array(X_off_t,dtype='float32')
Y_off_t=T8_t[['Current density']]
Y_off_data_t=np.array(Y_off_t,dtype='float32')

#Input layer
Input1 = tf.placeholder(tf.float32, [None,2])
Output1 = tf.placeholder(tf.float32, [None,1])
Input2 = tf.placeholder(tf.float32, [None,2])
Output2 = tf.placeholder(tf.float32, [None,1])
Input3 = tf.placeholder(tf.float32, [None,2])
Output3 = tf.placeholder(tf.float32, [None,1])
Input4 = tf.placeholder(tf.float32, [None,2])
Output4 = tf.placeholder(tf.float32, [None,1])

#Hidden layers
Hidden1_1 = add_layer(Input1, 2, 30, activation_function=tf1.nn.relu)
Hidden2_1 = add_layer(Hidden1_1, 30, 30, activation_function=tf1.tanh)
Hidden3_1 = add_layer(Hidden2_1, 30, 30, activation_function=tf1.tanh)
Hidden4_1 = add_layer(Hidden3_1, 30, 30, activation_function=tf1.sigmoid)
Hidden1_2 = add_layer(Input2, 2, 30, activation_function=tf1.nn.relu)
Hidden2_2 = add_layer(Hidden1_2, 30, 30, activation_function=tf1.tanh)
Hidden3_2 = add_layer(Hidden2_2, 30, 30, activation_function=tf1.tanh)
Hidden4_2 = add_layer(Hidden3_2, 30, 30, activation_function=tf1.sigmoid)
Hidden1_3 = add_layer(Input3, 2, 30, activation_function=tf1.nn.relu)
Hidden2_3 = add_layer(Hidden1_3, 30, 30, activation_function=tf1.tanh)
Hidden3_3 = add_layer(Hidden2_3, 30, 30, activation_function=tf1.tanh)
Hidden4_3 = add_layer(Hidden3_3, 30, 30, activation_function=tf1.sigmoid)
Hidden1_4 = add_layer(Input4, 2, 30, activation_function=tf1.nn.relu)
Hidden2_4 = add_layer(Hidden1_4, 30, 30, activation_function=tf1.tanh)
Hidden3_4 = add_layer(Hidden2_4, 30, 30, activation_function=tf1.tanh)
Hidden4_4 = add_layer(Hidden3_4, 30, 30, activation_function=tf1.sigmoid)

#Output layer
Pre1 = add_layer(Hidden4_1, 30, 1, activation_function=None)
Pre2 = add_layer(Hidden4_2, 30, 1, activation_function=None)
Pre3 = add_layer(Hidden4_3, 30, 1, activation_function=None)
Pre4 = add_layer(Hidden4_4, 30, 1, activation_function=None)

#Loss_function
Loss_function1 = tf.reduce_mean(tf.square(Output1 - Pre1))
Loss_function2 = tf.reduce_mean(tf.square(Output2 - Pre2))
Loss_function3 = tf.reduce_mean(tf.square(Output3 - Pre3))
Loss_function4 = tf.reduce_mean(tf.square(Output4 - Pre4))

#Minimization loss function
Learn_rate1 = tf.train.AdamOptimizer(0.0001).minimize(Loss_function1)
Learn_rate2 = tf.train.AdamOptimizer(0.00001).minimize(Loss_function2)
Learn_rate3 = tf.train.AdamOptimizer(0.00001).minimize(Loss_function3)
Learn_rate4 = tf.train.AdamOptimizer(0.00001).minimize(Loss_function4)

#initialization
init = tf.global_variables_initializer()

#Training
with tf.Session() as sess:
    sess.run(init)
    is_train = False
    is_mod1 = False
    is_mod2 = True
    saver = tf.train.Saver(max_to_keep=1)
    if is_train:
      if is_mod1:
       if is_mod2:
        # model_file1 = tf1.train.latest_checkpoint('save1/')
        # saver.restore(sess, model_file1)
        for i in range(10000):
         sess.run(Learn_rate1, feed_dict={Input1: X_rise_data, Output1: Y_rise_data})
         if i % 100 == 0:
          print(sess.run(Loss_function1, feed_dict={Input1: X_rise_data, Output1: Y_rise_data}), (i/10000)*100, '%')
        saver.save(sess, 'save1/model1', global_step=i + 1)
       else:
           # model_file2 = tf1.train.latest_checkpoint('save2/')
           # saver.restore(sess, model_file2)
           for i in range(10000):
               sess.run(Learn_rate2, feed_dict={Input2: X_plateau_data, Output2: Y_plateau_data})
               if i % 100 == 0:
                   print(sess.run(Loss_function2, feed_dict={Input2: X_plateau_data, Output2: Y_plateau_data}), (i/10000)*100, '%')
           saver.save(sess, 'save2/model2', global_step=i + 1)
      else:
       if is_mod2:
        # model_file3 = tf1.train.latest_checkpoint('save3/')
        # saver.restore(sess, model_file3)
        for i in range(10000):
         sess.run(Learn_rate3, feed_dict={Input3: X_fall_data, Output3: Y_fall_data})
         if i % 100 == 0:
          print(sess.run(Loss_function3, feed_dict={Input3: X_fall_data, Output3: Y_fall_data}), (i/10000)*100, '%')
        saver.save(sess, 'save3/model3', global_step=i + 1)
       else:
           # model_file4 = tf1.train.latest_checkpoint('save4/')
           # saver.restore(sess, model_file4)
           for i in range(10000):
               sess.run(Learn_rate4, feed_dict={Input4: X_off_data, Output4: Y_off_data})
               if i % 100 == 0:
                   print(sess.run(Loss_function4, feed_dict={Input4: X_off_data, Output4: Y_off_data}), (i/10000)*100, '%')

           saver.save(sess, 'save4/model4', global_step=i + 1)
    else:
#Prediction
        with open("Prediction data.CSV", "w", newline='') as f:
            b_csv = csv.writer(f)
            model_file1 = tf1.train.latest_checkpoint('save1/')
            saver.restore(sess, model_file1)
            print('MSE of Current density1:')
            b_csv.writerows(sess.run(Pre1, feed_dict={Input1: X_rise_data_t}))
            print(sess.run(Loss_function1, feed_dict={Input1: X_rise_data_t, Output1: Y_rise_data_t}))
            model_file2 = tf1.train.latest_checkpoint('save2/')
            saver.restore(sess, model_file2)
            b_csv.writerows(sess.run(Pre2, feed_dict={Input2: X_plateau_data_t}))
            print('MSE of Current density2:')
            print(sess.run(Loss_function2, feed_dict={Input2: X_plateau_data_t, Output2: Y_plateau_data_t}))
            model_file3 = tf1.train.latest_checkpoint('save3/')
            saver.restore(sess, model_file3)
            b_csv.writerows(sess.run(Pre3, feed_dict={Input3: X_fall_data_t}))
            print('MSE of Current density3:')
            print(sess.run(Loss_function3, feed_dict={Input3: X_fall_data_t, Output3: Y_fall_data_t}))
            model_file4 = tf1.train.latest_checkpoint('save4/')
            saver.restore(sess, model_file4)
            b_csv.writerows(sess.run(Pre4, feed_dict={Input4: X_off_data_t}))
            print('MSE of Current density4:')
            print(sess.run(Loss_function4, feed_dict={Input4: X_off_data_t, Output4: Y_off_data_t}))
    time_used = (time.time() - start)/3600
    print('Total time(h):')
    print(time_used)