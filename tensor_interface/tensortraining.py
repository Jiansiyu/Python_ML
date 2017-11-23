'''
@note: Tensorflow learning code used for analysis the stock price
        
        
        For the initial version, the idea is that: put the 50 days( by default, of cause it is changeable) informations, 
        and predict the next day price. This period may will change according to its peformance

@author: Siyu
@contact: jiansiyu@gmail.com
        
@version: 1.0.0
        
Any suggestions, Contributions is welcome.

'''
#from numpy.distutils.fcompiler import none

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class TensorTraining(object):
    
    def __init__(self):
        print("TensorTraining")
        tf.logging.set_verbosity(tf.logging.INFO)
    
    def TensorRun(self):
        print("Start training program")
    
    def Load_data_train_test(self,filename_input=None):
        if filename_input is None:
            self.DataFileName=['../dataIO/stockinfor.csv']
        else:
            self.DataFileName=filename_input
        
        print("Load Data :", self.DataFileName)
        stock_training = os.path.join(os.path.dirname(__file__), self.DataFileName[0])
        stock_test = os.path.join(os.path.dirname(__file__), '../dataIO/train_test.csv')
        
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=stock_training, target_dtype=np.int, features_dtype=np.float)
        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=stock_test, target_dtype=np.int, features_dtype=np.float)
        print(training_set.data)
        print(test_set.data)
        print(test_set.target)
        validation_metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                        metric_fn=tf.contrib.metrics.streaming_accuracy,
                        prediction_key="classes"),
            "precision":
                tf.contrib.learn.MetricSpec(
                        metric_fn=tf.contrib.metrics.streaming_precision,
                        prediction_key="classes"),
            "recall":
                tf.contrib.learn.MetricSpec(
                        metric_fn=tf.contrib.metrics.streaming_recall,
                        prediction_key="classes")
                              }
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            test_set.data,
            test_set.target,
            every_n_steps=500,
            metrics=validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=800)
        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=62)]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10, 20, 20,10],
            n_classes=4,
            model_dir="./stock_model",
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))  
        classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=2000,
                 monitors=[validation_monitor])
        
        # Evaluate accuracy.
        print()
        print()
        accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
        print("Accuracy: {0:f}".format(accuracy_score))
    def Load_data_cvs(self,filename_input=None):
        '''
        cvsfile IO interface
        @note: Self adaptence code that used for load the data from csv file
                
                The code will consider all the data in a line to be the data used for the training, and the last data in each line is the result
        '''
        
        if filename_input is None:
            self.DataFileName=['../dataIO/stockinfor.csv']
        else:
            self.DataFileName=filename_input
        print("Load Data :", self.DataFileName)
        
        filename_queue=tf.train.string_input_producer(self.DataFileName)
        reader=tf.TextLineReader()
        key, value=reader.read(filename_queue)
        tf.Print(key,[key])
        record_defaults = []
        for i in range(64):
            record_defaults.append([0.0])
        tensor_array_raw=tf.decode_csv(value, record_defaults=record_defaults)
        features =tf.stack(tensor_array_raw[0:-1])
        init_op = tf.global_variables_initializer()  
        local_init_op = tf.local_variables_initializer()  
        with tf.Session() as sess: 
            sess.run(init_op)
            sess.run(local_init_op)
            
            coord = tf.train.Coordinator()  
            threads = tf.train.start_queue_runners(coord=coord) 
            lable=0
            for _ in range(1000):
                try:
                    example, label = sess.run([features,tensor_array_raw[-1]])  
                    lable=lable+1
                    print(example)
                    print(lable)
                    #print(example,label)
                except tf.errors.OutOfRangeError:  
                    print ('Done !!!')
            '''while True: 
                try:
                    example, label = sess.run([features,tensor_array_raw[-1]])  
                    lable=lable+1
                    print(lable)
                    #print(example,label)
                except tf.errors.OutOfRangeError:  
                    break'''
                
         
            coord.request_stop()
            coord.join(threads=threads)
            sess.close()
if __name__ == '__main__':
    
    test = TensorTraining();
    #test.Load_data_cvs()
    test.Load_data_train_test()
    #test.Load_data_test()
    
    