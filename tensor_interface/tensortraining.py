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
from time import sleep

# disable the debug display for the TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import tensorflow as tf
import csv
# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 44
# config.inter_op_parallelism_threads = 44
# tf.Session(config=config)

tf.logging.set_verbosity(tf.logging.ERROR)

import logging
logging.getLogger().setLevel(logging.INFO)

class TensorTraining(object):
    
    def __init__(self,train_dataset_csvfilename=None,test_dataset_csv_filename=None,training_model_dir=None):
        print("TensorTraining")
        
        
        # load the test and train data set 
        self.Train_csvfilname='../dataIO/train_dataset.csv'
        if train_dataset_csvfilename is not None:
            self.Train_csvfilname=train_dataset_csvfilename
            
        self.Test_csvfilename='../dataIO/test_dataset.csv'
        if test_dataset_csv_filename is not None:
            self.Test_csvfilename=test_dataset_csv_filename
            
        self.TrainingModel_SaveDir="../Data/TrainingModel"
        if training_model_dir is not None:
            self.TrainingModel_SaveDir=training_model_dir
    def TensorRun(self):
        print("Start training program")
    
    def Load_data_train_test(self,train_dataset_filename=None,test_dataset_filename=None,evaluatedata_array=None,evaluatedata_csvfile=None,training_model_dir=None):
        '''
        @note: 
        
        @var var: 
        
        '''
        print('Start Training')
        train_dataset_filename_local=self.Train_csvfilname
        if train_dataset_filename is not None:
            train_dataset_filename_local=train_dataset_filename
        
        test_dataset_filename_local=self.Test_csvfilename
        if test_dataset_filename is not None:
            test_dataset_filename_local=test_dataset_filename
        
        TrainingModel_SaveDir_local="../Data/TrainingModel"
        if self.TrainingModel_SaveDir is not None:
            TrainingModel_SaveDir_local=self.TrainingModel_SaveDir
        if training_model_dir is not None:
            TrainingModel_SaveDir_local=training_model_dir
            
        print(" >>Training csv file: ",train_dataset_filename_local)
        print(" >>Test csv file: ",test_dataset_filename_local)
        
        stock_training = os.path.join(os.path.dirname(__file__), train_dataset_filename_local)
        stock_test = os.path.join(os.path.dirname(__file__), test_dataset_filename_local)
        
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=stock_training, target_dtype=np.int, features_dtype=np.float)
        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=stock_test, target_dtype=np.int, features_dtype=np.float)
        
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
            early_stopping_rounds=4000)
        
        # Specify that all features have real-value data
        
        
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(training_set.data[0]))]

        # Build 3 layer DNN with 10, 20, 10 units respectively.
        classifier = tf.contrib.learn.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[64,64,64,64,64,64,64,64,64,64,64,64,128,128,128,128,128],
            n_classes=21,
            model_dir=TrainingModel_SaveDir_local,
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))  
        classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=4000,
                 monitors=[validation_monitor])
        
        # Evaluate accuracy.
        accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
        print("Accuracy: {0:f}".format(accuracy_score))
        
        #predict the given array
        predictresult_list=[]
        
        if evaluatedata_csvfile is not None:
            samples_test = os.path.join(os.path.dirname(__file__), evaluatedata_csvfile)
            samples_test = tf.contrib.learn.datasets.base.load_csv_with_header(
                    filename=samples_test, target_dtype=np.int, features_dtype=np.float)
            new_samples=np.array(samples_test.data,dtype=float)
            predictresult_list = list(classifier.predict(new_samples))
            
            loop_counter_temp=0
            evaluation_result_array=[]
            while loop_counter_temp < len(predictresult_list):
                evaluation_result_array.append([samples_test.target[loop_counter_temp],predictresult_list[loop_counter_temp]])
                loop_counter_temp=loop_counter_temp+1
                print(evaluation_result_array)
            return accuracy_score,evaluation_result_array
                
            ''' if evaluatedata_array is not None or evaluatedata_csvfile is not None:
            if evaluatedata_array is not None:
                new_samples=np.array(evaluatedata_array.data,dtype=float)
                predictresult_list = list(classifier.predict(new_samples))
            if evaluatedata_csvfile is not None:
                samples_test = os.path.join(os.path.dirname(__file__), evaluatedata_csvfile)
                samples_test = tf.contrib.learn.datasets.base.load_csv_with_header(
                    filename=samples_test, target_dtype=np.int, features_dtype=np.float)
                new_samples=np.array(samples_test.data,dtype=float)
                predictresult_list = list(classifier.predict(new_samples))
        return accuracy_score, predictresult_list'''
    
        '''new_samples=np.array(test_set.data,dtype=float)
        save_data=[]
        for i in list(classifier.predict(new_samples)):
            temp=[i]
            save_data.append(temp)
        with open('output.csv','w') as csvfileoutput:
            writer = csv.writer(csvfileoutput,lineterminator='\n')
            writer.writerows(save_data)'''    
    def Predicted_result(self):
        '''
        '''
        print('loading the training data and predict the result by given array')
        
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
        for _ in range(64):
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
            for _ in range(1000):
                try:
                    example, label = sess.run([features,tensor_array_raw[-1]])  
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
    
    