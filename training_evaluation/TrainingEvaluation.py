'''
@note: Training efficency evaluation
       The initial motivation is that training the data and evaluate the new data, 
       then put the new data back and retraining the model
@author: Siyu
@contact: jiansiyu@gmail.com
          GO WAHOO !!!

@version: 1.0.0
@todo:    input data file, generate training data
          the training data will be used for training and eveluation
          the evaluation result will be saved in a file
'''
import csv
import time
import sys
from time import sleep
sys.path.append('../dataIO')
sys.path.append('../tensor_interface')
from tensortraining import TensorTraining
from data_preprocess import data_preprocess

class TrainingEvaluation(object):
    def __init__(self,evaluation_step=None,training_timelength=None,raw_csv_data_file=None,evaluation_result=None):
        
        '''
        
        @var evaluation_step: time interval before training again, by default the time inteval is 1 (days)
             training_timelength: training data length 1000 days by default
        
        '''
        
        print('Training evaluation function')
        
        # load the evaluation step
        if evaluation_step is None:
            self.Evaluation_step=1
        else:
            self.Evaluation_step=evaluation_step
        
        if training_timelength is None:
            self.Training_timelength=1000
        else:
            self.Training_timelength=training_timelength
        
        if raw_csv_data_file is None:
            self.raw_csv_date_file="../dataIO/googlefinance.csv"
        else:
            self.raw_csv_date_file=raw_csv_data_file
        
        self.Evaluation_result="../Data/TrainingEvaluation/evaluation_result.csv"
        if evaluation_result is not None:
            self.Evaluation_result=evaluation_result    
            
    def __Get_single_csv_data(self,training_timelength_begin,training_timelength=None,raw_csv_data_filename_in=None,output_filename=None):
        
        '''
        @note: in order to get each step training data, in this step,
                the function will generate each files with the predefined time
                intervals
        @var : training_timelength_begin: startline number (same as time)
               training_timelength:   time inteval 1000 by desault
               filename_in           : csv raw data file name
               output_filename       : output csv data filename 
        @return: output csv filename
                 True : this is not the end
                 False: the file reach the end
        '''
        
        raw_csv_data_filename_local=self.raw_csv_date_file
        if raw_csv_data_filename_in is not None:
            raw_csv_data_filename_local=raw_csv_data_filename_in
        
        training_timelength_local=self.Training_timelength
        if training_timelength is not None:
            training_timelength_local=training_timelength
        
        output_filename_local='../Data/tmp/temp.csv'
        if output_filename is not None:
            output_filename_local=output_filename
        
        # read the csv file
        with open(raw_csv_data_filename_local) as Stock_list_io:
            Stock_list_lines=Stock_list_io.readlines()
        csv_header_save_temp=[]
        for i in (Stock_list_lines[0]).split(','):
            csv_header_save_temp.append(i.split()[0])
        
        csv_save_buffer=[]
        csv_save_buffer.append(csv_header_save_temp)
        #print( training_timelength_begin+training_timelength_local)
        #print(len(Stock_list_lines)+1)
        
        if training_timelength_begin+training_timelength_local < len(Stock_list_lines)+1:
            for Single_line in Stock_list_lines[training_timelength_begin+1:training_timelength_begin+training_timelength_local+1]:
                Single_line_temp=Single_line.split(',')
                SingleLine_save_temp=[]
                for i in Single_line_temp:
                    SingleLine_save_temp.append((i.split())[0])
                
                csv_save_buffer.append(SingleLine_save_temp)
        else:
            for Single_line in Stock_list_lines[training_timelength_begin:]:
                #print(Single_line)
                Single_line_temp=Single_line.split(',')
                SingleLine_save_temp=[]
                for i in Single_line_temp:
                    SingleLine_save_temp.append((i.split())[0])
                csv_save_buffer.append(SingleLine_save_temp)
        
        with open(output_filename_local,'w') as csvfileoutput:
            writer = csv.writer(csvfileoutput,lineterminator='\n')
            writer.writerows(csv_save_buffer)
         
        if training_timelength_begin+training_timelength_local >= len(Stock_list_lines)+1:
            return output_filename_local,False            
        else:
            return output_filename_local,True
            
    def Run(self,evaluation_step=None,training_timelength=None,raw_csv_data_file=None):
        '''
        '''
        Evaluation_step_local=self.Evaluation_step
        if evaluation_step is not None:
            Evaluation_step_local=evaluation_step
            
        Training_timelength_local=self.Training_timelength
        if training_timelength is not None:
            Training_timelength_local=training_timelength
            
        raw_csv_date_file_local=self.raw_csv_date_file
        if raw_csv_data_file is not None:
            raw_csv_date_file_local=raw_csv_data_file
        
        time_step_counter=0
        
        Evaluation_array_all=[]  # buffer the evaluation result [[real, expected]]
        fileEnd_flag_temp=True
        
        evaluation_csvresult_filename='../Data/TrainingEvaluation/evaluationResult'+str(time.time())+'.csv'
        while fileEnd_flag_temp:
            '''
            generate the training and the test data 
            tranning the model 
            '''
            # loop to the next data point
            time_step_counter=time_step_counter+1
            train_raw_csv_filename_temp, fileEnd_flag_temp=self.__Get_single_csv_data(time_step_counter,training_timelength=510)
            
            train_dataset_filename='../Data/tmp/train_dataset_temp'+ str(time.time())+'.csv'
            test_dataset_filename='../Data/tmp/test_dataset_temp'+ str(time.time())+'.csv'
            evaluation_dataset_filename='../Data/tmp/evaluation_dataset_temp'+ str(time.time())+'.csv'
            trainModel_dir='../Data/TrainingModel/'+str(time.time())
            
            csv_preprocess=data_preprocess(raw_csv_date_file_local)
            csv_preprocess.cvsData_preprocess(train_raw_csv_filename_temp)
            csv_preprocess.Save_csv_training_test(trainFilename=train_dataset_filename,evaluationFilename=evaluation_dataset_filename)
            
            csv_preprocess1=data_preprocess(raw_csv_date_file_local)
            csv_preprocess1.cvsData_preprocess(train_raw_csv_filename_temp)
            csv_preprocess1.Save_csv(test_dataset_filename)
            # finish generate the training data
            
            #train_model=TensorTraining(train_dataset_csvfilename=train_dataset_filename,test_dataset_csv_filename=test_dataset_filename,training_model_dir=trainModel_dir)
            train_model=TensorTraining(train_dataset_csvfilename=train_dataset_filename,test_dataset_csv_filename=test_dataset_filename)
            accuracy,evaluation_result_temp=train_model.Load_data_train_test(evaluatedata_csvfile=evaluation_dataset_filename)
            for i in evaluation_result_temp:
                Evaluation_array_all.append(i)
                print(Evaluation_array_all)
                with open(evaluation_csvresult_filename,'w') as evaluationresult_io:
                    writer = csv.writer(evaluationresult_io,lineterminator='\n')
                    writer.writerows(Evaluation_array_all)
if __name__ == "__main__":
    test=TrainingEvaluation()
    test.Run()