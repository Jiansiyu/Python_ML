#from inspect import currentframe
#import sys
#from gevent.hub import sleep
import csv

class data_preprocess(object):
    def __init__(self,buffersize=None):
        """
        @author: Siyu
        @copyright
        
        """
        self.DataBuffer=[]
        self.DataBuffer_all=[]
        #load the buffer size
        if buffersize is None:
            self.DataBuffersize=50+1
        else:
            self.DataBuffersize=buffersize
        self.ItemIndex=['Open','High','Low','Close','Volume','pecentage_change','pecentage_change_0.005','range','int_range']
        
    def cvsData_preprocess(self,filename_in=None):
        #read the nasdaq componies name 
        
        Stock_list_filename='googlefinance.csv'
        if filename_in is not None:
            Stock_list_filename=filename_in
        data_path='./'
        
        with open(data_path+Stock_list_filename) as Stock_list_io:
            Stock_list_lines=Stock_list_io.readlines()
        # get the information position  
        
        self.Stock_header_infor=Stock_list_lines[0].split(',')
        #self.time_position =  self.Stock_header_infor.index('Date')
        self.Open_position=self.Stock_header_infor.index('Open')
        self.High_position=self.Stock_header_infor.index('High')
        self.Low_position=self.Stock_header_infor.index('Low')
        self.Close_position=self.Stock_header_infor.index('Close')
        self.Volume_position=self.Stock_header_infor.index('Volume')
        #self.pecentage_change_position=self.Stock_header_infor.index('pecentage_change')
        #self.range_position =self.Stock_header_infor.index('range')
        #self.range_position =self.Stock_header_infor.index('int_range\n')
        #range_change_position=Stock_header_infor.index('range')
        # get the individual nasdaq conpany name
        data_buffer_all_withoutheader=[]
        data_header=[]
        for Single_line in Stock_list_lines[1:]:
            self.Stock_infor=Single_line.split(',')
            data_buffer_temp , training_result =self.__Databuffer(self.Stock_infor[:],6)
            if data_buffer_temp is not None:
                data_buffer_temp.append(training_result)
                print(len(data_buffer_temp))
                data_buffer_all_withoutheader.append(data_buffer_temp)
        data_header.append(len(data_buffer_all_withoutheader))
        data_header.append(len(data_buffer_all_withoutheader[0])-1)
        self.DataBuffer_all.append(data_header)
        for i in data_buffer_all_withoutheader:
            self.DataBuffer_all.append(i)
        #self.Save_csv()
    def __Databuffer(self,singleday_data,buffer_size=None):
        '''
        
        '''
        if buffer_size is None:
            self.Buffersize_local=self.DataBuffersize
        else:
            self.Buffersize_local=buffer_size+1
        if singleday_data is None:
            raise Exception('The input data should not be zero [%S]',__name__);
        else:
            if len(self.DataBuffer) >= self.Buffersize_local*len(singleday_data):
                del self.DataBuffer[0:len(singleday_data)]
                for data in singleday_data:
                    self.DataBuffer.append((data.split()[0]))
            else:
                for data in singleday_data:
                    self.DataBuffer.append((data.split()[0]))

        '''
        @attention:  generate the result used for machine learning
        '''
        
        if len(self.DataBuffer) >= self.Buffersize_local*len(singleday_data):
            #training_result_1=(self.DataBuffer)[(self.Buffersize_local-1)*len(singleday_data)+self.ItemIndex.index('Close')]
            #training_result_2=(self.DataBuffer)[(self.Buffersize_local-2)*len(singleday_data)+self.ItemIndex.index('Close')]
            #training_result=(float(training_result_1)-float(training_result_2))/float(training_result_2)
            #print(self.DataBuffer)
            training_result=self.DataBuffer[-1]
            #print(training_result)
            #print(self.DataBuffer[0:-2])
            return self.DataBuffer[0:-2],training_result
        else:
            return None, None
        
        
    def Save_csv(self,filenameSave=None):
        
        '''
        
        '''
        
        if filenameSave is None:
            print('The outpt file will save as defult filename')
            self.filenameSave='stockinfor.csv'
        else:
            self.filenameSave=filenameSave
        with open(self.filenameSave,'w') as csvfileoutput:
            writer = csv.writer(csvfileoutput,lineterminator='\n')
            writer.writerows(self.DataBuffer_all)
    def Print(self):
        print(self.DataBuffer)
        #sleep(1)
if __name__ == '__main__':
    test=data_preprocess()
    test.cvsData_preprocess()
    test.Save_csv('train_dataset.csv')
    test_date=data_preprocess()
    test_date.cvsData_preprocess(filename_in='stock_test.csv')
    test_date.Save_csv('test_dataset.csv')
    