# coding:utf-8
import pandas as pd
import tushare as ts
import numpy as np

class HS300:
    # HS300-数据获取(基准组合)
    def load_data(self, start_dt, end_dt):
        '''
           加载全部股票数据
        '''
        read_file = pd.read_excel('HS300_data/HS300.xlsx')
        ts_code_list = list((read_file['股票代码'].dropna()))
        print('run：加载HS300数据')
        stock_colse = self.file_handle(ts_code_list, start_dt, end_dt)
        HS300_close = np.mat(stock_colse)
        return HS300_close

    def get_ts_data(self, single_code, start_dt, end_d):
        # 获取数据
        self.df = ts.get_hist_data(single_code, start=start_dt, end=end_d)
        return self.df

    def file_handle(self, ts_code_list, start_dt, end_dt):
        stock_colse = pd.DataFrame()
        count = 0
        for i in range(len(ts_code_list)):
            single_code = str(round(ts_code_list[i]))
            if len(single_code) == 1:
                single_code = '00000' + str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            if len(single_code) == 2:
                single_code = '0000' + str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            if len(single_code) == 3:
                single_code = '000' + str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            if len(single_code) == 4:
                single_code = '00' + str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            if len(single_code) == 5:
                single_code = '0' + str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            if len(single_code) == 6:
                single_code = str(single_code)
                self.get_ts_data(single_code, start_dt, end_dt)
            else:
                print('数据加载错误！')
                break
            count += 1
            print(count, 'run：', single_code)
            # 获取每支HS股票的收盘价
            stock_colse[i] = self.df['close']
        return stock_colse

    def to_csv(self, filedata, code_name):
        # 文件存储
        try:
            tofile = pd.DataFrame(filedata)
            filename = 'HS300_data/' + code_name + '.csv'
            tofile.to_csv(filename)
        except IOError as ie:
            print(ie)