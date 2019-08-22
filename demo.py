# coding:utf-8
# @Author:Aliked
from Data_Collect import data_collect
from HS300_data import HS300
from ProID_Cal import ID_cal
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

class model():
    # 股票行情数据分析
    def demo(self):
        # 加载文件名读取需要处理的数据信息
        stock_code = data_collect().load_path(filepath="Pro_Stock.xlsx")
        # 获取74支股票实情数据:time_queue为获取数据的时间区，即12、6、3、1；
        engine = create_engine('mysql+pymysql://deo:deo135@$^@192.168.0.12:3306/deo_quote', pool_pre_ping=True)
        close, high, low = data_collect().collectDATA(time_queue=12, stock_code=stock_code, engine=engine)
        # 获取HS300基准组合数据
        HS_colse = HS300().load_data(start_dt='2018-08-15', end_dt='2019-08-15')
        # 测试案例-244个交易日
        # file = open('HS300_data/HS300_收盘价.csv')
        # HS_colse = np.mat(pd.read_csv(file))[:, 1:301]
        # 计算五大类指标(21个)
        id = ID_cal()
        id.load_data(close, high, low, HS_colse)
        id.merge_ID()

if __name__ == '__main__':
    demo = model()
    demo.demo()