# -*- coding:utf8 -*-
# @Author: Aliked
import pymysql
import pandas as pd
from numpy import *
from sqlalchemy import create_engine

class data_collect():
    def load_path(self, filepath):
        # 获取股票组合信息
        stock_info = pd.read_excel(filepath)
        stock_list = stock_info['股票代码']
        stock_code = self.name_handle(stock_list)
        return stock_code

    def collectDATA(self, time_queue, stock_code, engine):
        # 初始化数据库连接，使用pymysql模块
        # 查询语句，选出表中的数据
        row = len(stock_code)
        stock_close = [];  stock_high = [];  stock_low = []
        print('run：加载行情数据')
        for i in range(row):
            if time_queue == 12:
                # 1年：244个交易日
                sql = "SELECT TRADINGDAY, HIGHESTPRICE, LOWESTPRICE, ADJUSTCLOSINGPRICE FROM stk_dailyquote a inner join  stk_basicinfo b on a.SECUCODE=b.SECUcode where b.TRADINGCODE='%s' AND a.TRADINGDAY between 20180815 and 20190815" % (stock_code[i])
                sql_df = pd.read_sql_query(sql, engine)
            elif time_queue == 6:
                # 6个月：125个交易日
                sql = "SELECT TRADINGDAY, HIGHESTPRICE, LOWESTPRICE, ADJUSTCLOSINGPRICE FROM stk_dailyquote a inner join  stk_basicinfo b on a.SECUCODE=b.SECUcode where b.TRADINGCODE='%s' AND a.TRADINGDAY between 20190215 and 20190815" % (stock_code[i])
                sql_df = pd.read_sql_query(sql, engine)
            elif time_queue == 3:
                # 3个月：66个交易日
                sql = "SELECT TRADINGDAY, HIGHESTPRICE, LOWESTPRICE, ADJUSTCLOSINGPRICE FROM stk_dailyquote a inner join  stk_basicinfo b on a.SECUCODE=b.SECUcode where b.TRADINGCODE='%s' AND a.TRADINGDAY between 20190515 and 20190815" % (stock_code[i])
                sql_df = pd.read_sql_query(sql, engine)
            elif time_queue == 1:
                # 1个月：24个交易日
                sql = "SELECT TRADINGDAY, HIGHESTPRICE, LOWESTPRICE, ADJUSTCLOSINGPRICE FROM stk_dailyquote a inner join  stk_basicinfo b on a.SECUCODE=b.SECUcode where b.TRADINGCODE='%s' AND a.TRADINGDAY between 20190715 and 20190815" % (stock_code[i])
                sql_df = pd.read_sql_query(sql, engine)
            # 获取收盘价：调整后的价格
            close_df = sql_df['ADJUSTCLOSINGPRICE']
            stock_close.append(close_df)
            # 获取股票最高价
            high_df = sql_df['HIGHESTPRICE']
            stock_high.append(high_df)
            # 获取股票最低价
            low_df = sql_df['LOWESTPRICE']
            stock_low.append(low_df)
            # m, n = shape(mat(close_df))
        stock_close = mat(stock_close).T  # 收盘价
        close = self.data_sort(stock_close)
        stock_high = mat(stock_high).T  # 最高价
        high = self.data_sort(stock_high)
        stock_low = mat(stock_low).T  # 最低价
        low = self.data_sort(stock_low)
        return close, high, low

    def to_csv(self, filedata, filename):
        # 存储数据
        to_file = pd.DataFrame(filedata)
        to_file.to_csv(filename)

    def name_handle(self, ts_code_list):
        # 股票代码处理
        count = 0
        stock_list = []
        for i in range(len(ts_code_list)):
            single_code = str(round(ts_code_list[i]))
            if len(single_code) == 1:
                single_code = '00000' + str(single_code)
            if len(single_code) == 2:
                single_code = '0000' + str(single_code)
            if len(single_code) == 3:
                single_code = '000' + str(single_code)
            if len(single_code) == 4:
                single_code = '00' + str(single_code)
            if len(single_code) == 5:
                single_code = '0' + str(single_code)
            if len(single_code) == 6:
                single_code = str(single_code)
            else:
                print('数据加载错误！')
                break
            count += 1
            stock_list.append(single_code)
        return stock_list

    def data_sort(self, data):
        # 数据倒序排列-按最新时间到旧时间排序
        m, n =shape(data)
        data_stock = []
        for line in reversed(range(m)):
            temp_line = data[line, :]
            data_stock.append(temp_line)
        data_stock = mat(array(data_stock))
        return data_stock