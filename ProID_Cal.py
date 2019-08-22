# coding:utf-8
import pandas as pd
# import CSI_data
from numpy import *
import tushare as ts
import time
from Data_Collect import data_collect
from HS300_data import HS300

'''
    计算区间的指标：
        1)收益：区间收益率(年化)、Alpha、正收益天数比例;
        2)风险/波动：区间波动率、Beta、负收益天数比例、负收益波动率、最大回撤、峰谷差值、Efficiency Ratio、Price Density;
        3)收益风险比：Information Ratio、夏普律、Treynor Ratio、Calmar Ratio、Sortino Ratio;
        4)收益曲线形态：Skewness、kurtosis、跟踪误差年化；
'''
class ID_cal:
    # 多支股票数据分析
    def load_data(self, close, high, low, HS_close):
        try:
            # 加载收盘数据
            self.df = close
            self.high_df = high
            self.low_df = low
            self.df_HS = HS_close
            # 权重配置
            self.weights_por = tile(1/74, shape(close)[1])
        except IOError as ioe:
            print(ioe)

    def cal_rate_income(self):
        # 01.计算多支股票的的日收益率-rate_income:R=(P2-P1)/P1, 序列
        data_row, data_col = shape(self.df)
        income = []
        for i in range(data_row - 1):
            prior_df = mat(self.df)[data_row - i - 1, :]
            temp_df = mat(self.df)[data_row - i - 2, :]
            # 收益率-rate_income
            diff_close = temp_df - prior_df
            income.append(diff_close / prior_df)
        d_rate_income = pd.DataFrame(mat(array(income))).fillna(0)  # 日间收益率

        # 02.计算年化收益率-复利：[(1+r1)(1+r2)……(1+rN)]**(m/T)-1, r1、r2……rN为日收益率序列
        rows, cols = shape(mat(array(d_rate_income)))
        rate_y = []
        for col in range(cols):
            star_rate = 1.0
            temp_J = mat(d_rate_income)[:, col]
            for row in range(rows):
                val_r = temp_J[row, :]
                temp_val = val_r + 1
                star_rate = temp_val * star_rate
            rate_y.append(star_rate)
        annual_rates = []
        for col in range(cols):
            temp_rate = mat(array(rate_y))[:, col]
            annual_rate = (array(temp_rate) ** (252/data_row))-1
            annual_rates.append(annual_rate)
        annual_rates = mat(array(annual_rates)).T  # 年化收益率_复利

        # 03.计算正收益天数和负收益天数
        z_income_day = []
        f_income_day = []
        for j in range(cols):
            rate_income = mat(d_rate_income)[:, j]
            sum_notzeros = 0
            for line in range(rows):
                if rate_income[line, :] > 0:
                    sum_notzeros = sum_notzeros + 1
                    continue
            z_income_day.append(sum_notzeros)
            f_income_day.append(data_row - mat(z_income_day))
        z_income_days_ratio = (mat(z_income_day).T + 1)/data_row
        z_income_days = (mat(z_income_day).T + 1)
        f_income_days_ratio = (f_income_day[-1].T - 1)/data_row
        f_income_days = (f_income_day[-1].T - 1)
        return d_rate_income, annual_rates, z_income_days, z_income_days_ratio, f_income_days, f_income_days_ratio

    def cal_stdev(self, d_rate_income):
        row, col = shape(d_rate_income)
        # 计算年化波动率:
        # 方式01：stdev(ri)*sqrt(252), ri：日间收益率
        stdev_ri = d_rate_income.std()  # 每只股票的标准差
        annual_Vt00 = stdev_ri * sqrt(252)
        # 方式02：sqrt[(252/(n-1))*sum(ri-ri_mean)**2]
        ri_mean = mat(mean(d_rate_income, axis=0))
        diff_ri = mat(d_rate_income) - tile(ri_mean, (row, 1))
        # 数组对应元素位置相乘:multiply函数;
        annual_Vt01 = (sqrt(sum(multiply(diff_ri, diff_ri), axis=0)*(252/row))).T
        # 计算年化负收益波动率
        f_annual_VtAll = []
        for j in range(col):
            new_d_rate = []
            temp_rate_j = d_rate_income.ix[:, j]
            for i in range(row):
                temp_rate_i = temp_rate_j.ix[i, :]
                if temp_rate_i <= 0:
                    new_d_rate.append(temp_rate_i)
                    continue
            # f_annual_Vt = (mat(new_d_rate).std()) * sqrt(252)
            row_f = shape(mat(new_d_rate).T)[0]
            f_ri_mean = mat(mean(new_d_rate, axis=0))
            d = mat(new_d_rate).T
            f_diff_ri = mat(new_d_rate).T - tile(f_ri_mean, (row_f, 1))
            # 数组对应元素位置相乘:multiply函数;
            f_annual_Vt01 = (sqrt(sum(multiply(f_diff_ri, f_diff_ri), axis=0) * (252 / row_f))).T
            f_annual_VtAll.append(f_annual_Vt01)
        f_annual_VtAll = mat(array(f_annual_VtAll)).T
        f_stdev_ri = f_annual_VtAll / sqrt(252)  # 年化负收益的标准差
        return annual_Vt01, stdev_ri, f_annual_VtAll, f_stdev_ri

    def cal_mdd(self, d_rate_income):
        # 计算最大回撤
        row, col = shape(d_rate_income)
        top_rowval = array(1.0)
        mdd_all = []
        MDD_all = []
        for j in range(col):
            rate_J = mat(d_rate_income)[:, j]
            value = array((1 + rate_J).cumprod()).T
            values = vstack((top_rowval, value))
            # 回撤值
            D_val = pd.DataFrame(values).cummax()-values
            # d_val = D_val/(D_val+values)
            # 最大回撤
            MDD_val = D_val.max()
            MDD_all.append(MDD_val)
            # 最大回撤率
            # mdd_val = d_val.max()
            # mdd_all.append(mdd_val)
        MDD_all = mat(MDD_all).T
        # self.to_csv(MDD_all, "MDD.csv")
        return MDD_all

    def cal_beta_alpha(self, d_rate_income, c_annual_rates):
        SH300_d_rate, HS300_weight_por, HS_annual_rates, pro_HS_annual_rates = self.SH300_income()  # HS300的日收益率
        rate_income = array(d_rate_income)  # 获取每股的收益率
        # 计算HS300的组合收益率
        HS300_por_rate= SH300_d_rate.mul(HS300_weight_por, axis=1).sum(axis=1)
        # 求解上证指数与股票的beta值：beta=cov(A收益率, B收益率)/var(B收益率)；
        # 其中，A为单一金融工具的行情走势，B为一个投资组合或者指数的行情走势；
        row, col = shape(rate_income)
        beta = []
        HS300_rate = mat(HS300_por_rate).T
        for j in range(col):
            ret = hstack((array(mat(rate_income[:, j])).T, array(HS300_rate)))
            beta_single = pd.DataFrame(ret).cov().iat[0, 1]/HS300_rate.var()
            beta.append(beta_single)
        beta = mat(beta).T

        # 计算Alpha值：投资或基金的绝对回报和按照β系数计算的预期风险回报之间的差额
        # Rf = (1 + 2.75 / 100) ** (1 / 360) - 1  # 无风险收益率
        Rf = 1.5 / 100  # 无风险收益率
        # start_rate_sh = 1
        # for k in range(row):
        #     temp_sh = HS300_rate[k, :]
        #     temp_val = temp_sh + 1
        #     start_rate_sh = temp_val * start_rate_sh
        # # 获取指数的年化收益率-复利:[(1+r1)(1+r2)……(1+rN)]**(m/T)-1
        # HS_annual_rates = (array(start_rate_sh) ** (252 / (row+1))) - 1
        alpha = []
        for c in range(col):
            # 计算公式：alpha=Pr-[rf+beta*(Br-rf)]
            alpha_single = c_annual_rates[c, :] - (Rf + (pro_HS_annual_rates - Rf)*beta[c, :])
            alpha.append(alpha_single)
        alpha = mat(array(alpha)).T
        return beta, alpha, HS300_rate, HS300_weight_por

    def cal_sharpe(self, d_rate_income, annual_Vt, annual_rates):
        # 夏普比率 =（年化收益率annual_rates - 无风险收益率） / 年度风险annual_Vt，其中"年度风险"与"年化波动率"等值
        # 如果夏普比率为正值，说明在衡量期内基金的平均净值增长率超过了无风险利率，在以同期银行存款利率作为无风险利率的情况下，说明投资基金比银行存款要好
        # 夏普比率越大，说明基金单位风险所获得的风险回报越高
        # 夏普比率为负时，按大小排序没有意义
        # 计算无风险收益率--短期国债收益率20190725(2:39:48)为3.835
        # Rf = (1 + 3.835/100)**(1/360)-1
        # Rf = (1 + 2.75/100)**(1/360)-1
        Rf = 1.5/100
        Sharpe = (annual_rates - Rf)/annual_Vt
        return Rf, Sharpe

    def cal_Information(self, annual_Vt, annual_rates):
        # 计算Information Ratio:
        # 年化收益率(annual_rates)/年度风险(annual_Vt)，其中"年度风险"与"年化波动率"等值
        IR = annual_rates/mat(annual_Vt)
        return IR

    def cal_efficiency(self):
        # 计算效益比例：价格变化的净值(正数)/个股价格变化的总和(正数)
        # 或者 ER=|Pt-Pt-n|/sum(Pi-Pi-1),n为测试周期，P为相关金融工具的价格
        # 获取收盘价格
        close_df = self.df
        row, col = shape(close_df)
        # 获取整个周期的初始价格和最后价格的差值，即价格变化的净值(正数)
        diff_Pclose = abs(close_df[row-1, :] - close_df[0, :])
        # 获取个股价格变化的总和(正数)
        sum_close_J = []
        for j in range(col):
            close_df_J = close_df[:, j]
            diff_close_J = []
            for i in range(row-1):
                diff_close_I = abs(close_df_J[row-i-2, :] - close_df_J[row-i-1, :]).round(2)
                diff_close_J.append(diff_close_I)
            sum_diff_close = sum(diff_close_J).round(2)
            sum_close_J.append(sum_diff_close)
        # 每支股票价格变化的净值(正数)
        diff_Pclose = mat(diff_Pclose).T
        # 个股价格变化的总和(正数)
        sum_close_J = mat(sum_close_J).T
        # 效益比例--噪声检测
        ER = mat(pd.DataFrame(diff_Pclose/sum_close_J).fillna(0))
        return ER

    def cal_Treynor_Calmar(self, d_rate_income, HS300_pro_rate, Rf, annual_rates, MDD_all):
        # 计算特雷诺比率：(年化收益率:annual_rates-无风险收益率:Rf)/程序的beta系数
        # 程序的beta系数：各成分股beta值的加权计算
        weights_por = mat(self.weights_por)
        pro_rate = multiply(mat(d_rate_income), weights_por).sum(axis=1)
        ret = hstack((array(HS300_pro_rate), array(mat(pro_rate))))
        pro_beta = pd.DataFrame(ret).cov().iat[0, 1] / HS300_pro_rate.var()
        Treynor = (annual_rates-Rf)/pro_beta
        # 计算卡尔玛比率：年化收益率/最大跌幅，Calmar比率描述的是收益和最大回撤之间的关系。
        # 计算方式为年化收益率(annual_rates)与历史最大回撤(MDD_all)之间的比率。
        # 疑问：最大跌幅是最大回撤(一只股票或基金从价格（净值）最高点到最低点的跌幅)
        # Calmar比率数值越大，基金的业绩表现越好；反之，基金的业绩表现越差。
        Calmar = annual_rates/mat(MDD_all)
        return Treynor, Calmar

    def cal_Sortino(self, d_rate_income, annual_rates, Rf, HS300_pro_rate):
        # 计算索迪诺比率：(年化收益率annual_rates-无风险利率Rf)/年化下行波动率
        # 计算年化下行波动率：使用基准组合(HS300)收益为目标收益，作为向上波动和向下波动的判断标准
        row, col= shape(d_rate_income)
        sigma = []
        for j in range(col):
            temp_rate_J = mat(d_rate_income.ix[:, j]).T
            sum_diff_Ii = 0
            for i in range(row):
                temp_rate_I = temp_rate_J[i, :]
                HS300_rate_I = HS300_pro_rate[i, :]
                I_i = 0  # 初始默认为向上波动状态
                if temp_rate_I < HS300_rate_I:
                    I_i = 1  # 状态转换
                    diff_rate = temp_rate_I - HS300_rate_I
                    I_diff = multiply(diff_rate, diff_rate)*I_i  # 向下波动
                else:
                    diff_rate = temp_rate_I - HS300_rate_I
                    I_diff = multiply(diff_rate, diff_rate) * I_i  # 向下波动
                sum_diff_Ii = sum_diff_Ii + I_diff
            single_sigma = sqrt((252/row)*sum_diff_Ii)
            sigma.append(single_sigma)
        sigma_d = mat(array(sigma)).T  # 年化下行波动率
        # 计算索迪诺比率
        Sortino = (annual_rates - Rf)/sigma_d
        return Sortino

    def cal_Skewness_Kurtosis(self):
        # 计算偏度值：[sum(Pi-mean_P)**3]/[(n-1)*std**3],P为价格:收盘价
        # 计算峰度值：[sum(Pi-mean_P)**4]/[(n-1)*std**4]
        close_price = self.df
        row, col = shape(mat(close_price))
        meanP_colse = mat(mean(close_price, axis=0))
        S_k_all = []; K_all =[]
        for j in range(col):
            diff_all = []
            close_price_J = close_price[:, j]
            mean_PCJ = meanP_colse[:, j]
            for i in range(row):
                close_price_I = close_price_J[row-i-1, :]
                diff = close_price_I - mean_PCJ
                diff_all.append(diff)
            diff_all = (array(diff_all))
            std_Pclose = mat(close_price_J).std()
            # 计算偏度值
            S_k = (sum(diff_all*diff_all*diff_all))/((row-1)*(std_Pclose**3))
            S_k_all.append(S_k)
            # 计算峰度值
            K = (sum(diff_all*diff_all*diff_all*diff_all))/((row-1)*(std_Pclose**4))
            K_all.append(K)
        S_k_all = mat(pd.DataFrame(S_k_all).fillna(0))
        K_all = mat(pd.DataFrame(K_all).fillna(0))
        return S_k_all, K_all

    def cal_Tracking_Error(self, d_rate_income, SH300_d_rate, SH300_weight_por):
        # 计算跟踪误差:以沪深300指数作为基准
        wpor_income = SH300_d_rate.mul(SH300_weight_por, axis=1)
        # 基准组合的收益率
        HS300_por_income = mat(wpor_income.sum(axis=1)).T
        d_rate_income = mat(d_rate_income)
        row, col = shape(d_rate_income)
        # 计算跟踪偏离度:Rpa(i)=Rp(i)-Rb(i)，其中Rp(i)为日收益率，Rb(i)为基准组合日收益率
        Rpa = d_rate_income - tile(HS300_por_income, col)  # 日主动收益率
        M_Rpa = Rpa.mean(axis=0)
        # 计算每支股票的年化跟踪误差
        diff_Rpa = mat(array(Rpa - tile(M_Rpa, (row, 1)))**2)
        annual_Tracking_Error = mat(sqrt((252/row)*sum(diff_Rpa, axis=0))).T
        return annual_Tracking_Error

    def cal_PriceDensity(self):
        # 计算价格密度：[sum(high_i-low_i)]/[max_high-min_low]
        high_df = self.high_df  # 股票每日最高价数据
        low_df = self.low_df  # 股票每日最低价数据
        diff_HL = mat(high_df-low_df)
        row, col = shape(diff_HL)
        PD = []
        for c in range(col):
            # 计算每支股票的价格密度
            PD_c = sum(diff_HL[:, c])/(high_df[:, c].max() - low_df[:, c].min())
            PD.append(PD_c)  # 汇总
        PD = mat(pd.DataFrame(PD).fillna(0))
        return PD

    def cal_DIFF_FtoV(self, d_rate_income):
        # 计算峰谷差值
        d_rate = d_rate_income
        row, col = shape(d_rate)
        DIFF_FtoV = []
        for c in range(col):
            d_rate_c = d_rate.ix[:, c]
            max_drate = d_rate_c.max()
            min_drate = d_rate_c.min()
            Diff_FtoV = max_drate-min_drate
            DIFF_FtoV.append(Diff_FtoV)
        DIFF_FtoV = mat(DIFF_FtoV).T
        return DIFF_FtoV

    def SH300_income(self):
        # 计算基准组合的的日收益率
        df_HS = self.df_HS
        # 查看哪些列存在缺失值
        # df_isnull = list(df_HS.isnull().sum())
        # 缺失值-均值填充
        df_HS = pd.DataFrame(df_HS).fillna(df_HS.mean())
        data_row = shape(df_HS)[0]
        income = []
        for i in range(data_row - 1):
            prior_df = mat(df_HS)[data_row - i - 1, :]
            temp_df = mat(df_HS)[data_row - i - 2, :]
            # 基准组合收益率-rate_income
            diff_close = temp_df - prior_df
            income.append(diff_close / prior_df)
        SH300_d_rate = pd.DataFrame(mat(array(income))).fillna(0)
        # 计算基准组合的年化收益率-复利
        rows, cols = shape(mat(array(SH300_d_rate)))
        rate_y = []
        for col in range(cols):
            star_rate = 1.0
            temp_J = mat(SH300_d_rate)[:, col]
            for row in range(rows):
                val_r = temp_J[row, :]
                temp_val = val_r + 1
                star_rate = temp_val * star_rate
            rate_y.append(star_rate)
        annual_rates = []
        for col in range(cols):
            temp_rate = mat(array(rate_y))[:, col]
            annual_rate = (array(temp_rate) ** (252 / data_row)) - 1
            annual_rates.append(annual_rate)
        HS_annual_rates = mat(array(annual_rates)).T  # 单股基准年化收益率
        # 获取SH300的权重数据
        SH300_weight = pd.read_excel('HS300_data/HS300.xlsx')
        weights_por = array(SH300_weight['权重(%)'])
        SH300_weight_por = weights_por/sum(weights_por)
        weights_HS = mat(SH300_weight_por).T
        # 基准组合的年化收益率
        pro_HS_annual_rates = multiply(HS_annual_rates, weights_HS).sum(axis=0)
        return SH300_d_rate, SH300_weight_por, HS_annual_rates, pro_HS_annual_rates

    def portfolio_Stock(self, d_rate_income, stdev_ri, f_stdev_ri, id_all):
        # 投资组合的ID指标计算--方差var和标准差std除外
        weights_por = mat(self.weights_por).T  # 投资组合权重
        IDPor_stcck = multiply(id_all, tile(weights_por, 21)).sum(axis=0)
        self.to_csv(filedata=IDPor_stcck, filename='ProID_Result.csv')

        # 计算投资组合的年化波动率--标准差也被称为波动率
        stdev_ri = mat(stdev_ri).T
        f_stdev_ri = f_stdev_ri  # 负收益波动率
        w_std = ((array(weights_por) ** 2)*(array(stdev_ri) ** 2)).sum(axis=0)
        fw_std = ((array(weights_por) ** 2)*(array(f_stdev_ri) ** 2)).sum(axis=0)
        # 收益率之间的相关系数
        corr_ij = mat(pd.DataFrame(d_rate_income).corr())
        wstd_add = 0
        f_wstd_add = 0
        for std_i in range(shape(weights_por)[0]):
            temp_wi = weights_por[std_i, :]
            for std_j in range(shape(weights_por)[0]):
                if std_i != std_j:
                    temp_wj = weights_por[std_j, :]
                    wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * stdev_ri[std_i, :] * stdev_ri[std_j, :]
                    f_wstd_multi = temp_wi * temp_wj * corr_ij[std_i, std_j] * f_stdev_ri[std_i, :] * f_stdev_ri[std_j, :]
                    wstd_add = wstd_multi + wstd_add
                    f_wstd_add = f_wstd_multi + f_wstd_add
                    continue
        # 方差计算结果
        var_sum = w_std + wstd_add; f_var_sum = fw_std + f_wstd_add
        # 投资组合的年化波动率
        por_stdev = sqrt(var_sum) * sqrt(252)
        f_por_stdev = sqrt(f_var_sum) * sqrt(252)
        print("===============计算结果中需替换的值================")
        print('投资组合的年化波动率：', por_stdev, '；投资组合的年化负收益的波动率：', f_por_stdev)
        return corr_ij, wstd_add

    def risk_porStock(self, d_rate_income, HS300_rate):
        top_rowval = 1.0
        # 计算加权后：投资组合的收益率-pro_rate
        weights_por = mat(self.weights_por)
        pro_rate = multiply(mat(d_rate_income), weights_por).sum(axis=1)
        # self.to_csv(pro_rate, 'pro_rate.csv')
        # 计算投资组合的最大回撤
        pro_values = vstack((top_rowval, array((top_rowval + mat(pro_rate)).cumprod()).T))  # 净值
        # self.to_csv(pro_values, 'pro_values.csv')
        pro_MDD_T = (pd.DataFrame(pro_values).cummax() - pro_values).max()
        print('投资组合的最大回撤值：', pro_MDD_T)
        # 计算投资组合的最大损失
        max_loss = top_rowval - pro_values.min()
        print('投资组合的的最大损失：', max_loss)
        # 计算投资组合的Beta值
        ret = hstack((array(HS300_rate), array(mat(pro_rate))))
        pro_beta = pd.DataFrame(ret).cov().iat[0, 1] / HS300_rate.var()
        print('投资组合的Beta值：', pro_beta)
        return pro_MDD_T, max_loss, pro_beta

    def to_csv(self, filedata, filename):
        # 文件存储
        tofile = pd.DataFrame(filedata)
        tofile.to_csv(filename)

    def merge_ID(self):
        '''
            1)区间收益率(年化)annual_rates 、Alpha、正收益天数比例z_income_days_ratio;
            2)区间波动率annual_Vt、Beta、负收益天数比例f_income_days_ratio
                    负收益波动率f_annual_VtAll、最大回撤MDD_all、峰谷差值DIFF_FtoV、
                    Efficiency Ratio、Price Density;
            3)Information Ratio、夏普律sharpe、Treynor Ratio、Calmar Ratio、Sortino Ratio;
            4)Skewness、kurtosis、跟踪误差年化Tracking_Error；
        '''
        print('run：计算五大类指标(21个)')
        d_rate_income, annual_rates, z_income_days, z_income_days_ratio, f_income_days, f_income_days_ratio = self.cal_rate_income()
        annual_Vt, stdev_ri, f_annual_VtAll, f_stdev_ri = self.cal_stdev(d_rate_income)
        MDD_all = self.cal_mdd(d_rate_income)
        beta, alpha, HS300_pro_rate, SH300_weight_por = self.cal_beta_alpha(d_rate_income, annual_rates)
        Rf, Sharpe = self.cal_sharpe(d_rate_income, annual_Vt, annual_rates)
        IR = self.cal_Information(annual_Vt, annual_rates)
        ER = self.cal_efficiency()
        Treynor, Calmar = self.cal_Treynor_Calmar(d_rate_income, HS300_pro_rate, Rf, annual_rates, MDD_all)
        Sortino = self.cal_Sortino(d_rate_income, annual_rates, Rf, HS300_pro_rate)
        Skewness, Kurtosis = self.cal_Skewness_Kurtosis()
        PriceDensity = self.cal_PriceDensity()
        DIFF_FtoV = self.cal_DIFF_FtoV(d_rate_income)
        SH300_d_rate, SH300_weight_por, HS_annual_rates, pro_HS_annual_rates = self.SH300_income()
        Tracking_Error = self.cal_Tracking_Error(d_rate_income, SH300_d_rate, SH300_weight_por)
        # 合并单股21个指标值
        id_all = hstack((annual_rates, z_income_days, z_income_days_ratio, f_income_days,
                         f_income_days_ratio, annual_Vt, f_annual_VtAll, MDD_all, DIFF_FtoV,
                         beta, alpha, IR, Sharpe, ER, PriceDensity, Treynor, Calmar, Sortino,
                         Skewness, Kurtosis, Tracking_Error))
        print('run：计算投资组合的指标结果')
        self.portfolio_Stock(d_rate_income, stdev_ri, f_stdev_ri, id_all)
        self.risk_porStock(d_rate_income, HS300_pro_rate)