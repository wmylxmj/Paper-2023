# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:37:17 2022

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from itertools import repeat

class Covid19Reader(object):
    
    def __init__(self, GeoID, DataFrame):
        self.GeoID = GeoID
        self.df = DataFrame.loc[DataFrame["ID"]==GeoID]
        self.read()
        pass
    
    def read(self):
        self.dates = []
        for index, row in self.df.iterrows():
            date = str(row["Date"])
            if date not in self.dates:
                self.dates.append(date)
                pass
            pass
        # cases
        self.active_cases = []
        self.confirmed_cases = []
        self.deaths_cases = []
        self.hospitalized_cases = []
        self.recovered_cases = []
        # cases new
        self.active_cases_new = []
        self.confirmed_cases_new = []
        self.deaths_cases_new = []
        self.hospitalized_cases_new = []
        self.recovered_cases_new = []
        for date in self.dates:
            date_df = self.df.loc[self.df["Date"].astype('str')==date]
            active_df = date_df.loc[date_df["Type"]=="Active"]
            confirmed_df = date_df.loc[date_df["Type"]=="Confirmed"]
            deaths_df = date_df.loc[date_df["Type"]=="Deaths"]
            hospitalized_df = date_df.loc[date_df["Type"]=="Hospitalized"]
            recovered_df = date_df.loc[date_df["Type"]=="Recovered"]
            # cases
            self.active_cases.append(active_df["Cases"].mean())
            self.confirmed_cases.append(confirmed_df["Cases"].mean())
            self.deaths_cases.append(deaths_df["Cases"].mean())
            self.hospitalized_cases.append(hospitalized_df["Cases"].mean())
            self.recovered_cases.append(recovered_df["Cases"].mean())
            # cases new
            self.active_cases_new.append(active_df["Cases_New"].mean())
            self.confirmed_cases_new.append(confirmed_df["Cases_New"].mean())
            self.deaths_cases_new.append(deaths_df["Cases_New"].mean())
            self.hospitalized_cases_new.append(hospitalized_df["Cases_New"].mean())
            self.recovered_cases_new.append(recovered_df["Cases_New"].mean())
            pass
        pass
    
    pass

    
class PolicyReader(object):
    
    def __init__(self, GeoID, DataFrame):
        self.GeoID = GeoID
        self.df = DataFrame.loc[DataFrame["ID"]==GeoID]
        self.PolicyType = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
        self.policyType_index_dict = {}
        self.index_policyType_dict = {}
        for i, p in enumerate(self.PolicyType):
            self.policyType_index_dict[p] = i
            self.index_policyType_dict[i] = p
            pass
        self.read()
        pass
    
    def read(self):
        self.dates = []
        for index, row in self.df.iterrows():
            date = str(row["Date"])
            if date not in self.dates:
                self.dates.append(date)
                pass
            pass
        self.policy_seq = []
        for date in self.dates:
            date_policy = list(repeat(0, len(self.PolicyType)))
            date_df = self.df.loc[self.df["Date"].astype('str')==date]
            for index, row in date_df.iterrows():
                policy_type = row["PolicyType"]
                policy_flag = row["PolicyFlag"]
                if policy_type in self.PolicyType and policy_flag == True:
                    date_policy[self.policyType_index_dict[policy_type]] = 1
                    pass
                pass
            self.policy_seq.append(date_policy)
            pass
        pass
    
    pass


class DataReader(object):
    
    def __init__(self, GeoID, Covid19DF, PolicyDF):
        self.__GeoID = GeoID
        self.__covid19_reader = Covid19Reader(GeoID, Covid19DF)
        self.__policy_reader = PolicyReader(GeoID, PolicyDF)
        self.__start_date = max(self.__covid19_reader.dates[0], self.__policy_reader.dates[0])
        self.__end_date = min(self.__covid19_reader.dates[-1], self.__policy_reader.dates[-1])
        self.__covid19_start_index = self.__covid19_reader.dates.index(self.__start_date)
        self.__policy_start_index = self.__policy_reader.dates.index(self.__start_date)
        self.__covid19_end_index = self.__covid19_reader.dates.index(self.__end_date)
        self.__policy_end_index = self.__policy_reader.dates.index(self.__end_date)
        pass
    
    @property
    def start_date(self):
        return self.__start_date
    
    @start_date.setter
    def start_date(self, start_date):
        self.__start_date = str(start_date)
        self.__covid19_start_index = self.__covid19_reader.dates.index(self.__start_date)
        self.__policy_start_index = self.__policy_reader.dates.index(self.__start_date)
        pass
    
    @property
    def end_date(self):
        return self.__end_date
    
    @end_date.setter
    def end_date(self, end_date):
        self.__end_date = str(end_date)
        self.__covid19_end_index = self.__covid19_reader.dates.index(self.__end_date)
        self.__policy_end_index = self.__policy_reader.dates.index(self.__end_date)
        pass
    
    @property
    def confirmed_cases_new(self):
        confirmed_cases_new = self.__covid19_reader.confirmed_cases_new[self.__covid19_start_index:self.__covid19_end_index+1]
        confirmed_cases_new = [0 if math.isnan(x) else x for x in confirmed_cases_new]
        return confirmed_cases_new
    
    @property
    def recovered_cases_new(self):
        recovered_cases_new = self.__covid19_reader.recovered_cases_new[self.__covid19_start_index:self.__covid19_end_index+1]
        recovered_cases_new = [0 if math.isnan(x) else x for x in recovered_cases_new]
        recovered_cases_new = [0 if x < 0 else x for x in recovered_cases_new]
        return recovered_cases_new
    
    @property
    def deaths_cases_new(self):
        deaths_cases_new = self.__covid19_reader.deaths_cases_new[self.__covid19_start_index:self.__covid19_end_index+1]
        deaths_cases_new = [0 if math.isnan(x) else x for x in deaths_cases_new]
        deaths_cases_new = [0 if x < 0 else x for x in deaths_cases_new]
        return deaths_cases_new
    
    @property
    def policy_series(self):
        policy_series = self.__policy_reader.policy_seq[self.__policy_start_index:self.__policy_end_index+1]
        policy_series = [[0 if math.isnan(x) else x for x in line] for line in policy_series]
        return policy_series
    
    @property
    def dates(self):
        return self.__policy_reader.dates[self.__policy_start_index:self.__policy_end_index+1]
    
    def plot(self):
        Id_covid19_series = self.confirmed_cases_new
        Rd_covid19_series = self.recovered_cases_new
        policy_series = self.policy_series
        plt.plot(Id_covid19_series, label="confirmed cases new")
        plt.plot(Rd_covid19_series, label="recovered cases new")
        scale = np.max(np.array(Id_covid19_series))
        n_policy_valid = np.sum((np.sum(np.array(policy_series), axis=0)>0).astype(int))
        j = 0
        for i, p in enumerate(self.__policy_reader.PolicyType):
            if np.sum(np.array(policy_series)[:, i]) > 0:
                j += 1
                plt.plot(scale*np.array(policy_series)[:, i]*j/(1.0*n_policy_valid), label="policy: "+p)
                pass
            pass
        x_day_interval = max(1, (len(self.dates)) // 10)
        plt.xticks(list(range(0, len(self.dates), x_day_interval)), self.dates[0::x_day_interval], rotation="vertical")
        plt.legend(bbox_to_anchor=(1.05, 1.0), borderaxespad=0)
        plt.show()
        pass
    
    pass

