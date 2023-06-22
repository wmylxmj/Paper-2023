# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:57:42 2023

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import math
from itertools import repeat
import pyecharts
import calendar
from pyecharts.datasets import register_url

# use the js online or install the package
try:
    register_url("https://echarts-maps.github.io/echarts-countries-js/")
    pass
except:
    print("please check the network")
    pass

csse_us_csv = r"COVID-19-master\COVID-19-master\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_US.csv"
csse_us_df = pd.read_csv(csse_us_csv)

class JHUCSSEUSDataReader(object):
    
    def __init__(self, DataFrame):
        self.df = DataFrame
        self.dates = list(csse_us_df.columns[ord("L")-ord("A"):])
        self.read()
        pass
    
    def read(self):
        self.data = {}
        self.province_state_data = {}
        for key in self.df["Combined_Key"]:
            key_split = key.split(',')
            province = key_split[-2].strip()
            if province not in self.province_state_data.keys():
                province_df = np.array(self.df.loc[self.df["Province_State"].astype('str')==str(province)].iloc[:, ord("L")-ord("A"):])
                province_array = np.sum(np.array(province_df), axis=0)
                self.province_state_data[province] = province_array.tolist()
                pass
            if len(key_split) == 3:
                _Admin2, _Province_State, _Country_Region = key_split
                country = _Country_Region.strip()
                if country not in self.data.keys():
                    self.data[country] = {}
                    pass
                province = _Province_State.strip()
                if province not in self.data[country].keys():
                    self.data[country][province] = {}
                    pass
                admin2 = _Admin2.strip()
                df = self.df.loc[self.df["Combined_Key"].astype('str')==str(key)].iloc[:, ord("L")-ord("A"):]
                self.data[country][province][admin2] = np.sum(np.array(df), axis=0).tolist()
                pass
            else:
                _Province_State, _Country_Region = key_split
                country = _Country_Region.strip()
                if country not in self.data.keys():
                    self.data[country] = {}
                    pass
                province = _Province_State.strip()
                df = self.df.loc[self.df["Combined_Key"].astype('str')==str(key)].iloc[:, ord("L")-ord("A"):]
                self.data[country][province] = np.sum(np.array(df), axis=0).tolist()
                pass
            pass
        pass
    
    pass

def get_pairs(csse_us_data_reader, date):
    pairs = []
    min_val, max_val = None, None
    for province in csse_us_data_reader.province_state_data.keys():
        index = csse_us_data_reader.dates.index(date)
        value = csse_us_data_reader.province_state_data[province][index]
        if min_val == None:
            min_val = value
            max_val = value
            pass
        min_val = min(value, min_val)
        max_val = max(value, max_val)
        pair = (province, value)
        pairs.append(pair)
        pass
    return pairs, max_val, min_val

def fig_map(csse_us_data_reader, date):
    month, day, year = date.split("/")
    title = "Confirmed cases of COVID-19 in US states on {} {}, 20{}".format(calendar.month_abbr[int(month)], day, year)
    pairs, max_val, min_val = get_pairs(csse_us_data_reader, date)
    us_map = pyecharts.charts.Map(init_opts=pyecharts.options.InitOpts()) 
    us_map.add(series_name='', data_pair=pairs, maptype="美国", is_map_symbol_show=False)
    visualmap_opts = pyecharts.options.VisualMapOpts(max_=max_val, min_=min_val, is_piecewise=False, \
                                                     range_color=['#FFFFFF', '#E6B8B8', '#CC7A7A', \
                                                                  '#B34747', '#991F1F', '#800000'])
    us_map.set_global_opts(title_opts=pyecharts.options.TitleOpts(title=title), \
                           visualmap_opts=visualmap_opts)
    us_map.set_series_opts(label_opts=pyecharts.options.LabelOpts(is_show=False))
    us_map.render(path="{}.html".format(title))
    pass
        
data_reader = JHUCSSEUSDataReader(csse_us_df)
date = "12/1/20"
fig_map(data_reader, date)

