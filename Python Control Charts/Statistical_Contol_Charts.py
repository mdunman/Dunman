#!/usr/bin/env python
# coding: utf-8

import sys
import csv
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas.plotting import table

def prep_data(data):
    data = data.sort_values('Timestamp',ascending=True).reset_index()
    day = []
    for i in range(len(data)):
        day.append(data.loc[i]['Timestamp'].split(' ')[0])
    data['Date']=day
    return data

def xbars_and_stddevs(clean_data):
    n = 25; xbars = {}; stddevs = {};
    devices = list(set(clean_data['ID']))
    for i in range(len(devices)):
        dev = clean_data[clean_data['ID']==devices[i]]
        dates = sorted(list(set(dev['Date'])))
        nums = list(dev.groupby('Date').count()['Timestamp'])
        good_dates = []
        for j in range(len(nums)):
            if nums[j]>=25:
                good_dates.append(dates[j])
        good_dates.sort()
        devrr_rand = []
        for k in good_dates:
            sub = dev[dev['Date']==k]
            div = len(sub)/n
            d = []
            for j in range(0,n):
                subset = list(sub[int(div*j):max(n,int(div*(j+1)))]['Rate'])
                randi = np.random.randint(0,len(subset))
                d.append(round(subset[randi],2))
            devrr_rand.append(d)
        devrr_rand = np.array(devrr_rand)
        xbar = []; stddev = []
        for p in range(len(devrr_rand)):
            xbar.append(round(devrr_rand[p].mean(),2))
            stddev.append(round(devrr_rand[p].std(),2))
        xbars[str(devices[i])] = xbar
        stddevs[str(devices[i])] = stddev
    for i in devices:
        if len(xbars[i]) < 2:
            devices.remove(i)
            del xbars[i]
            del stddevs[i]
    return devices, xbars, stddevs

def control_limits(devices, xbars, stddevs):
    CLx = {}; CLs = {};
    for i in devices:
        CLx[i] = round(np.array(xbars[i]).mean(),2)
        CLs[i] = round(np.array(stddevs[i]).mean(),2)
    return CLx, CLs

def control_charts(new_data, CLx, CLs):
    devices, xbars_t, stddevs_t = xbars_and_stddevs(new_data)
    A3 = 0.606; 
    B4 = 1.435; B5 = 1.29; B6 = 1.145;
    B3 = 0.565; B7 = 0.71; B8 = 0.855;
    UCLx = {}; LCLx = {}; UCLx2 = {}; UCLx1 = {}; LCLx2 = {}; LCLx1 = {};
    UCLs = {}; LCLs = {}; UCLs2 = {}; UCLs1 = {}; LCLs2 = {}; LCLs1 = {};
    alert_table = pd.DataFrame({'Device ID':[], 'Alert Type':[]})
    for i in devices:
        alerts = []
        UCLx[i] = round(CLx[i] + A3*CLs[i],2);
        UCLx2[i] = round(CLx[i] + (2/3)*A3*CLs[i],2); UCLx1[i] = round(CLx[i] + (1/3)*A3*CLs[i],2)
        LCLx[i] = max(0,round(CLx[i] - A3*CLs[i],2))
        LCLx2[i] = max(0,round(CLx[i] - (2/3)*A3*CLs[i],2)); LCLx1[i] = max(0,round(CLx[i] - (1/3)*A3*CLs[i],2))
        UCLs[i] = round(B4*CLs[i],2)
        UCLs2[i] = round(B5*CLs[i],2); UCLs1[i] = round(B6*CLs[i],2)
        LCLs[i] = max(0,round(B3*CLs[i],2))
        LCLs2[i] = max(0,round(B7*CLs[i],2)); LCLs1[i] = max(0,round(B8*CLs[i],2))
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        plt.plot(xbars_t[i], color = 'b')
        plt.hlines(CLx[i], xmin=0, xmax=len(xbars_t[i])-1, color='black', linewidth=2)
        plt.hlines(UCLx[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=1.5)
        plt.hlines(UCLx2[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(UCLx1[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(LCLx[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=1.5)
        plt.hlines(LCLx2[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(LCLx1[i], xmin=0, xmax=len(xbars_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.xticks(np.arange(0, max(len(xbars_t[i]), 1.0)))
        for j in range(len(xbars_t[i])):
            if xbars_t[i][j]>UCLx[i]:
                plt.scatter(j,xbars_t[i][j], color="red",s=30)
                alerts.append('Avg: Above Upper Limit')
            if xbars_t[i][j]<LCLx[i]:
                plt.scatter(j,xbars_t[i][j], color="red",s=30)
                alerts.append('Avg: Below Lower Limit')
        for j in range(2,len(xbars_t[i])):
            if (xbars_t[i][j-1]>UCLx2[i])&(xbars_t[i][j]>UCLx2[i]):
                xs = [j-1,j]; ys = [xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Upper Limit')
            if (xbars_t[i][j-2]>UCLx2[i])&(xbars_t[i][j]>UCLx2[i]):
                xs = [j-2,j]; ys = [xbars_t[i][j-2],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Upper Limit')
            if (xbars_t[i][j-2]>UCLx2[i])&(xbars_t[i][j-1]>UCLx2[i]):
                xs = [j-2,j-1]; ys = [xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Upper Limit')
            if (xbars_t[i][j-1]<LCLx2[i])&(xbars_t[i][j]<LCLx2[i]):
                xs = [j-1,j]; ys = [xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Lower Limit')
            if (xbars_t[i][j-2]<LCLx2[i])&(xbars_t[i][j]<LCLx2[i]):
                xs = [j-2,j]; ys = [xbars_t[i][j-2],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Lower Limit')
            if (xbars_t[i][j-2]<LCLx2[i])&(xbars_t[i][j-1]<LCLx2[i]):
                xs = [j-2,j-1]; ys = [xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 2 Points Near Lower Limit')
        for j in range(4,len(xbars_t[i])):
            if (xbars_t[i][j-3]>UCLx1[i])&(xbars_t[i][j-2]>UCLx1[i])&(xbars_t[i][j-1]>UCLx1[i])&(xbars_t[i][j]>UCLx1[i]):
                xs = [j-3,j-2,j-1,j]; ys = [xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Above Center')
            if (xbars_t[i][j-4]>UCLx1[i])&(xbars_t[i][j-2]>UCLx1[i])&(xbars_t[i][j-1]>UCLx1[i])&(xbars_t[i][j]>UCLx1[i]):
                xs = [j-4,j-2,j-1,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Above Center')
            if (xbars_t[i][j-4]>UCLx1[i])&(xbars_t[i][j-3]>UCLx1[i])&(xbars_t[i][j-1]>UCLx1[i])&(xbars_t[i][j]>UCLx1[i]):
                xs = [j-4,j-3,j-1,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Above Center')
            if (xbars_t[i][j-4]>UCLx1[i])&(xbars_t[i][j-3]>UCLx1[i])&(xbars_t[i][j-2]>UCLx1[i])&(xbars_t[i][j]>UCLx1[i]):
                xs = [j-4,j-3,j-2,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Above Center')
            if (xbars_t[i][j-4]>UCLx1[i])&(xbars_t[i][j-3]>UCLx1[i])&(xbars_t[i][j-2]>UCLx1[i])&(xbars_t[i][j-1]>UCLx1[i]):
                xs = [j-4,j-3,j-2,j-1]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Above Center')
            if (xbars_t[i][j-3]<LCLx1[i])&(xbars_t[i][j-2]<LCLx1[i])&(xbars_t[i][j-1]<LCLx1[i])&(xbars_t[i][j]<LCLx1[i]):
                xs = [j-3,j-2,j-1,j]; ys = [xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Below Center')
            if (xbars_t[i][j-4]<LCLx1[i])&(xbars_t[i][j-2]<LCLx1[i])&(xbars_t[i][j-1]<LCLx1[i])&(xbars_t[i][j]<LCLx1[i]):
                xs = [j-4,j-2,j-1,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Below Center')
            if (xbars_t[i][j-4]<LCLx1[i])&(xbars_t[i][j-3]<LCLx1[i])&(xbars_t[i][j-1]<LCLx1[i])&(xbars_t[i][j]<LCLx1[i]):
                xs = [j-4,j-3,j-1,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Below Center')
            if (xbars_t[i][j-4]<LCLx1[i])&(xbars_t[i][j-3]<LCLx1[i])&(xbars_t[i][j-2]<LCLx1[i])&(xbars_t[i][j]<LCLx1[i]):
                xs = [j-4,j-3,j-2,j]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Below Center')
            if (xbars_t[i][j-4]<LCLx1[i])&(xbars_t[i][j-3]<LCLx1[i])&(xbars_t[i][j-2]<LCLx1[i])&(xbars_t[i][j-1]<LCLx1[i]):
                xs = [j-4,j-3,j-2,j-1]; ys = [xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 4 Points >1 Deviation Below Center')
        for j in range(6,len(xbars_t[i])):
            if (xbars_t[i][j-6]>xbars_t[i][j-5])&(xbars_t[i][j-5]>xbars_t[i][j-4])&(xbars_t[i][j-4]>xbars_t[i][j-3])&(xbars_t[i][j-3]>xbars_t[i][j-2])&(xbars_t[i][j-2]>xbars_t[i][j-1])&(xbars_t[i][j-1]>xbars_t[i][j]):
                xs = [j-6,j-5,j-4,j-3,j-2,j-1]; ys = [xbars_t[i][j-6],xbars_t[i][j-5],xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 6 Successive Points Increasing')
            if (xbars_t[i][j-6]<xbars_t[i][j-5])&(xbars_t[i][j-5]<xbars_t[i][j-4])&(xbars_t[i][j-4]<xbars_t[i][j-3])&(xbars_t[i][j-3]<xbars_t[i][j-2])&(xbars_t[i][j-2]<xbars_t[i][j-1])&(xbars_t[i][j-1]<xbars_t[i][j]):
                xs = [j-6,j-5,j-4,j-3,j-2,j-1]; ys = [xbars_t[i][j-6],xbars_t[i][j-5],xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 6 Successive Points Decreasing')
        for j in range(7,len(xbars_t[i])):
            if (xbars_t[i][j-7]>CLx[i])&(xbars_t[i][j-6]>CLx[i])&(xbars_t[i][j-5]>CLx[i])&(xbars_t[i][j-4]>CLx[i])&(xbars_t[i][j-3]>CLx[i])&(xbars_t[i][j-2]>CLx[i])&(xbars_t[i][j-1]>CLx[i])&(xbars_t[i][j]>CLx[i]):
                xs = [j-7,j-6,j-5,j-4,j-3,j-2,j-1,j]; ys = [xbars_t[i][j-7],xbars_t[i][j-6],xbars_t[i][j-5],xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 8 Successive Points Above Center')
            if (xbars_t[i][j-7]<CLx[i])&(xbars_t[i][j-6]<CLx[i])&(xbars_t[i][j-5]<CLx[i])&(xbars_t[i][j-4]<CLx[i])&(xbars_t[i][j-3]<CLx[i])&(xbars_t[i][j-2]<CLx[i])&(xbars_t[i][j-1]<CLx[i])&(xbars_t[i][j]<CLx[i]):
                xs = [j-7,j-6,j-5,j-4,j-3,j-2,j-1,j]; ys = [xbars_t[i][j-7],xbars_t[i][j-6],xbars_t[i][j-5],xbars_t[i][j-4],xbars_t[i][j-3],xbars_t[i][j-2],xbars_t[i][j-1],xbars_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Avg: 8 Successive Points Below Center')
        plt.title("X-Bar Chart")
        plt.subplot(2,1,2)
        plt.plot(stddevs_t[i], color = 'b')
        plt.hlines(CLs[i], xmin=0, xmax=len(stddevs_t[i])-1, color='black', linewidth=2)
        plt.hlines(UCLs[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=1.5)
        plt.hlines(UCLs2[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(UCLs1[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(LCLs[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=1.5)
        plt.hlines(LCLs2[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.hlines(LCLs1[i], xmin=0, xmax=len(stddevs_t[i])-1, color='gray', linestyles='dashed', linewidth=0.75)
        plt.xticks(np.arange(0, max(len(stddevs_t[i]), 1.0)))
        for j in range(len(stddevs_t[i])):
            if stddevs_t[i][j]>UCLs[i]:
                plt.scatter(j,stddevs_t[i][j], color="red",s=30)
                alerts.append('Std Dev: Above Upper Limit')
            if stddevs_t[i][j]<LCLs[i]:
                plt.scatter(j,stddevs_t[i][j], color="red",s=30)
                alerts.append('Std Dev: Below Upper Limit')
        for j in range(2,len(stddevs_t[i])):
            if (stddevs_t[i][j-1]>UCLs2[i])&(stddevs_t[i][j]>UCLs2[i]):
                xs = [j-1,j]; ys = [stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Upper Limit')
            if (stddevs_t[i][j-2]>UCLs2[i])&(stddevs_t[i][j]>UCLs2[i]):
                xs = [j-2,j]; ys = [stddevs_t[i][j-2],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Upper Limit')
            if (stddevs_t[i][j-2]>UCLs2[i])&(stddevs_t[i][j-1]>UCLs2[i]):
                xs = [j-2,j-1]; ys = [stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Upper Limit')
            if (stddevs_t[i][j-1]<LCLs2[i])&(stddevs_t[i][j]<LCLs2[i]):
                xs = [j-1,j]; ys = [stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Lower Limit')
            if (stddevs_t[i][j-2]<LCLs2[i])&(stddevs_t[i][j]<LCLs2[i]):
                xs = [j-2,j]; ys = [stddevs_t[i][j-2],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Lower Limit')
            if (stddevs_t[i][j-2]<LCLs2[i])&(stddevs_t[i][j-1]<LCLs2[i]):
                xs = [j-2,j-1]; ys = [stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 2 Points Near Lower Limit')
        for j in range(4,len(stddevs_t[i])):
            if (stddevs_t[i][j-3]>UCLs1[i])&(stddevs_t[i][j-2]>UCLs1[i])&(stddevs_t[i][j-1]>UCLs1[i])&(stddevs_t[i][j]>UCLs1[i]):
                xs = [j-3,j-2,j-1,j]; ys = [stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Above Center')
            if (stddevs_t[i][j-4]>UCLs1[i])&(stddevs_t[i][j-2]>UCLs1[i])&(stddevs_t[i][j-1]>UCLs1[i])&(stddevs_t[i][j]>UCLs1[i]):
                xs = [j-4,j-2,j-1,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Above Center')
            if (stddevs_t[i][j-4]>UCLs1[i])&(stddevs_t[i][j-3]>UCLs1[i])&(stddevs_t[i][j-1]>UCLs1[i])&(stddevs_t[i][j]>UCLs1[i]):
                xs = [j-4,j-3,j-1,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Above Center')
            if (stddevs_t[i][j-4]>UCLs1[i])&(stddevs_t[i][j-3]>UCLs1[i])&(stddevs_t[i][j-2]>UCLs1[i])&(stddevs_t[i][j]>UCLs1[i]):
                xs = [j-4,j-3,j-2,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Above Center')
            if (stddevs_t[i][j-4]>UCLs1[i])&(stddevs_t[i][j-3]>UCLs1[i])&(stddevs_t[i][j-2]>UCLs1[i])&(stddevs_t[i][j-1]>UCLs1[i]):
                xs = [j-4,j-3,j-2,j-1]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Above Center')
            if (stddevs_t[i][j-3]<LCLs1[i])&(stddevs_t[i][j-2]<LCLs1[i])&(stddevs_t[i][j-1]<LCLs1[i])&(stddevs_t[i][j]<LCLs1[i]):
                xs = [j-3,j-2,j-1,j]; ys = [stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Below Center')
            if (stddevs_t[i][j-4]<LCLs1[i])&(stddevs_t[i][j-2]<LCLs1[i])&(stddevs_t[i][j-1]<LCLs1[i])&(stddevs_t[i][j]<LCLs1[i]):
                xs = [j-4,j-2,j-1,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Below Center')
            if (stddevs_t[i][j-4]<LCLs1[i])&(stddevs_t[i][j-3]<LCLs1[i])&(stddevs_t[i][j-1]<LCLs1[i])&(stddevs_t[i][j]<LCLs1[i]):
                xs = [j-4,j-3,j-1,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Below Center')
            if (stddevs_t[i][j-4]<LCLs1[i])&(stddevs_t[i][j-3]<LCLs1[i])&(stddevs_t[i][j-2]<LCLs1[i])&(stddevs_t[i][j]<LCLs1[i]):
                xs = [j-4,j-3,j-2,j]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Below Center')
            if (stddevs_t[i][j-4]<LCLs1[i])&(stddevs_t[i][j-3]<LCLs1[i])&(stddevs_t[i][j-2]<LCLs1[i])&(stddevs_t[i][j-1]<LCLs1[i]):
                xs = [j-4,j-3,j-2,j-1]; ys = [stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 4 Points >1 Deviation Below Center')
        for j in range(6,len(stddevs_t[i])):
            if (stddevs_t[i][j-6]>stddevs_t[i][j-5])&(stddevs_t[i][j-5]>stddevs_t[i][j-4])&(stddevs_t[i][j-4]>stddevs_t[i][j-3])&(stddevs_t[i][j-3]>stddevs_t[i][j-2])&(stddevs_t[i][j-2]>stddevs_t[i][j-1])&(stddevs_t[i][j-1]>stddevs_t[i][j]):
                xs = [j-6,j-5,j-4,j-3,j-2,j-1]; ys = [stddevs_t[i][j-6],stddevs_t[i][j-5],stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 6 Successive Points Increasing')
            if (stddevs_t[i][j-6]<stddevs_t[i][j-5])&(stddevs_t[i][j-5]<stddevs_t[i][j-4])&(stddevs_t[i][j-4]<stddevs_t[i][j-3])&(stddevs_t[i][j-3]<stddevs_t[i][j-2])&(stddevs_t[i][j-2]<stddevs_t[i][j-1])&(stddevs_t[i][j-1]<stddevs_t[i][j]):
                xs = [j-6,j-5,j-4,j-3,j-2,j-1]; ys = [stddevs_t[i][j-6],stddevs_t[i][j-5],stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 6 Successive Points Decreasing')
        for j in range(7,len(stddevs_t[i])):
            if (stddevs_t[i][j-7]>CLs[i])&(stddevs_t[i][j-6]>CLs[i])&(stddevs_t[i][j-5]>CLs[i])&(stddevs_t[i][j-4]>CLs[i])&(stddevs_t[i][j-3]>CLs[i])&(stddevs_t[i][j-2]>CLs[i])&(stddevs_t[i][j-1]>CLs[i])&(stddevs_t[i][j]>CLs[i]):
                xs = [j-7,j-6,j-5,j-4,j-3,j-2,j-1,j]; ys = [stddevs_t[i][j-7],stddevs_t[i][j-6],stddevs_t[i][j-5],stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 8 Successive Points Above Center')
            if (stddevs_t[i][j-7]<CLs[i])&(stddevs_t[i][j-6]<CLs[i])&(stddevs_t[i][j-5]<CLs[i])&(stddevs_t[i][j-4]<CLs[i])&(stddevs_t[i][j-3]<CLs[i])&(stddevs_t[i][j-2]<CLs[i])&(stddevs_t[i][j-1]<CLs[i])&(stddevs_t[i][j]<CLs[i]):
                xs = [j-7,j-6,j-5,j-4,j-3,j-2,j-1,j]; ys = [stddevs_t[i][j-7],stddevs_t[i][j-6],stddevs_t[i][j-5],stddevs_t[i][j-4],stddevs_t[i][j-3],stddevs_t[i][j-2],stddevs_t[i][j-1],stddevs_t[i][j]]
                plt.scatter(xs, ys, color="red", s=30)
                alerts.append('Std Dev: 8 Successive Points Below Center')
        plt.title("S Chart")
        plt.tight_layout()
        plt.suptitle("Respiration Control Charts - Device ID: "+str(i),y=1.03, fontsize=14)
        for k in range(len(alerts)):
            to_append = [i,alerts[k]]
            df_length = len(alert_table)
            alert_table.loc[df_length] = to_append
    alert_table = alert_table.drop_duplicates(subset=None, keep='first', inplace=False).reset_index(drop=True)
    pd.set_option('precision', 0)
    display(alert_table)
    
def run_control_charts(baseline_data,run_data):
    baseline_data = prep_data(baseline_data)
    run_data = prep_data(run_data)
    devices, xbars, stddevs = xbars_and_stddevs(baseline_data)
    CLx, CLs = control_limits(devices, xbars, stddevs)
    control_charts(run_data, CLx, CLs)
    
if __name__ == "__main__":
    run_control_charts(sys.argv[1],sys.argv[2])