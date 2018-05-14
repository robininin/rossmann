import sys
sys.path.append("/anaconda3/envs/udacity_tensorflow_py3/lib/python3.5/site-packages")
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import math
from isoweek import Week

with open("combined_fb_train.pickle","rb") as f:
    df_combined_train = pickle.load(f)
    
with open("combined_fb_test.pickle","rb") as f:
    df_combined_test = pickle.load(f)

store_with_sales = (df_combined_train["Sales"]!=0)
df_combined_train = df_combined_train[store_with_sales]

df_target = df_combined_train.pop("Sales")
df_combined_train.pop("Customers")

#split training data into train and dummy test
train_ratio = 0.97
num_records = len(df_combined_train)
features_train = df_combined_train[0:int(num_records*train_ratio)]
target_train = df_target[0:int(num_records*train_ratio)]
features_dummy_test = df_combined_train[int(num_records*train_ratio):]
target_dummy_test = df_target[int(num_records*train_ratio):]

def abcd2int(storetype):
    #used on StateHoliday, StoreType, Assortment
    d = {'0': 3, 0: 3, 'a': 0, 'b': 1, 'c': 2, 'd': 3}
    return d[storetype]

def year2int(year):
    #use apply in dataframe
    dict = {2013:0,2014:1,2015:2}
    return dict[year]

def datetoymd(date):
    #date is dataframe series
    date = pd.to_datetime(date)
    year = date.dt.year
    month = date.dt.month
    day = date.dt.day
    week_of_year = date.apply(lambda x:(x.isocalendar()[1]))
    year = year.apply(year2int)
    return year,month,day,week_of_year

def compdistance(distance):
    distance = math.log(distance+1)/10.0
    return distance

def hasCompetitionmonths(date, CompetitionOpenSinceYear, CompetitionOpenSinceMonth):
    date = np.array(date)
    compyear = np.array(CompetitionOpenSinceYear)
    compmonth = np.array(CompetitionOpenSinceMonth)
    num = len(date)
    sincemonth = np.zeros([num],dtype=int)
    for i in range(num):
        
        if compyear[i] == "NA":
            sincemonth[i] = 0
        else:
            competition_open = datetime(year=int(compyear[i]),
                                   month=int(compmonth[i]),
                                   day=15)
            sincemonth[i] = (datetime.strptime(date[i], '%Y-%m-%d') - competition_open).days // 30
        if sincemonth[i] < 0:
            sincemonth[i] = 0
        sincemonth[i] = int(min(sincemonth[i], 24))
    return sincemonth

def hasPromo2weeks(date, Promo2SinceYear, Promo2SinceWeek):
    #calculate how many weeks that it has been since launching promo2. Categorized into buckets if it's more than 25 weeks
    date = np.array(date)
    promoyear = np.array(Promo2SinceYear)
    promoweek = np.array(Promo2SinceWeek)
    num = len(date)
    sinceweek = np.zeros([num],dtype=int)
    for i in range(num):
        if promoyear[i] == "NA":
            sinceweek[i] = 0
        else:
            start_promo2 = Week(int(promoyear[i]), int(promoweek[i])).monday()
            sinceweek[i] = (datetime.strptime(date[i], '%Y-%m-%d').date() - start_promo2).days // 7
            if sinceweek[i] < 0:
                sinceweek[i] = 0
            elif sinceweek[i]<=25:
                pass
            elif sinceweek[i]<=100:
                sinceweek[i] = 26
            elif sinceweek[i]<=200:
                sinceweek[i] = 27
            else:
                sinceweek[i] = 28
    return sinceweek

def latest_promo2_months(date, promointerval, Promo2SinceYear, Promo2SinceWeek):
    #the number of months since last promo. If no PromoInterval or hasn't joined promo2, return 0. Output set:[0,1,2,3]
    promo2int = promointerval2int(promointerval)
    promo2int = np.array(promo2int)
    date = np.array(date)
    promoyear = np.array(Promo2SinceYear)
    promoweek = np.array(Promo2SinceWeek)
    
    months_since_latest_promo2 = np.zeros([len(date)],dtype=int)
    for i in range(len(date)):  
        if promo2int[i] == 0:
            months_since_latest_promo2[i] = 0
            continue
            
        date[i] = datetime.strptime(date[i], '%Y-%m-%d').date()
        start_promo2 = Week(int(promoyear[i]), int(promoweek[i])).monday()
        if date[i] < start_promo2:
            months_since_latest_promo2[i] = 0
            continue
            
        if date[i].month < promo2int[i]:
            latest_promo2_start_year = date[i].year - 1
            latest_promo2_start_month = promo2int[i] + 12 - 3
        else:
            latest_promo2_start_year = date[i].year
            latest_promo2_start_month = ((date[i].month - promo2int[i]) // 3) * 3 + promo2int[i]

        latest_promo2_start_day = datetime(year=latest_promo2_start_year,
                                       month=latest_promo2_start_month,
                                       day=1)
        months_since_latest_promo2[i] = (date[i] - latest_promo2_start_day.date()).days // 30
    return months_since_latest_promo2

def promointerval2int(promointerval):
    d = {'NA': 0, 'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
    result = promointerval.apply(lambda x:d[x])
    return result

def state2int(state):
    d = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5,
         'SN': 6, 'BE': 7, 'HE': 8, 'SH': 9, 'BY': 10, 'NW': 11}
    return state.apply(lambda x:d[x])

def prep_features(df):
    state_holiday = df["StateHoliday"].apply(abcd2int)
    store_type = df["StoreType"].apply(abcd2int)
    assortment = df["Assortment"].apply(abcd2int)
    year, month, day, week_of_year = datetoymd(df["Date"])
    comp_distance = df["CompetitionDistance"].apply(compdistance)
    comp_since_total_month = hasCompetitionmonths(df["Date"], df["CompetitionOpenSinceYear"], df["CompetitionOpenSinceMonth"])
    promo_since_total_week = hasPromo2weeks(df["Date"], df["Promo2SinceYear"], df["Promo2SinceWeek"])
    promo_since_last_interval = latest_promo2_months(df["Date"],df["PromoInterval"], df["Promo2SinceYear"], df["Promo2SinceWeek"])
    promo_interval = promointerval2int(df["PromoInterval"])
    states = state2int(df["State"])

    features = {"store" : np.array(df["Store"],dtype=int),
               "dow" : np.array(df["DayOfWeek"],dtype=int),
               "year" : np.array(year,dtype=int),
               "month" : np.array(month,dtype=int),
               "day" : np.array(day,dtype=int),
               "promo" : np.array(df["Promo"]),
               "state_holiday" : np.array(state_holiday,dtype=int),
               "school_holiday" : np.array(df["SchoolHoliday"],dtype=int),
               "store_type" : np.array(store_type,dtype=int),
               "assortment" : np.array(assortment,dtype=int),
               "comp_distance" : np.array(comp_distance),
               "comp_since_total_month" : np.array(comp_since_total_month,dtype=int),
               "promo_since_total_week" : np.array(promo_since_total_week,dtype=int),
               "promo2" : np.array(df["Promo2"]),
               "promo_since_last_interval" : np.array(promo_since_last_interval,dtype=int),
               "promo_interval" : np.array(promo_interval,dtype=int),
               "states" : np.array(states,dtype=int),
               "week_of_year" : np.array(week_of_year,dtype=int),
               "backward_open" : np.array(df["backward_open"],dtype=int),
               "forward_open" : np.array(df["forward_open"],dtype=int),
                "backward_promo" : np.array(df["backward_promo"],dtype=int),
               "forward_promo" : np.array(df["forward_promo"],dtype=int),
                "backward_school" : np.array(df["backward_school"],dtype=int),
               "forward_school" : np.array(df["forward_school"],dtype=int)}
    return features

X_train = prep_features(features_train)
X_dummy_test = prep_features(features_dummy_test)
X_test = prep_features(df_combined_test)
X_all_train = prep_features(df_combined_train)

target_train = np.array(target_train,dtype=int)
target_dummy_test = np.array(target_dummy_test,dtype=int)
df_target = np.array(df_target,dtype=int)

with open("X_train.pickle","wb") as f:
	pickle.dump((X_train,target_train),f,-1)

with open("X_dummy_test.pickle","wb") as f:
	pickle.dump((X_dummy_test,target_dummy_test),f,-1)

with open("X_test.pickle","wb") as f:
	pickle.dump(X_test,f,-1)

with open("X_all_train.pickle","wb") as f:
	pickle.dump((X_all_train,df_target),f,-1)

