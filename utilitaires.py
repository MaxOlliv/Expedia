from sklearn.cross_validation import KFold
import random
import numpy as np
import sklearn.linear_model as rln
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

import time
import sklearn.decomposition as dec


#Validation 10 fold
def iteration(train,test,model,label,grid_elt=None):
    def temp(i,val):
        actual[i] = [val]
    if grid_elt is None:
        prediction,modele = model(train,test,label)
    else :
        prediction,modele = model(train,test,label,grid_elt)
         
    actual = range(len(test['hotel_cluster']))
    map(temp,range(len(test['hotel_cluster'])),test['hotel_cluster'])
    return mapk(actual,prediction)

def tenfold(data,label,model,aff=False):

    kf_index = KFold(len(data),n_folds=10) 
    kf = np.empty((10,2),dtype=object)
    
    for i,(train,test) in enumerate(kf_index):
        kf[i] = [data.iloc[train],data.iloc[test]]
        
    map_res = np.empty((10,1),dtype=object)
    for i,(test,train) in enumerate(kf):
        if aff:
            print '{} sur 10'.format(i+1)
        map_res[i] = iteration(train,test,model,label)

    return np.mean(map_res)

#Validation avec decoupage temporel 
def temporal_validation(train_val,test_val,model,label, grid_elt=None):    
    return iteration(train_val,test_val,model,label,grid_elt)

#Gestion des variables qualitatives et des NaN

numAdaptData = 1
do_pca = False
pca_component = 42


#RETIRES : 'orig_destination_distance','co_month','co_quarter','co_dayofweek','ci_day','ci_dayofweek','ci_month','ci_quarter','co_day','stay_span','hiver','printemps','ete','automne'
param_pca = ['PC{}'.format(i+1) for i in range(pca_component)]
param1 = ['dest_1','dest_2','dest_3','weekend','co_month','co_dayofweek','ci_quarter','co_quarter','stay_span','duree_prevoyance','ci_day','ci_dayofweek','ci_month','co_day','stay_span','hiver','printemps','ete','automne','continent_0','continent_1','continent_2','continent_3','continent_4','year','month','quarter','user_location_country','user_location_region','user_location_city','user_id','is_mobile','is_package','channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']
param1_lolo = ['year','month','quarter','continent_0','continent_1','continent_2','continent_3','continent_4','user_location_country','user_location_region','user_location_city','user_id','is_mobile','is_package','channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']


def getNumAdaptData():
    return numAdaptData

def getPCA():
    return do_pca

def getParam():
    if do_pca:
        return param_pca
    elif numAdaptData == 1:
        param = param1
    else:
        param = [] 
        print 'numAdaptData a corriger dans utilitaires'
    return param

def adaptData(train,test,param=getParam()):
    if numAdaptData == 1:
        return adaptData1(train,test,param)
    else:
        return

#Plusieurs possibilites d'adaptation des donnees 
def adaptData1(train,test,param):
  
#     train = continent(train)
#     test = continent(test)
#     train = calc_fast_features(train)
#     test = calc_fast_features(test)
    
    unique_users = train.user_id.unique()
    sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000))]
    train = train[train.user_id.isin(sel_user_ids)]
     
    train = saison(train)
    test = saison(test)
    
    train = weekend(train)
    test = weekend(test)
    
    if do_pca:
        p = dec.PCA(n_components=pca_component)
        p.fit(train[param1])
        print 'PCA effectuee'
        print 'variance expliquee : {}'.format(p.explained_variance_ratio_)
        
        train_pca = p.transform(train[param1])
        test_pca = p.transform(test[param1])
        
        for index,pc in enumerate(param_pca):
            train[pc] = train_pca[:,index]
            test[pc] = test_pca[:,index]
        
    train_eval = train[((train.year == 2013) | ((train.year == 2014) & (train.month < 8)))]
    test_eval = train[((train.year == 2014) & (train.month >= 8))]

    return train,test,train_eval,test_eval


# Encodage "one-hot" pour les continents
def continent(data):
    enc = OneHotEncoder()
    temp = enc.fit_transform(data[["posa_continent"]])
    temp2 = pd.DataFrame(temp.toarray(),columns=["continent_0","continent_1","continent_2","continent_3","continent_4"])

    # On concatene data avec temp2
    # On commence par drop la colonne "posa_continent"
    data = data.drop('posa_continent', axis=1)
    data = pd.concat([temp2,data],axis=1)
    return data


def apk(actual, predicted):
    if len(predicted)>5:
        predicted = predicted[:5]

    score = 0.0
    num_hits = 0.0
    
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0
    return score / min(len(actual), 5)

def mapk(actual, predicted):
    return np.mean(map(apk,actual,predicted))

def argmax_five(line):
    argmax_table = [-1,-1,-1,-1,-1]
    val_table = [0.0,0.0,0.0,0.0,0.0]
    for i in range(len(line)):
        for j in range(5):
            if line[i]>val_table[j]:
                for k in range(4,j,-1):
                    val_table[k] = val_table[k-1]
                    argmax_table[k] = argmax_table[k-1]
                val_table[j] = line [i]
                argmax_table[j] = i
                break
    #val_table = [x for x in val_table if x >= 0.001]
    #return argmax_table[:max(len(val_table),1)]
    return argmax_table

def best_proba(prediction):
    result = range(len(prediction))
#     import multiprocessing as mp 
#     pool = mp.Pool()
#     result = pool.map(argmax_five,prediction)
    for i,line in enumerate(prediction):
        result[i] = argmax_five(line)
#     pool.close()
    return result

def to_string (prediction):
    result = range(len(prediction))
    for i,line in enumerate(prediction):
        if not line :
            result[i]= np.nan
        else :
            temp_str = '{}'.format(line[0])
            for j in range(1,len(line)):
                temp_str += ' {}'.format(line[j])
            result[i] = temp_str
        
    return result
    
def calc_fast_features(df):
    df.is_copy = False
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month    
     
    dest_small=pd.read_csv("../Data/dest_small.csv")
     
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
     
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
     
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
 
    date_props = ["month", "day", "dayofweek", "quarter","year"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')

    props["duree_prevoyance"]=(df["srch_ci"] - df["date_time"]).astype('timedelta64[h]') 
    ret = pd.DataFrame(props)
     
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
 
    ret = ret.drop("srch_destination_iddest", axis=1)

    return ret
    
def find_matches(row,match_param,groups):
#     for idx in train_size_red.index:
#         res = True
#         for col in match_param:
#             res = res & (row[col] == train_size_red.ix[idx][col])
#         if res:
#             return [train_size_red.ix[idx]['hotel_cluster'].astype(int)]
#     return []
    index = tuple([row[t] for t in match_param])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    cluster = list(set(group.hotel_cluster)) 
    if not cluster :
        return []
    else:
        #return [[num] for num, count in Counter(clus).most_common(5)]
        return cluster
#         return [max(set(clus), key=clus.count)]
    
def dataLeak(train,test):
    match_param1 = ['user_location_country', 'user_location_region','user_location_city', 'hotel_market', 'orig_destination_distance','hotel_cluster']
    match_param2 = ['user_location_country', 'user_location_region','user_location_city', 'hotel_market', 'orig_destination_distance']
    train = train[pd.notnull(train['orig_destination_distance'])]
    print '-> calcul des frequences de clusters'
    train_size =  pd.DataFrame({'size' : train.groupby(match_param1).size()}).reset_index()
    train_size_red = train_size.loc[train_size.groupby(match_param2)['size'].idxmax()]
    groups = train_size_red.groupby(match_param2)
    exact_matches = np.empty((len(test),1),dtype=object)
    print '-> recherche de correspondances'
    l = len(test)
    prct = 1
    for i in range(test.shape[0]):
        temp = find_matches(test.iloc[i],match_param2,groups)
        if len(temp)>0:
            exact_matches[i]=temp
        if float(i+1)/float(l)*100.0>=prct:
            print '--- {}% effectues'.format(prct)
            prct += 1
    return exact_matches
              
def group_result(seq_leak,seq):
    result = []
    if (seq_leak=='null'):
        result.append(seq)
    else :
        print('test')
        result.append(seq_leak)
    return result

def leak_fusion(leak,row):
    row = [x for x in row if x != leak]
    return [leak,row[0],row[1],row[2],row[3]]

def print_time(start_time):
    total = time.time()-start_time
    if total<60:
        print 'temps ecoule : {}s'.format(int(total))
    elif total<3600:
        m = int(total/60)
        s = int(total%60)
        print 'temps ecoule : {}m {}s'.format(m,s)
    else :
        h = int(total/3600)
        m = int((total%3600)/60)
        s = int(total%60)
        print 'temps ecoule : {}h {}m {}s'.format(h,m,s)
    return

def saison(data):
    data.is_copy = False
    data.loc[(data["ci_month"] > 12) | (data["ci_month"] <= 3),"saison"] = "hiver"
    data.loc[(data["ci_month"] > 3) & (data["ci_month"] <= 6),"saison"] = "printemps"
    data.loc[(data["ci_month"] > 6) & (data["ci_month"] <= 9),"saison"] = "ete"
    data.loc[(data["ci_month"] > 9) & (data["ci_month"] <= 12),"saison"] = "automne"

    lab = LabelEncoder()
    data["saison"] = lab.fit_transform(data["saison"])
    label_binarizer = LabelBinarizer()
    saison = label_binarizer.fit_transform(data.saison)
    data['hiver']=saison[:,1]
    data['printemps']=saison[:,0]
    data['automne']=saison[:,3]
    data['ete']=saison[:,2]
    data = data.drop('saison', axis=1)

    return data

def weekend(data):
    
    data.is_copy = False
    data.loc[(data["ci_dayofweek"] < 5),"tempd"] = "semaine"
    data.loc[(data["ci_dayofweek"] >= 5),"tempd"] = "weekend"
    label_encoder = LabelEncoder()
    data["tempd"] = label_encoder.fit_transform(data["tempd"])
    label_binarizer = LabelBinarizer()
    tempd = label_binarizer.fit_transform(data.tempd)
    data['weekend']=tempd[:,0]
    data = data.drop('tempd', axis=1)
    
    return data

def guess_Nan(data):
    col = ["ci_year","co_year","ci_month","ci_day","ci_dayofweek","co_dayofweek",\
           "co_month","co_day",'co_quarter','ci_quarter',\
           'stay_span','duree_prevoyance']
    parametres= ["ci_year","co_year","ci_month","ci_day","ci_dayofweek","co_dayofweek","co_month","co_day",\
                 'co_quarter','ci_quarter','stay_span','duree_prevoyance',\
                 'continent_0','continent_1','continent_2','continent_3','continent_4','year','month',\
                 'quarter','user_location_country','user_location_region','user_location_city','user_id',\
                 'is_mobile','is_package','channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt',\
                 'srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']
    
    data=data[parametres]
    incr=0
    data.is_copy=False
    
    for i in col:
        
        train = data.loc[(data[i].notnull())]
        test = data.loc[(data[i].isnull())]
        
        if len(test)>0 :
            print i
            y = train.values[:,incr] 
            x = train.values[:,(len(col))::]
            
            reglin = rln.LinearRegression()
            reglin.fit(x, y)
            predicted = reglin.predict(test.values[:,len(col)::])
            data.loc[data[i].isnull(), i] =predicted
            incr+=1


            
    return data
