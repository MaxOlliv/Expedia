import utilitaires as util
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import pandas as pd
import time 
import pip
from scipy.optimize import minimize


def compute_matrix_pred(train,test,label):
    
    reglog = linear_model.LogisticRegression()
    reglog.fit(train[label],train['hotel_cluster'])
    prediction_reglog = reglog.predict_proba(test[label])
    
    gradb = GradientBoostingClassifier(n_estimators=10)
    gradb.fit(train[label],train['hotel_cluster'])
    prediction_gradboost = gradb.predict_proba(test[label])
    
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train[label],train['hotel_cluster'])
    prediction_knn = neigh.predict_proba(test[label])
    
    return [prediction_reglog,prediction_gradboost,prediction_knn]
    

def error(weight,matrix_pred,test):
   
    prediction_final = map(np.multiply,weight,matrix_pred)
    prediction_final = np.sum(prediction_final,axis=0)
    
    clusters = util.best_proba(prediction_final)
    
    def temp(i,val):
        actual[i] = [val]
         
    actual = range(len(test['hotel_cluster']))
    map(temp,range(len(test['hotel_cluster'])),test['hotel_cluster'])
    
    return 1-util.mapk(actual,clusters)

start_time = time.time()

print 'Lecture des fichiers'
installed_packages = pip.get_installed_distributions()
installed_packages = [i.key for i in installed_packages]
if "feather" in installed_packages == True:
    import transfo_data
    import feather
    if os.path.isfile("../Data/train_with_dest.feather")==False:
        print "Appel a transfo_data."
        transfo_data.transfo_data("../Data/train_with_dest")

    train = feather.read_dataframe('../Data/train_with_dest.feather')
    
else:
    train = pd.read_csv('../Data/train_with_dest.csv', header=0)

util.print_time(start_time)

print 'Feature engineering' 
label = util.getParam()
train,test,train_eval,test_eval = util.adaptData(train,train,label)
util.print_time(start_time)
print '{} lignes selectionnees'.format(len(train))
print '{} variables : {}'.format(len(label),label)

print 'Generation de la matrice de prediction'
matrix_pred = compute_matrix_pred(train_eval, test_eval, label)
util.print_time(start_time)

print 'Minimisation'
x0 = [1/3.,1/3.,1/3.]
arguments = matrix_pred,test_eval

start_time = time.time()
print 'Nelder-Mead'
weights = minimize(error, x0,args=arguments,method='Nelder-Mead')
util.print_time(start_time)

print 'Poids obtenus : {}'.format(weights.x)
print 'Score : {}'.format(1-weights.fun)
print weights

# print 'Powell'
# weights = minimize(error, x0,args=arguments,method='Powell')
# util.print_time(start_time)
# 
# print 'Poids obtenus : {}'.format(weights.x)
# print 'Score : {}'.format(1-weights.fun)
# print weights

