import pandas as pd
import utilitaires as util
import numpy as np
from joblib import Parallel, delayed
import time
import os 
import pip



def iteration(i,loop,subset,train,model):
            
    label = np.asarray(subset)
    temp = util.temporal_validation(train,model,label) 
        
    return temp

def find_subset(list_start):
    list_res = []
    for i in range(len(list_start)):
        list_res.append(list_start[:i]+list_start[i+1:])
    return list_res

def init(param):
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
        if os.path.isfile("../Data/test_with_dest.feather")==False:
            print "Appel a transfo_data."
            transfo_data.transfo_data("../Data/test_with_dest")

        train = feather.read_dataframe('../Data/train_with_dest.feather')
        test  = feather.read_dataframe('../Data/test_with_dest.feather')

    else:
        train = pd.read_csv('../Data/train_with_dest.csv', header=0)
        test = pd.read_csv('../Data/test_with_dest.csv', header=0)
    util.print_time(start_time)
    
    print 'Feature engineering' 
    train,test,train_eval,test_eval = util.adaptData(train,test)
    util.print_time(start_time)
    print '{} lignes selectionnees'.format(len(train))
    print '{} variables : {}'.format(len(param),param)
    
    return start_time,train_eval,test_eval

#ARGUMENTS :
#param : ensemble des variables pouvant servir de label
#model : fonction modele utilise pour la prediction

def best_formula(param,model):
    
    start_time,train_eval,test_eval = init(param)
    
    loop = float(len(param)*(len(param)-1))/2.;
    print "{} calculs au maximum".format(int(loop))
    
    print 'initialisation'
    best_subset = param
    best_score = util.temporal_validation(train_eval,test_eval,model,param)
    first_score = best_score
    util.print_time(start_time)
    print 'score obtenu : {}'.format(best_score)
    
    current = 1
    while len(best_subset)>1:
        print '\n-> etape {}'.format(current)
        current += 1
        subset_list = find_subset(best_subset)
        tem_val = Parallel(n_jobs=-1,verbose=50)(delayed(util.temporal_validation)(pd.DataFrame.copy(train_eval),pd.DataFrame.copy(test_eval),model,subset) for subset in subset_list)
        util.print_time(start_time)
        argmax = np.argmax(tem_val)  
        if tem_val[argmax]<best_score:
            break
        best_subset = subset_list[argmax]
        diff = tem_val[argmax]-best_score
        best_score = tem_val[argmax]
        print 'score actuel : {}'.format(best_score)
        print 'amelioration de {}'.format(diff)
        print 'parametres actuels : {}'.format(best_subset)
        
    print("")
    util.print_time(start_time)
    print("Resultat :")
    print "Meilleur taux obtenu : {}".format(best_score)
    print 'amelioration totale de {}'.format(best_score-first_score)
    print("pour les parametres suivants :")
    print (best_subset)
    
    return

def best_formula_grid(param,model,start_time,train_eval,test_eval,grid_elt):
    
    loop = float(len(param)*(len(param)-1))/2.;
    print "{} calculs au maximum".format(int(loop))
    
    print 'initialisation'
    best_subset = param
    best_score = util.temporal_validation(train_eval,test_eval,model,param)
    first_score = best_score
    util.print_time(start_time)
    print 'score obtenu : {}'.format(best_score)
    
    current = 1
    while len(best_subset)>1:
        print '\n-> etape {}'.format(current)
        current += 1
        subset_list = find_subset(best_subset)
        tem_val = Parallel(n_jobs=-1,verbose=50)(delayed(util.temporal_validation)(pd.DataFrame.copy(train_eval),pd.DataFrame.copy(test_eval),model,subset,grid_elt) for subset in subset_list)
        util.print_time(start_time)
        argmax = np.argmax(tem_val)  
        if tem_val[argmax]<best_score:
            break
        best_subset = subset_list[argmax]
        diff = tem_val[argmax]-best_score
        best_score = tem_val[argmax]
        print 'score actuel : {}'.format(best_score)
        print 'amelioration de {}'.format(diff)
        print 'parametres actuels : {}'.format(best_subset)
        
    print("")
    util.print_time(start_time)
    print("Resultat :")
    print "Meilleur taux obtenu : {}".format(best_score)
    print 'amelioration totale de {}'.format(best_score-first_score)
    print("pour les parametres suivants :")
    print (best_subset)
    
    return best_score,best_subset