import utilitaires as util
import os
import pandas as pd
import time 
import pip
from sklearn.externals import joblib

#ARGUMENTS :
#func_train : fonction a appliquer aux donnees d'apprentissage
#func_test : fonction a appliquer aux donnees de test
#label : variables de prediction
#model : fonction modele utilise pour la prediction
#path : chemin pour l'exportation en csv

def predictor(label,model,path,gen_csv=True):

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
        if os.path.isfile("../Data/test_data_leak.feather")==False:
            print "Appel a transfo_data."
            transfo_data.transfo_data("../Data/test_data_leak")

        train = feather.read_dataframe('../Data/train_with_dest.feather')
        test  = feather.read_dataframe('../Data/test_with_dest.feather')
        leak = feather.read_dataframe('../Data/test_data_leak.feather')

    else:
        train = pd.read_csv('../Data/train_with_dest.csv', header=0)
        test = pd.read_csv('../Data/test_with_dest.csv', header=0)
        leak = pd.read_csv('../Data/test_data_leak.csv',header=0)
    util.print_time(start_time)

    print 'Feature engineering' 
    train,test,train_eval,test_eval = util.adaptData(train,test,label)
    util.print_time(start_time)
    print '{} lignes selectionnees'.format(len(train))
    print '{} variables : {}'.format(len(label),label)
    
    print 'Validation temporelle'
    taux = util.temporal_validation(train_eval,test_eval,model,label)
    util.print_time(start_time)
    print "Taux de predictions correctes pour les parametres actuels : {0}".format(taux)
    
    if gen_csv:
        print 'Apprentissage & Prediction'
        prediction,modele = model(train,test,label)
        
#         print 'Exportation du modele conserve'
#         nom_export = "{0}.bin".format(modele).split("(")[0]
#         joblib.dump(modele,nom_export)
#         util.print_time(start_time)
        
        print 'Data leak'
        for i in range (len(leak)):
            prediction[leak['id'][i]] = util.leak_fusion(leak['hotel_cluster'][i],prediction[leak['id'][i]])
            
        test['hotel_cluster'] = util.to_string(prediction)
        util.print_time(start_time)
        
        print 'Generation du fichier csv'
        test[["id","hotel_cluster"]].to_csv(path, index=False)
        util.print_time(start_time)
    print 'termine'
    
    return taux