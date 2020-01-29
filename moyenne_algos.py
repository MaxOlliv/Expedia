# -*- coding: utf-8 -*-

import lecture as l
import time
start_time = time.time()

liste_algo=[]

liste_algo.append("../prediction_kaggle/xgb27965.csv")
liste_algo.append("../prediction_kaggle/gradb28401.csv")
liste_algo.append("../prediction_kaggle/gradb28792.csv")
liste_algo.append("../prediction_kaggle/gradb28752.csv")
liste_algo.append("../prediction_kaggle/xgb28657.csv")
liste_algo.append("../prediction_kaggle/knn34528.csv")
liste_algo.append("../prediction_kaggle/knn35242.csv")
liste_algo.append("../prediction_kaggle/knn35398.csv")


liste_coefficient=[]

liste_coefficient.append(1)
liste_coefficient.append(1)
liste_coefficient.append(1)
liste_coefficient.append(1)
liste_coefficient.append(1)
liste_coefficient.append(6)
liste_coefficient.append(6)
liste_coefficient.append(7)


fonction_valeur=l.classique

l.concatenation_fichiers(liste_algo,start_time)
l.principal('../Predictions/resultat_submission.csv',start_time,liste_coefficient,fonction_valeur)
