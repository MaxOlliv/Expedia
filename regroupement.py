# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:56:54 2016

@author: jean-baptiste
"""

import pandas as pd



def concatenation_fichiers(liste_algo):
    liste_df=liste_algo
    i=0
    while i<len(liste_algo):
        nom=pd.read_csv(liste_algo[i], header=0)
        liste_df[i]=pd.DataFrame(nom)
        i=i+1
       
    final=pd.merge(liste_df[0],liste_df[1],on='id')
    i=2
    while i<len(liste_algo):
        final=pd.merge(final,liste_df[i],on='id')
        i=i+1
    
    del final["id"]
    final.to_csv('../Data/fichier_resultat.csv')









"""
liste_algo=["a","a","a"]
liste_df=liste_algo
liste_algo[0]="../Data/hotel1.csv"
liste_algo[1]="../Data/hotel2.csv"
liste_algo[2]="../Data/hotel3.csv"

i=0
#print len(liste_algo)
while i<len(liste_algo):
    nom=pd.read_csv(liste_algo[i], header=0)
    liste_df[i]=pd.DataFrame(nom)
    i=i+1


final=pd.merge(liste_df[0],liste_df[1],on='id')
i=2
while i<len(liste_algo):
    final=pd.merge(final,liste_df[i],on='id')
    i=i+1

del final["id"]
final.to_csv('../Data/example.csv')
"""