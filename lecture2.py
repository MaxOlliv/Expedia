# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:33:31 2016

@author: jean-baptiste
"""

import lecture as l
import regroupement as r


liste_algo=["a","a","a"]
liste_algo[0]="../Data/hotel1.csv"
liste_algo[1]="../Data/hotel2.csv"
liste_algo[2]="../Data/hotel3.csv"

r.concatenation_fichiers(liste_algo)
l.principal('../Data/resultat_submission.csv')
