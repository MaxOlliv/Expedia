# -*- coding: utf-8 -*-

import csv as csv 
import pandas as pd
import utilitaires as util
import numpy as np
    
def to_string (resultat):
    final=str(resultat[0])
    i=1
    while (i<5):
        if (resultat[i] != -1):
            final=final+" "+str(resultat[i])
        i=i+1
    return final
    
    
def indice_max(liste):
    i=0
    indice=-1
    valeur=0
    while i<len(liste):
        if liste[i]>valeur:
            valeur=liste[i]
            indice=i
        i=i+1
    return indice
        
def cinq_indices(liste):
    resultat=[-1]*5
    i=0
    while i<5:
        resultat[i]=indice_max(liste)
        liste[resultat[i]]=0
        i=i+1
    return resultat

def traitement_ligne(ligne,liste_coefficients,fonction):
    nb_occurences=[0]*100
    taille=1
    while taille<len(ligne):
        x=parseur(ligne[taille])
        i=0
        while i<5:
            if (x[i]!=-1):
                nb_occurences[x[i]]=nb_occurences[x[i]]+fonction(i)*liste_coefficients[taille-1]
            i=i+1
        taille=taille+1
    resultat=cinq_indices(nb_occurences)
    resultat=to_string(resultat)
    #print resultat
    return resultat
       


def parseur(case):
    hotel=[-1]*5
    taille=0
    nb_espaces=0
    nb=0
    negatif=1
    while taille<len(case):
        y=case[taille]
        if (y==" "):
            hotel[nb_espaces]=nb
            nb_espaces=nb_espaces+1
            nb=0
        elif (y=="-"):
            negatif=-1
        else:
            nb=(10*nb+int(y))*negatif
        taille=taille+1
    hotel[nb_espaces]=nb
    return hotel
            
def ouverture_fichier(chemin):
    csv_file_object = csv.reader(open(chemin, 'rb')) 
    csv_file_object.next() 
    data=[]                         
    for row in csv_file_object:      
        data.append(row)             
    data = np.array(data) 
    return data	     

def ecriture_fichier(chemin):
    ecriture = csv.writer(open(chemin, "wb"))
    ecriture.writerow(["id","hotel_cluster"])
    return ecriture

def principal(chemin,start_time,liste_coefficient,fonction):
    data=ouverture_fichier('../Data/fichier_resultat.csv')
    ecriture=ecriture_fichier(chemin)
    print "fichier de resultat ouvert et fichier d'ecriture cree"
    util.print_time(start_time)
    nb_lignes=0
    print "traitement des lignes en cours"
    while nb_lignes<len(data):
        prediction=traitement_ligne(data[nb_lignes],liste_coefficient,fonction)
        ecriture.writerow([data[nb_lignes][0],prediction])
        nb_lignes=nb_lignes+1
    print "termine"
    util.print_time(start_time)
    
def concatenation_fichiers(liste_algo,start_time):
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
    print "fichier recapitulatif cree"
    util.print_time(start_time)

def lineaire(i):
    return (5-i)

def exponentiel(i):
    return (2**(5-i))
    
def classique(i):
    if(i==0):
        return 11
    elif(i==1):
        return 7
    elif(i==2):
        return 4
    elif(i==3):
        return 2
    else:
        return 1
    
    
def classique2(i):
    if(i==0):
        return 21
    elif(i==1):
        return 13
    elif(i==2):
        return 7
    elif(i==3):
        return 3
    else:
        return 1
        
def constante(i):
    return 1
    
def logarithme(i):
    return (1-1/(2**(5-i)))
    
def trois_premiers(i):
    if(i==0):
        return 30
    elif(i==1):
        return 20
    elif(i==2):
        return 10
    elif(i==3):
        return 1
    else:
        return 1

def classique3(i):
    if(i==0):
        return 5
    elif(i==1):
        return 3
    elif(i==2):
        return 2
    elif(i==3):
        return 1
    else:
        return 1

        
    