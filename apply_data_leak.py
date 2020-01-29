import pandas as pd
import utilitaires as util
import time

#TODO: A DEBUGGER

name = 'gradb'

start_time = time.time()

print 'Lecture des fichiers' 
pred = pd.read_csv('../Predictions/{}.csv'.format(name), header=0)
leak = pd.read_csv('../Data/test_data_leak.csv',header=0)
util.print_time(start_time)

print 'Data leak'
for i in range (len(leak)):
    pred.loc[leak['id'][i],'hotel_cluster'] = leak['hotel_cluster'][i] 
util.print_time(start_time)

print 'Generation du fichier csv'
pred[["id","hotel_cluster"]].to_csv('../Predictions/{}_data_leak.csv'.format(name), index=False)
util.print_time(start_time)
print 'termine'