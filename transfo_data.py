import pandas as pd
import feather 
# Pour avoir feather  pip install feather-format

def transfo_data(path):
	
	# Importation des donnees en csv
	print "Importation des donnees en csv"
	train = pd.read_csv(path+".csv",header=0)

	# Exportation en feather
	print "Exportation en donnees feather"
	feather.write_dataframe(train,path+".feather")
	