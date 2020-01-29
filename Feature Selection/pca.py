from sklearn.decomposition import PCA
import pandas as pd
import csv


def pca_dest():
    destinations = pd.read_csv("../Data/destinations.csv")
    
    pca = PCA(n_components=3)
    dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
    dest_small = pd.DataFrame(dest_small)
    dest_small["srch_destination_id"] = destinations["srch_destination_id"]
    
    dest_small.to_csv("../Data/dest_small.csv", index=False)