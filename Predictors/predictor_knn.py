from sklearn.neighbors import KNeighborsClassifier
import utilitaires as util
import predictor as p


def model_knn(train,test,label):
    
    neigh = KNeighborsClassifier(n_neighbors=30)
    neigh.fit(train[label],train['hotel_cluster'])
    
    prediction = neigh.predict_proba(test[label])
    
    return util.best_proba(prediction), neigh

if __name__ == '__main__':
    
    if util.getPCA():
        label= ['PC3', 'PC5', 'PC7', 'PC8', 'PC9', 'PC12', 'PC13', 'PC14', 'PC16', 'PC18', 'PC24', 'PC25']
    else:
        label = ['dest_3', 'weekend', 'ci_quarter', 'co_quarter', 'ci_dayofweek', 'hiver', 'automne', 'continent_1', 'continent_3', 'year', 'quarter', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_country', 'hotel_market']
    model = model_knn
    path = "../Predictions/knn.csv"

    p.predictor(label,model,path)