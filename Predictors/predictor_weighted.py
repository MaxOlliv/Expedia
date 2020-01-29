import utilitaires as util
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import predictor as p


def model_weighted(train,test,label):
    
    weight = []
    
    reglog = linear_model.LogisticRegression()
    reglog.fit(train[label],train['hotel_cluster'])
    prediction_reglog = reglog.predict_proba(test[label])
    
    gradb = GradientBoostingClassifier(n_estimators=10)
    gradb.fit(train[label],train['hotel_cluster'])
    prediction_gradboost = gradb.predict_proba(test[label])
    
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train[label],train['hotel_cluster'])
    prediction_knn = neigh.predict_proba(test[label])
    
    matrix_pred = [prediction_reglog,prediction_gradboost,prediction_knn]
   
    prediction_final = map(np.multiply,weight,matrix_pred)
    prediction_final = np.sum(prediction_final,axis=0)
    
    return util.best_proba(prediction_final)

if __name__ == '__main__':
    
    if util.getPCA():
        label = util.getParam()
    else:
        label = util.getParam()
        
    model = model_weighted
    path = "../Predictions/weighted.csv"

    p.predictor(label,model,path)