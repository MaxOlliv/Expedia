import utilitaires as util
from sklearn.ensemble import AdaBoostClassifier
import predictor as p 

def model_adaboost(train,test,label):

    adab = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
    
    adab.fit(train[label],train['hotel_cluster'])

    prediction = adab.predict_proba(test[label])
    
    return util.best_proba(prediction), adab
    
if __name__ == '__main__':
    
    label = util.getParam()
    model = model_adaboost
    path = "../Predictions/adaboost.csv"

    p.predictor(label,model,path)
