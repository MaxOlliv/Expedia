import utilitaires as util
from sklearn.ensemble import GradientBoostingClassifier
import predictor as p 

def model_gradboost(train, test, label):

    gradb = GradientBoostingClassifier(n_estimators=10)
    
    gradb.fit(train[label],train['hotel_cluster'])

    prediction = gradb.predict_proba(test[label])
    
    return util.best_proba(prediction) , gradb
    
if __name__ == '__main__':
    
    na = util.getNumAdaptData()
    if na == 1:
        label = ['weekend', 'co_month', 'co_dayofweek', 'ci_quarter', 'co_quarter', 'stay_span', 'duree_prevoyance', 'ci_day', 'ci_month', 'co_day', 'stay_span', 'hiver', 'automne', 'continent_0', 'continent_1', 'continent_3', 'continent_4', 'year', 'quarter', 'user_location_country', 'user_location_region', 'user_id', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_country', 'hotel_market']
    else:
        label = util.getParam()
    model = model_gradboost
    path = "../Predictions/gradb.csv"

    p.predictor(label,model,path)