import utilitaires as util
from xgboost import sklearn
import predictor as p 
import pandas as pd

def model_xgb(train, test, label):

    xgb = sklearn.XGBClassifier(nthread=4, n_estimators=10)

    xgb.fit(train[label],train['hotel_cluster'])

    prediction = xgb.predict_proba(test[label])
    
    df = pd.DataFrame(prediction).transpose().tail(test[label].shape[0])
    
    return util.best_proba(df.as_matrix()) , xgb
    
if __name__ == '__main__':
    
    na = util.getNumAdaptData()
    if na == 1:
        label = ['weekend', 'co_month', 'co_dayofweek', 'ci_quarter', 'co_quarter', 'stay_span', 'duree_prevoyance', 'ci_day', 'ci_month', 'co_day', 'hiver', 'automne', 'continent_0', 'continent_1', 'continent_3', 'continent_4', 'year', 'quarter', 'user_location_country', 'user_location_region', 'user_id', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_country', 'hotel_market']
    else:
        label = util.getParam()
    model = model_xgb
    path = "../Predictions/xgb.csv"

    p.predictor(label,model,path)