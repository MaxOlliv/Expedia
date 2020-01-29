import sklearn.ensemble as ske
import utilitaires as util 
import predictor as p 

def model_rforest(train,test,label):
    rf = ske.RandomForestClassifier(n_estimators=10,criterion="gini")
    
    rf.fit(train[label],train['hotel_cluster'])

    prediction = rf.predict_proba(test[label])
    
    return util.best_proba(prediction), rf

if __name__ == '__main__':
    
    label = ['dest_1', 'dest_2', 'dest_3', 'weekend', 'co_month', 'co_dayofweek', 'ci_quarter', 'co_quarter', 'stay_span', 'ci_day', 'ci_dayofweek', 'ci_month', 'co_day', 'stay_span', 'hiver', 'ete', 'automne', 'continent_0', 'continent_1', 'continent_2', 'continent_3', 'continent_4', 'year', 'month', 'quarter', 'user_location_region', 'user_location_city', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']
    model = model_rforest
    path = "../Predictions/rforest.csv"

    p.predictor(label,model,path)