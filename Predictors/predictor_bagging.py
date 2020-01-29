from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
import utilitaires as util
import predictor as p
import warnings

warnings.filterwarnings("ignore")

def model_bagging(train,test,label,grid_elt=None):
    
    if grid_elt is None:
        bag = ensemble.BaggingClassifier(KNeighborsClassifier(),n_jobs=1,n_estimators=30,max_samples=1.0, max_features=0.1)
    else:
        bag = ensemble.BaggingClassifier(KNeighborsClassifier(),n_jobs=1,n_estimators=grid_elt[0],max_samples=grid_elt[1], max_features=grid_elt[2])
   
    bag.fit(train[label],train['hotel_cluster'])

    prediction = bag.predict_proba(test[label])
    
    return util.best_proba(prediction),bag

if __name__ == '__main__':
    
    if util.getPCA():
        label = util.getParam()
    else:
        label = ['dest_1', 'dest_2', 'dest_3', 'weekend', 'co_month', 'co_dayofweek', 'ci_quarter', 'co_quarter', 'stay_span', 'duree_prevoyance', 'ci_day', 'ci_dayofweek', 'ci_month', 'co_day', 'stay_span', 'hiver', 'printemps', 'ete', 'continent_0', 'continent_1', 'continent_2', 'continent_3', 'continent_4', 'year', 'month', 'quarter', 'user_location_country', 'user_location_region', 'user_location_city', 'user_id', 'is_mobile', 'is_package', 'channel', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']
        
    model = model_bagging
    path = "../Predictions/bagging.csv"

    p.predictor(label,model,path)