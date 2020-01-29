from sklearn import linear_model
import utilitaires as util
import predictor as p


def model_reglog(train,test,label):

    reglog = linear_model.LogisticRegression()
    reglog.fit(train[label],train['hotel_cluster'])

    prediction = reglog.predict_proba(test[label])
    
    return util.best_proba(prediction),reglog

if __name__ == '__main__':
    
    if util.getPCA():
        label = util.getParam()
    else:
        label = ['continent_0','continent_2','continent_3','continent_4','month','quarter','user_location_country','user_location_region','is_package','channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']
        
    model = model_reglog
    path = "../Predictions/regression_logistique.csv"

    p.predictor(label,model,path)