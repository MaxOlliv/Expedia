import sklearn.semi_supervised as sm
import utilitaires as util 
import predictor as p

def model_labelprop(train,test,label):
    
    lp = sm.LabelPropagation(kernel='rbf')
    lp.fit(train[label],train['hotel_cluster'])
    
    prediction = lp.predict_proba(test[label])
    
    return util.best_proba(prediction), lp

if __name__ == '__main__':
    
    label = util.getParam()
    model = model_labelprop
    path = "../Predictions/labelprop.csv"

    p.predictor(label,model,path)