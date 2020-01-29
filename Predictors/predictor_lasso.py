from sklearn import linear_model
import utilitaires as util
import predictor as p


def model_lasso(train,test,label):
    C = 0.01
    lasso = linear_model.LogisticRegression(C = C, penalty="l2")
    lasso.fit(train[label],train['hotel_cluster'])

    prediction = lasso.predict_proba(test[label])
    
    return util.best_proba(prediction),lasso

if __name__ == '__main__':
    

    label = util.getParam()
    model = model_lasso
    path = "../Predictions/regression_lasso.csv"

    p.predictor(label,model,path)