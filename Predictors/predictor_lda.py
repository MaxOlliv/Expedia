from sklearn import discriminant_analysis
import utilitaires as util
import predictor as p 

def model_lda(train,test,label):
    
    reglin = discriminant_analysis.LinearDiscriminantAnalysis()
    reglin.fit(train[label],train['hotel_cluster'])

    prediction = reglin.predict_proba(test[label])
    
    return util.best_proba(prediction), reglin
    
if __name__ == '__main__':
    
    label = util.getParam()
    model = model_lda
    path = "../Predictions/lda.csv"

    p.predictor(label,model,path)
