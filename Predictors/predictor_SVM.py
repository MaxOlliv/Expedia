from sklearn import svm
import utilitaires as util
import predictor as p 

def model_SVM(train,test,label):
    SVM = svm.SVC(kernel='rbf',probability=True)
    
    SVM.fit(train[label],train['hotel_cluster'])

    prediction = SVM.predict_proba(test[label])
    
    return util.best_proba(prediction) , SVM

if __name__ == '__main__':
    
    label = util.getParam()
    model = model_SVM
    path = "../Predictions/SVM.csv"

    p.predictor(label,model,path)