from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import utilitaires as util
import predictor as p
import warnings

warnings.filterwarnings("ignore")

def model_voting(train,test,label,grid_elt=None):
    
    clf1 = LogisticRegression(random_state=1)
    clf2 = GradientBoostingClassifier(n_estimators=10)
    clf3 = KNeighborsClassifier(n_neighbors=30)
    
    if grid_elt is None:
        eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('knn', clf3)], voting='soft',weights=[1,1,1])
    else:
        eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gb', clf2), ('knn', clf3)], voting='soft',weights=grid_elt)
   
    eclf1.fit(train[label],train['hotel_cluster'])

    prediction = eclf1.predict_proba(test[label])
    
    return util.best_proba(prediction),eclf1

if __name__ == '__main__':
    
    if util.getPCA():
        label = util.getParam()
    else:
        label = util.getParam()
        
    model = model_voting
    path = "../Predictions/voting.csv"
    
    p.predictor(label,model,path)