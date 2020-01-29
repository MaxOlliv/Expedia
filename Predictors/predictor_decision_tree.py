from sklearn.tree import DecisionTreeClassifier
import utilitaires as util
import predictor as p

def model_dec_tree(train,test,label):
    
    dectree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, max_features=None)
    dectree.fit(train[label], train["hotel_cluster"]) 

    
    prediction = dectree.predict_proba(test[label])
    
    return util.best_proba(prediction), dectree

if __name__ == '__main__':
    
    label = util.getParam()
    model = model_dec_tree
    path = "../Predictions/dec_tree.csv"

    p.predictor(label,model,path)