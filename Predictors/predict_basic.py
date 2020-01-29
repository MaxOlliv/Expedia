import utilitaires as util
import predictor as p 
import numpy as np

def model_basic(train,test,label):
    
    most_common_clusters =list(train.hotel_cluster.value_counts().head().index)
    prediction =[most_common_clusters for i in range(test.shape[0])]

    pred_2 = np.empty((len(prediction),5),dtype=object)
    for i,liste in enumerate(prediction):
        pred_2[i]=np.asarray(map(int,liste))
        
    return pred_2


if __name__ == '__main__':
    
    label = util.getParam()
    model = model_basic
    path = "../Predictions/basique.csv"

    p.predictor(label,model,path)