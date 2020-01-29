import sklearn.ensemble as ske
import utilitaires as util 
import predictor as p
from keras.models import Sequential
from keras.layers import Dense, Activation



def model_neural(train,test,label):
    model = Sequential()
    model.add(Dense(12, input_dim=len(label), init='uniform', activation='relu'))
    model.add(Dense(len(label), init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train[label], train['hotel_cluster'], nb_epoch=150, batch_size=10)
    

    prediction = model.predict_proba(test[label])
    
    return util.best_proba(prediction)

if __name__ == '__main__':
    
    label = util.getParam()
    model = model_neural
    path = "../Predictions/neural.csv"

    p.predictor(label,model,path)

