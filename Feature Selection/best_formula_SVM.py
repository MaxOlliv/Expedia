import predictor_SVM as svm
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = svm.model_SVM

if __name__ == '__main__':
    bf.best_formula(param,model)