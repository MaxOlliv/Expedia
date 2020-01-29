import predictor_knn as knn
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = knn.model_knn

if __name__ == '__main__':
    bf.best_formula(param,model)
#     bf.best_formula_loop(param,model,range(0,3000001,10000)) 
    