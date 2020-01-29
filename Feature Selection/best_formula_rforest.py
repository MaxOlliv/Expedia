import predictor_rforest as rf
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = rf.model_rforest

if __name__ == '__main__':
    bf.best_formula(param,model)