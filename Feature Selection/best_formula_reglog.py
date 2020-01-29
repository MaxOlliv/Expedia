import predictor_reglog as rlg
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = rlg.model_reglog 

if __name__ == '__main__':
    bf.best_formula(param,model)