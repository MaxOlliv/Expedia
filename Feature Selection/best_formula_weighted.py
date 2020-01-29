import predictor_weighted as weighted
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = weighted.model_weighted

if __name__ == '__main__':
    bf.best_formula(param,model)