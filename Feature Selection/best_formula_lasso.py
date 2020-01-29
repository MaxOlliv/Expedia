import predictor_lasso as lasso
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = lasso.model_lasso

if __name__ == '__main__':
    bf.best_formula(param,model)