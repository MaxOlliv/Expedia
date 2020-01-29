import predictor_gradboost as gb
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = gb.model_gradboost

if __name__ == '__main__':
    bf.best_formula(param,model)