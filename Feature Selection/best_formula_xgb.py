import predictor_xgb as xgb
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = xgb.model_xgb

if __name__ == '__main__':
    bf.best_formula(param,model)