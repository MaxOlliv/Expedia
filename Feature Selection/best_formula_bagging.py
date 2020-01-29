import predictor_bagging as bagging
import utilitaires as util
import best_formula as bf
import warnings

warnings.filterwarnings("ignore")

param = util.getParam()
model = bagging.model_bagging

if __name__ == '__main__':
    bf.best_formula(param,model)