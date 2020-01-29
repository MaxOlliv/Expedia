import predictor_decision_tree as dt
import utilitaires as util
import best_formula as bf

param = util.getParam()
model = dt.model_dec_tree

if __name__ == '__main__':
    bf.best_formula(param,model)