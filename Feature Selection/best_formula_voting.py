import predictor_voting as voting
import utilitaires as util
import best_formula as bf
import warnings

warnings.filterwarnings("ignore")

param = util.getParam()
model = voting.model_voting 

if __name__ == '__main__':
    bf.best_formula(param,model)