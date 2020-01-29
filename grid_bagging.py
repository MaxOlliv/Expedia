import numpy as np
import predictor_bagging as bagging
import utilitaires as util
import best_formula as bf
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    grid = []
    for a in range (10,100,10):
        for b in np.linspace(0.1,1,6):
            for c in np.linspace(0.1,1,6):
                grid.append([a,b,c])
                
    print 'Grille conteant {} elements'.format(len(grid))
    
    param = util.getParam()
    model = bagging.model_bagging
    
    start_time,train_eval,test_eval = bf.init(param)
    
    best_score = 0
    best_subset = []
    best_grid_elt = []
    current = 0
    
    for grid_elt in grid:
        current += 1
        print 'Calcul {}/{}'.format(current,len(grid))
        print 'avec n_estimators={}, max_samples={} et max_features={}'.format(grid_elt[0],grid_elt[1],grid_elt[2])
        score_temp,subset_temp = bf.best_formula_grid(param, model, start_time, train_eval, test_eval, grid_elt)
        if score_temp>best_score:
            best_score = score_temp
            best_subset = subset_temp
            best_grid_elt = grid_elt
        print '\n#################################################################################################'
        print 'Meilleur score actuel : {}'.format(best_score)
        print 'pour les parametres : {}'.format(best_subset)
        print 'avec n_estimators={}, max_samples={} et max_features={}'.format(best_grid_elt[0],best_grid_elt[1],best_grid_elt[2])
        print '#################################################################################################\n'