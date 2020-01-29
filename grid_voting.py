import predictor_voting as voting
import utilitaires as util
import best_formula as bf
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    grid = []
    for a in range (5):
        for b in range(5):
            for c in range(5):
                if a==b and b==c and c!=1:
                    continue
                if b==0 and c==0 and a!=1:
                    continue
                if c==0 and a==0 and b!=1:
                    continue
                if c<a or c<b or b<a:
                    continue
                if a==0 and b==0:
                    continue
                if b==0 and c==0:
                    continue
                if c==0 and a==0:
                    continue
                grid.append([a,b,c])
 
    print 'Grille conteant {} elements'.format(len(grid))
    print grid
    
# on en est a 102 avec 010 meilleur
    param = util.getParam()
    model = voting.model_voting
     
    start_time,train_eval,test_eval = bf.init(param) 
     
    best_score = 0 
    best_subset = []
    best_grid_elt = []
    current = 0
     
    for grid_elt in reversed(grid):
        current += 1
        print 'Calcul {}/{}'.format(current,len(grid))
        print 'avec les poids suivants : {}'.format(grid_elt)
        score_temp,subset_temp = bf.best_formula_grid(param, model, start_time, train_eval, test_eval, grid_elt)
        if score_temp>best_score:
            best_score = score_temp
            best_subset = subset_temp
            best_grid_elt = grid_elt
        print '\n#################################################################################################'
        print 'Meilleur score actuel : {}'.format(best_score)
        print 'pour les parametres : {}'.format(best_subset)
        print 'avec les poids suivants : {}'.format(best_grid_elt)
        print '#################################################################################################\n'