import random
import numpy as np
from deap import base, creator, tools, algorithms
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB

# importando os módulos
from utils.helpers import imputar_dados, multiclass_gmean, introduzir_missing_mcar

# configurando o deap
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def avaliar_fitness_ga(individual, X_data, y_data, random_state):
    # lista de feats completas e incompletas da base
    features_completas = X_data.columns[X_data.notna().all()].tolist()
    features_incompletas = X_data.columns[X_data.isna().any()].tolist()
    
    # seleciona as features incompletas com base na lista de genes do individuo
    features_incompletas_selecionadas = [feat for i, feat in enumerate(features_incompletas) if individual[i] == 1]
    
    if not features_incompletas_selecionadas: return 0.,

    
    features_finais = features_completas + features_incompletas_selecionadas
    X_subset = X_data[features_finais].copy() # df de treino
    X_imputed = imputar_dados(X_subset).values # imput no df de treino

    # k fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    gms = []
    for train_idx, val_idx in cv.split(X_imputed, y_data):
        X_train_fold, X_val_fold = X_imputed[train_idx], X_imputed[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        
        # smote para balancear
        try:
            smote = SMOTE(random_state=random_state)
            X_train_fold_bal, y_train_fold_bal = smote.fit_resample(X_train_fold, y_train_fold)
        except ValueError:
            X_train_fold_bal, y_train_fold_bal = X_train_fold, y_train_fold
        
        # treino e avaliação
        clf = GaussianNB()
        clf.fit(X_train_fold_bal, y_train_fold_bal)
        y_pred = clf.predict(X_val_fold)
        gms.append(multiclass_gmean(y_val_fold, y_pred))
        
    avg_gm = np.mean(gms)
    
    # oenalidade para incentivar solucoes eficientes
    penalty_factor = 0.01 
    num_selecionadas = len(features_incompletas_selecionadas) 
    num_total_incompletas = len(features_incompletas)
    
    # quanto + feat, + ppenalidade
    if num_total_incompletas > 0:
        penalty = 1.0 - (penalty_factor * (num_selecionadas / num_total_incompletas))
    else:
        penalty = 1.0
    # fitness com desconto da penalidade    
    return (avg_gm * penalty,)

def executar_selecao_features_ga(X_data, y_data, ga_params, random_state, percentual_missing):
    # se n tem missing, eu uso todas e não faço seleção de feat
    if not percentual_missing:
        return X_data.columns.tolist()
    
    # adc missing
    X_data_com_missing = introduzir_missing_mcar(X_data, percentual_missing)
    features_incompletas = X_data_com_missing.columns[X_data_com_missing.isna().any()].tolist()

    # classe toolbox
    toolbox = base.Toolbox()

    # gerador de atributo
    toolbox.register("attr_bool", random.randint, 0, 1) # gene do indv sorteia um inteiro entre 0 e 1 
    # gerador de cromossomo
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(features_incompletas))
    # gerador de populacao
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # função de fitness/avaliacao
    toolbox.register("evaluate", avaliar_fitness_ga, X_data=X_data_com_missing, y_data=y_data, random_state=random_state)
    
    # funções genéticas
    toolbox.register("mate", tools.cxTwoPoint) # cruzamento
    toolbox.register("mutate", tools.mutFlipBit, indpb=ga_params['indpb']) # mutação
    toolbox.register("select", tools.selTournament, tournsize=ga_params['tournsize']) # seleção
    
    pop = toolbox.population(n=ga_params['pop_size'])
    hof = tools.HallOfFame(1) # armazena o melhor individuo
    
    # executa o algoritmo
    algorithms.eaSimple(pop, toolbox, cxpb=ga_params['cxpb'], mutpb=ga_params['mutpb'],
                        ngen=ga_params['ngen'], halloffame=hof, verbose=False)
    
    best_individual = hof[0]
    
    # lista final de features selecionadas
    features_completas = X_data.columns[X_data.notna().all()].tolist()
    selected_incomplete_features = [feat for i, feat in enumerate(features_incompletas) if best_individual[i] == 1]
    
    return features_completas + selected_incomplete_features