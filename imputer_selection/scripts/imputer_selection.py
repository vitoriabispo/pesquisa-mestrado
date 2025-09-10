import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from config import METODOS_DE_IMPUTACAO, INDICES_IMPUTADORES

# config do deap
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# aplica a receita (lista de métodos selecionados)
def aplicar_receita_imputacao(X, features_incompletas, receita):
    X_imputado = X.copy()
    for i, feature_name in enumerate(features_incompletas):
        if feature_name not in X_imputado.columns:
            continue
        
        # seleciona o método do gene
        metodo_idx = receita[i]
        imputer = METODOS_DE_IMPUTACAO[metodo_idx]
        dados_coluna = X_imputado[[feature_name]].values
        
        # verifica se realmente tem valores ausentes para aplicar
        if np.isnan(dados_coluna).sum() > 0:
            try:
                imputer_clone = clone(imputer)
                X_imputado[feature_name] = imputer_clone.fit_transform(dados_coluna)
            except ValueError:
                X_imputado[feature_name].fillna(0, inplace=True)
    
    # verificando se ainda tem missing
    if X_imputado.isnull().values.any():
        X_imputado.fillna(0, inplace=True)
    return X_imputado

# função de fitness do ga
def avaliar_fitness_selecao_imputacao(individual, X_data, y_data, classifier, random_state):
    features_incompletas = X_data.columns[X_data.isna().any()].tolist()
    
    # aplica a receita de imputação (imputa com os métodos selecionados)
    X_imputed_df = aplicar_receita_imputacao(X_data, features_incompletas, individual)
    X_imputed_values = X_imputed_df.values
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, val_idx in cv.split(X_imputed_values, y_data):
        X_train_fold, X_val_fold = X_imputed_values[train_idx], X_imputed_values[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        
        # treina e avalia o classificador
        clf = clone(classifier)
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, y_pred))
            
    return (np.mean(scores),)

# exibição dos métodos selecionados pelo GA
def traduzir_e_exibir_receita(features_incompletas, melhor_receita):
    print("\nMelhor Receita de Imputação Encontrada")
    nomes_metodos = [type(m).__name__ + str(m.get_params()) for m in METODOS_DE_IMPUTACAO]
    df_receita = pd.DataFrame({
        'Feature Incompleta': features_incompletas,
        'Método Escolhido': [nomes_metodos[i] for i in melhor_receita]
    })
    print(df_receita.to_string(index=False))

# execucao completa do GA
def executar_selecao_imputacao_ga(X_data, y_data, ga_params, classifier, numerical_cols_original, categorical_cols_original, random_state):
    print("Executando GA")
    features_incompletas = X_data.columns[X_data.isna().any()].tolist()
    if not features_incompletas:
        print("Nenhuma feature com dados ausentes encontrada")
        return None, []

    toolbox = base.Toolbox()
    
    def criar_individuo():
        genes = []
        for feature in features_incompletas:
            # verifica se a feature atual pertence a uma categórica original
            is_categorical = any(feature.startswith(cat_col) for cat_col in categorical_cols_original)
            
            # sorteia um método de imputação da lista seguindo o tipo da feature
            if is_categorical:
                genes.append(random.choice(INDICES_IMPUTADORES['categorical']))
            else:
                genes.append(random.choice(INDICES_IMPUTADORES['numerical']))
        return creator.Individual(genes)

    toolbox.register("individual", criar_individuo)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", avaliar_fitness_selecao_imputacao, X_data=X_data, y_data=y_data, classifier=classifier, random_state=random_state)
    toolbox.register("mate", tools.cxTwoPoint)

    # mutação customizada para respeitar os tipos das features 
    def mutCustom(individual, indpb):
        for i, feature in enumerate(features_incompletas):
            if random.random() < indpb:
                is_categorical = any(feature.startswith(cat_col) for cat_col in categorical_cols_original)
                if is_categorical:
                    individual[i] = random.choice(INDICES_IMPUTADORES['categorical'])
                else:
                    individual[i] = random.choice(INDICES_IMPUTADORES['numerical'])
        return individual,

    toolbox.register("mutate", mutCustom, indpb=ga_params['indpb'])
    toolbox.register("select", tools.selTournament, tournsize=ga_params['tournsize'])
    
    # executando GA
    pop = toolbox.population(n=ga_params['pop_size'])
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    print("Iniciando evolução do GA com elitismo e early stopping")
    patience = 20
    best_fitness_so_far = 0.0
    generations_without_improvement = 0
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ga_params['ngen']):
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(pop), **record)
        
        current_max_fitness = hof[0].fitness.values[0]
        if current_max_fitness > best_fitness_so_far:
            best_fitness_so_far = current_max_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        if generations_without_improvement >= patience:
            break

        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, ga_params['cxpb'], ga_params['mutpb'])
        pop[:] = offspring

    print("\nLog da evolução final")
    print(logbook)
    
    melhor_receita = hof[0]
    # traduzir_e_exibir_receita(features_incompletas, melhor_receita)
    return melhor_receita, features_incompletas