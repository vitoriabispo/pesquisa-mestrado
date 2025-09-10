from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# configurações para debugar
MODO_DEBUG = False
FRACAO_AMOSTRA_DEBUG = 1.0
DATASET_DEBUG = 'statlog_heart'

# configurações da fitness
FITNESS_EVALUATOR_MODEL = 'LR' 
FITNESS_CLASSIFIERS = {
    'LR': LogisticRegression(random_state=42, max_iter=1000),
    'NB': GaussianNB()
}

# parâmetros do experimento
COLUNA_ALVO = 'target'     # novo nome da coluna target
N_SPLITS_KFOLD = 10        # splits do k fold
PERCENTUAL_MISSING = 0.20  # % de missing a ser adicionado nas bases

# parametros GA
GA_PARAMS = {
    'ngen': 100,      # número total de gerações 
    'pop_size': 50,   # número total de indivíduos em cada geração
    'cxpb': 0.7,      # probabilidade de dois individuos selecionados realizarem o cruzamento
    'mutpb': 0.04,    # probabilidade de um individuo ser selecionado para sofrer mutação
    'indpb': 0.04,    # probabilidade de mutação de cada gene 
    'tournsize': 5,   # número de indivíduos que participam de um torneio
}

# imputadores
METODOS_DE_IMPUTACAO = [
    SimpleImputer(strategy='mean'),          # 0
    SimpleImputer(strategy='median'),        # 1
    SimpleImputer(strategy='most_frequent'), # 2
    KNNImputer(n_neighbors=3),               # 3
    KNNImputer(n_neighbors=7),               # 4
    IterativeImputer(max_iter=10, random_state=0), # 5
    IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=0), max_iter=5, random_state=0), # 6
    IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), max_iter=5, random_state=0)  # 7
]

# separacao de metodos
INDICES_IMPUTADORES = {
    'numerical': [0, 1, 3, 4, 5, 6, 7],
    'categorical': [2, 3, 4, 5, 6, 7]
}

# config datasets UCI
# ID do UCI | coluna target | transformando o valor de target em binario
DATASET_CONFIGS = {
    'statlog_heart': { 'id': 145, 'binarize_lambda': lambda s: (s == 2).astype(int) },
    'heart_disease': { 'id': 45,  'binarize_lambda': lambda s: (s > 0).astype(int)  },
    'german_credit': { 'id': 144, 'binarize_lambda': lambda s: (s == 2).astype(int) }
}