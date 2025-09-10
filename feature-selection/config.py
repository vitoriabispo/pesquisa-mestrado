# parametros experimento
COLUNA_ALVO = 'target'     # novo nome da coluna target
N_SPLITS_KFOLD = 10        # splits do k fold
PERCENTUAL_MISSING = 0.10  # % de missing a ser adicionado nas bases

# parametros GA
GA_PARAMS = {
    'ngen': 100,      # número total de gerações 
    'pop_size': 50,   # número total de indivíduos em cada geração
    'cxpb': 0.7,      # probabilidade de dois individuos selecionados realizarem o cruzamento
    'mutpb': 0.04,    # probabilidade de um individuo ser selecionado para sofrer mutação
    'indpb': 0.04,    # probabilidade de mutação de cada gene 
    'tournsize': 5,   # número de indivíduos que participam de um torneio
}

# config datasets UCI 
DATASET_CONFIGS = {
    'statlog_heart': {
        'id': 145,                                                     # ID do UCI
        'original_target_col': 'class',                                # coluna de target da base
        'binarize_lambda': lambda series: (series == 2).astype(int)    # transformando os valores de target em binario 
    }
}