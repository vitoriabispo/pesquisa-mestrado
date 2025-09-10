import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB

from config import DATASET_CONFIGS, COLUNA_ALVO, N_SPLITS_KFOLD, GA_PARAMS, PERCENTUAL_MISSING, FITNESS_EVALUATOR_MODEL, FITNESS_CLASSIFIERS
from utils.helpers import preparar_dataset, introduzir_missing_mcar, imputar_dados, formatar_tempo
from scripts.imputer_selection import executar_selecao_imputacao_ga, aplicar_receita_imputacao

# modelo para experimentos (só executa esse se rodar pelo main.py)
FINAL_MODEL_CONFIG = {
    'model': GaussianNB(),
    'name': 'GAUSSIANNB',
    'param_grid': {}
}

def rodar_experimento(model_config):
    # parametros do modelo
    classifier = model_config['model']
    model_name = model_config['name']
    param_grid = model_config['param_grid']
    
    # contabilização do tempo de execução dos experimetntos
    start_time_total = time.monotonic()
    
    print("\n" + "="*80)
    print(f"INICIANDO EXPERIMENTO PARA O CLASSIFICADOR FINAL: {model_name.upper()}")
    print(f"AVALIADOR DE FITNESS DO GA: {FITNESS_EVALUATOR_MODEL}")
    print("="*80)

    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n-> Processando dataset: {dataset_name.upper()}")
        
        df_limpo, numerical_cols, categorical_cols = preparar_dataset(config)
        
        X = df_limpo.drop(columns=[COLUNA_ALVO])
        y = df_limpo[COLUNA_ALVO]

        skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
        
        scores_baseline, scores_imputacao_simples, scores_evo = [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nProcessando Fold {fold + 1}/{N_SPLITS_KFOLD}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
            
             # introduz valores ausentes nos conjuntos
            X_train_com_missing = introduzir_missing_mcar(X_train_scaled, PERCENTUAL_MISSING)
            X_test_com_missing = introduzir_missing_mcar(X_test_scaled, PERCENTUAL_MISSING)
            
            def treinar_modelo_final(X_train_data, y_train_data):
                # inicia com o modelo base
                model = clone(classifier)
                model.fit(X_train_data, y_train_data)

                # otimiza com GridSearchCV
                if param_grid:
                    grid_search = GridSearchCV(
                        estimator=clone(classifier),
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=3,
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_data, y_train_data)
                    return grid_search.best_estimator_
                
                # retorna o modelo base já treinado
                return model

            # --- cenário 1: dados perfeitos e normalizados ---
            print("[CENÁRIO 1] Baseline")
            modelo_base = treinar_modelo_final(X_train_scaled, y_train)
            scores_baseline.append(accuracy_score(y_test, modelo_base.predict(X_test_scaled)))

            # --- cenário 2: imputação Simples (média) ---
            print("[CENÁRIO 2] Imputação Média")
            X_train_imputado_simples = imputar_dados(X_train_com_missing)
            X_test_imputado_simples = imputar_dados(X_test_com_missing)
            modelo_simples = treinar_modelo_final(X_train_imputado_simples, y_train)
            scores_imputacao_simples.append(accuracy_score(y_test, modelo_simples.predict(X_test_imputado_simples)))

            # --- cenário 3: GA para seleção de imputador
            print("[CENÁRIO 3] - GA")
            fitness_classifier = FITNESS_CLASSIFIERS[FITNESS_EVALUATOR_MODEL]
            melhor_receita, features_incompletas = executar_selecao_imputacao_ga(
                X_train_com_missing, y_train.copy(), GA_PARAMS, fitness_classifier,
                numerical_cols, categorical_cols, random_state=fold
            )
            
            if melhor_receita:
                print("Aplicando a melhor receita aos conjuntos de treino e teste")
                X_train_evo_imputado = aplicar_receita_imputacao(X_train_com_missing, features_incompletas, melhor_receita)
                X_test_evo_imputado = aplicar_receita_imputacao(X_test_com_missing, features_incompletas, melhor_receita)
                modelo_evo = treinar_modelo_final(X_train_evo_imputado, y_train)
                scores_evo.append(accuracy_score(y_test, modelo_evo.predict(X_test_evo_imputado)))
            else: 
                scores_evo.append(np.mean(scores_imputacao_simples))

        print("\n" + "=" * 50)
        print(f"ANÁLISE FINAL ({model_name.upper()}) PARA {dataset_name.upper()}")
        print("=" * 50)

        resultados_finais = pd.DataFrame({
            'Cenário': ['Baseline', 'Imputação Média', 'GA'],
            'Acurácia Média': [np.mean(scores_baseline), np.mean(scores_imputacao_simples), np.mean(scores_evo)],
            'Desvio Padrão (Std)': [np.std(scores_baseline), np.std(scores_imputacao_simples), np.std(scores_evo)]
        }).round(4)
        print(resultados_finais.to_string(index=False))

    print(f"\n\nEXPERIMENTO COM {model_name.upper()} FINALIZADO.")
    print(f"Tempo Total: {formatar_tempo(time.monotonic() - start_time_total)}")

if __name__ == '__main__':
    rodar_experimento(FINAL_MODEL_CONFIG)