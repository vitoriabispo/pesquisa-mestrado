import time
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from config import DATASET_CONFIGS, N_SPLITS_KFOLD, COLUNA_ALVO, GA_PARAMS, PERCENTUAL_MISSING
from utils.helpers import preparar_dataset, introduzir_missing_mcar, imputar_dados, formatar_tempo
from scripts.feature_selection import executar_selecao_features_ga

def rodar_experimento():
    # contabilização do tempo de execução dos experimetntos
    start_time_total = time.monotonic()
    
    for dataset_name, config in DATASET_CONFIGS.items():
        start_time_dataset = time.monotonic()
        
        output_dir = f'resultados/{dataset_name}'
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"INICIANDO EXPERIMENTO PARA O DATASET: {dataset_name.upper()}")
        print("="*80)
        
        df_limpo = preparar_dataset(config)
        X = df_limpo.drop(columns=[COLUNA_ALVO])
        y = df_limpo[COLUNA_ALVO]

        skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
        
        scores_baseline = []
        scores_imputacao_simples = []
        scores_evo = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            fold_start_time = time.monotonic()
            print(f"\n--- Processando Fold {fold + 1}/{N_SPLITS_KFOLD} ---")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            
            # introduz valores ausentes no conjunto de teste
            X_test_com_missing = introduzir_missing_mcar(X_test_scaled, PERCENTUAL_MISSING)
            
            # --- cenário 1: dados perfeitos e normalizados ---
            modelo_base = GaussianNB().fit(X_train_scaled, y_train)
            y_pred_base = modelo_base.predict(X_test_scaled)
            scores_baseline.append(accuracy_score(y_test, y_pred_base))

            # --- cenário 2: imputação Simples (média) ---
            X_test_imputado_simples = imputar_dados(X_test_com_missing)
            modelo_simples = GaussianNB().fit(X_train_scaled, y_train)
            y_pred_simples = modelo_simples.predict(X_test_imputado_simples)
            scores_imputacao_simples.append(accuracy_score(y_test, y_pred_simples))

            # --- cenário 3: GA+Média ---
            print("   - Executando GA para seleção de features...")
            selected_features = executar_selecao_features_ga(
                X_train_scaled.copy(), y_train.copy(), GA_PARAMS, random_state=fold, percentual_missing=PERCENTUAL_MISSING
            )
            
            modelo_evo = GaussianNB().fit(X_train_scaled[selected_features], y_train)
            X_test_evo_imputado = imputar_dados(X_test_com_missing[selected_features])
            y_pred_evo = modelo_evo.predict(X_test_evo_imputado)
            scores_evo.append(accuracy_score(y_test, y_pred_evo))
            
            print(f"\n   --- Resultados do Fold {fold + 1} ---")
            print(f"   Acurácia Baseline:       {scores_baseline[-1]:.4f}")
            print(f"   Acurácia Imputação Média: {scores_imputacao_simples[-1]:.4f}")
            print(f"   Acurácia Evo:     {scores_evo[-1]:.4f}")
            print(f"   -----------------------------------")
            print(f"   Tempo do Fold: {formatar_tempo(time.monotonic() - fold_start_time)}")

        print("\n\n" + "="*50)
        print(f"ANÁLISE FINAL (K-FOLD) PARA {dataset_name.upper()}")
        print("="*50)
        
        resultados_finais = pd.DataFrame({
            'Cenário': ['Baseline (Dados Perfeitos)', 'Imputação Média (sem FS)', 'Evo (com FS)'],
            'Acurácia Média': [np.mean(scores_baseline), np.mean(scores_imputacao_simples), np.mean(scores_evo)],
            'Desvio Padrão (Std)': [np.std(scores_baseline), np.std(scores_imputacao_simples), np.std(scores_evo)]
        }).round(4)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_filename = f"{dataset_name}_resultados_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"ANÁLISE DOS RESULTADOS - {dataset_name.upper()}\n")
            f.write(f"Executado em: {timestamp}\n")
            f.write(f"Tempo Total para este dataset: {formatar_tempo(time.monotonic() - start_time_dataset)}\n\n")
            f.write(f"Resultados Agregados (Validação Cruzada de {N_SPLITS_KFOLD} folds):\n")
            f.write(resultados_finais.to_string(index=False))

        print(resultados_finais.to_string(index=False))
        print(f"\n[+] Resultados salvos com sucesso em: {output_path}")

    print(f"\n\nEXPERIMENTO COMPLETO FINALIZADO.")
    print(f"Tempo Total de todos os datasets: {formatar_tempo(time.monotonic() - start_time_total)}")

if __name__ == '__main__':
    rodar_experimento()