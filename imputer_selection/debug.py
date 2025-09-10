# debug.py
# Este arquivo serve para rodar uma única iteração do GA para fins de depuração.
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from config import DATASET_CONFIGS, COLUNA_ALVO, GA_PARAMS, PERCENTUAL_MISSING, DATASET_DEBUG
from utils.helpers import preparar_dataset, introduzir_missing_mcar
from scripts.imputer_selection import executar_selecao_imputacao_ga

def rodar_debug_em_amostra():
    print("\n" + "="*80)
    print("MODO DE ANÁLISE/DEBUG ATIVADO")
    print("="*80)
    
    config = DATASET_CONFIGS[DATASET_DEBUG]
    df_limpo = preparar_dataset(config)
    
    X = df_limpo.drop(columns=[COLUNA_ALVO])
    y = df_limpo[COLUNA_ALVO]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_com_missing = introduzir_missing_mcar(X_scaled, PERCENTUAL_MISSING)

    print("\n--- Executando o GA para seleção de método de imputação na amostra ---")
    
    # Use um classificador qualquer para o debug
    classifier_debug = XGBClassifier(random_state=42)
    
    executar_selecao_imputacao_ga(
        X_com_missing, y, GA_PARAMS, classifier_debug, random_state=42
    )

if __name__ == '__main__':
    rodar_debug_em_amostra()