import time
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score

def preparar_dataset(config):
    print(f"Carregando dataset ID {config['id']} do repositório UCI...")
    dataset = fetch_ucirepo(id=config['id'])
    X = dataset.data.features
    y = dataset.data.targets.copy()
    
    # binarizacao de target
    original_target_col_name = y.columns[0]
    y.rename(columns={original_target_col_name: 'target'}, inplace=True)
    y['target'] = config['binarize_lambda'](y['target'])
    print("   - Alvo binarizado para 0 e 1.")

    # converte para bool todas as features
    X_tratado = pd.get_dummies(X, drop_first=True, dummy_na=False) # dummy_na=False para não criar coluna para NaNs
    for col in X_tratado.columns:
        if X_tratado[col].dtype == 'bool':
            X_tratado[col] = X_tratado[col].astype(int)
    print("   - Variáveis categóricas convertidas para formato numérico.")

    # junta features e alvo e remove linhas com valores ausentes
    df_completo = pd.concat([X_tratado, y], axis=1)
    df_limpo = df_completo.dropna().reset_index(drop=True)
    print(f"   - Dataset limpo pronto. Tamanho: {len(df_limpo)} linhas.")
    
    return df_limpo

# introduz percentual_missing% de missing no df (totalmente aleatorio)
def introduzir_missing_mcar(df, percentual_missing):
    df_com_missing = df.copy()
    colunas_features = df_com_missing.columns.drop('target', errors='ignore')

    for col in colunas_features:
        indices_nao_nulos = df_com_missing[col].dropna().index
        num_valores_a_remover = int(np.floor(percentual_missing * len(indices_nao_nulos)))
        if num_valores_a_remover > 0:
            indices_para_remover = np.random.choice(indices_nao_nulos, num_valores_a_remover, replace=False)
            df_com_missing.loc[indices_para_remover, col] = np.nan
    return df_com_missing

# imputa os dados com a média
def imputar_dados(X):
    X_imputado = X.copy()
    colunas_para_imputar = X_imputado.columns[X_imputado.isnull().any()].tolist()
    if colunas_para_imputar:
        imputer_mean = SimpleImputer(strategy='mean')
        X_imputado[colunas_para_imputar] = imputer_mean.fit_transform(X_imputado[colunas_para_imputar])
    return X_imputado

# gmean p classificação
def multiclass_gmean(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return np.prod(recalls) ** (1.0 / len(recalls)) if len(recalls) > 0 else 0.0

# formatação do tempo p imprimir
def formatar_tempo(segundos):
    if segundos < 60: return f"{segundos:.2f} segundos"
    minutos, segs_restantes = divmod(segundos, 60)
    if minutos < 60: return f"{int(minutos)} minuto(s) e {segs_restantes:.2f} segundo(s)"
    horas, mins_restantes = divmod(minutos, 60)
    return f"{int(horas)} hora(s), {int(mins_restantes)} minuto(s) e {segs_restantes:.2f} segundo(s)"