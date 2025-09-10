import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score

def preparar_dataset(config, usar_amostra=False, fracao_amostra=1.0):
    print(f"Carregando dataset ID {config['id']} do repositório UCI")
    dataset = fetch_ucirepo(id=config['id'])
    X_original = dataset.data.features
    y = dataset.data.targets.copy() # recebe o target do repositorio do UCI 
    
    # binarizacao do target
    original_target_col_name = y.columns[0]
    y.rename(columns={original_target_col_name: 'target'}, inplace=True)
    y['target'] = config['binarize_lambda'](y['target'])
    # print("   - Alvo binarizado para 0 e 1.")

    df_completo = pd.concat([X_original, y], axis=1)
    df_completo.dropna(subset=['target'], inplace=True)

    df_para_processar = df_completo
    if usar_amostra and 0 < fracao_amostra < 1:
        print(f"   - MODO DE AMOSTRAGEM ATIVADO. Extraindo {fracao_amostra*100:.0f}% dos dados.")
        df_para_processar = df_completo.groupby('target', group_keys=False).apply(
            lambda x: x.sample(frac=fracao_amostra, random_state=42)
        )

    X_final = df_para_processar.drop(columns=['target'])
    y_final = df_para_processar['target']
    
    # identificacao do tipo das features
    numerical_features_original = X_final.select_dtypes(include=np.number).columns.tolist()
    categorical_features_original = X_final.select_dtypes(exclude=np.number).columns.tolist()

    #print(f"Features numéricas: {numerical_features_original}")
    #print(f"Features categóricas: {categorical_features_original}")
    
    # Garante que só tentará fazer o get_dummies se houver colunas categóricas
    if categorical_features_original:
        X_tratado = pd.get_dummies(X_final, columns=categorical_features_original, drop_first=True)
    else:
        X_tratado = X_final.copy()

    for col in X_tratado.columns:
        if X_tratado[col].dtype == 'bool':
            X_tratado[col] = X_tratado[col].astype(int)
            
    df_final = pd.concat([X_tratado, y_final], axis=1).dropna().reset_index(drop=True)
    # print(f"Dataset limpo pronto. Tamanho: {len(df_final)} linhas.")
    
    return df_final, numerical_features_original, categorical_features_original

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

# imputa os dados com a média (usada no CENÁRIO 2)
def imputar_dados(X):
    X_imputado = X.copy()
    colunas_para_imputar = X_imputado.columns[X_imputado.isnull().any()].tolist()
    if colunas_para_imputar:
        imputer_mean = SimpleImputer(strategy='mean')
        X_imputado[colunas_para_imputar] = imputer_mean.fit_transform(X_imputado[colunas_para_imputar])
    return X_imputado

# formatação do tempo p imprimir
def formatar_tempo(segundos):
    if segundos < 60: return f"{segundos:.2f} segundos"
    minutos, segs_restantes = divmod(segundos, 60)
    if minutos < 60: return f"{int(minutos)} minuto(s) e {segs_restantes:.2f} segundo(s)"
    horas, mins_restantes = divmod(minutos, 60)
    return f"{int(horas)} hora(s), {int(mins_restantes)} minuto(s) e {segs_restantes:.2f} segundo(s)"