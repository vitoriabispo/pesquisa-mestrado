import sys
import os
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rodar_experimento 

print("="*40)
print("INICIANDO EXPERIMENTO COM REGRESSÃO LOGÍSTICA")
print("="*40)

model_config = {
    'model': LogisticRegression(random_state=42, max_iter=1000),
    'name': 'LogisticRegression',
    'param_grid': {
        'C': [0.1, 1.0, 10],
        'solver': ['liblinear', 'saga']
    }
}

rodar_experimento(model_config)
print("Experimento com Regressão Logística finalizado.")