import sys
import os
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rodar_experimento 

print("="*40)
print("INICIANDO EXPERIMENTO COM XGBOOST")
print("="*40)

model_config = {
    'model': XGBClassifier(random_state=42, eval_metric='logloss'),
    'name': 'XGBoost',
    'param_grid': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.05],
    }
}

rodar_experimento(model_config)
print("Experimento com XGBoost finalizado.")