import sys
import os
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rodar_experimento 

print("="*40)
print("INICIANDO EXPERIMENTO COM RANDOM FOREST")
print("="*40)

model_config = {
    'model': RandomForestClassifier(random_state=42),
    'name': 'RandomForest',
    'param_grid': {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
    }
}

rodar_experimento(model_config)
print("Experimento com RandomForest finalizado.")