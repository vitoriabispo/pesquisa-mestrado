import sys
import os
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rodar_experimento 

print("="*40)
print("INICIANDO EXPERIMENTO COM SVC")
print("="*40)

model_config = {
    'model': SVC(random_state=42, probability=True),
    'name': 'SVC',
    'param_grid': {
        'C': [0.1, 1.0, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }
}

rodar_experimento(model_config)
print("Experimento com SVC finalizado.")