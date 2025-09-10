import sys
import os
from sklearn.naive_bayes import GaussianNB

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import rodar_experimento 

print("="*40)
print("INICIANDO EXPERIMENTO COM GAUSSIAN NAIVE BAYES")
print("="*40)

model_config = {
    'model': GaussianNB(),
    'name': 'GaussianNB',
    'param_grid': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

rodar_experimento(model_config)

print("\nExperimento com Gaussian Naive Bayes finalizado.")