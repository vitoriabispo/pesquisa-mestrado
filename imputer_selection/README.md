## Instalar bibliotecas necessárias
```pip install -r requirements.txt```

## Como executar

O projeto é estruturado com scripts "executáveis" (run_*.py), onde cada um roda o pipeline completo para um classificador final específico.

### Para executar com o classificador final Gaussian Naive Bayes:
```python ./classificadores/run_gaussiannb.py```

### Para executar com o classificador final Logistic Regression:
```python ./classificadores/run_logistic.py```

### Para executar com o classificador final Random Forest:
```python ./classificadores/run_rf.py```

### Para executar com o classificador final SVC (Support Vector Classifier):
```python ./classificadores/run_svc.py``` 

### Para executar com o classificador final XGBoost:
```python ./classificadores/run_xgb.py``` 

>> O arquivo main.py é um script de exemplo que por padrão executa o experimento com o GaussianNB, funcionando de forma similar ao run_gaussiannb.py.