# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # загрузка гиперпараметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # загрузка данных
    with open('models/fitted_model.pkl', 'rb') as fd:
        pipeline = joblib.load(fd)
    data = pd.read_csv('data/initial_data.csv')

    # Проверка качества на кросс-валидации
    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        pipeline,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
    )

    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    # сохранение результатов
    os.makedirs('cv_results', exist_ok=True) # создание директории, если её ещё нет
    with open("cv_results/cv_res.json", "w") as json_file:
        json.dump(cv_res, json_file)


if __name__ == '__main__':
    evaluate_model()