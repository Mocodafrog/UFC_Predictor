import pandas as pd
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = os.path.abspath('data')
MODELS_DIR = os.path.abspath('models')
os.makedirs(MODELS_DIR, exist_ok=True)

fight_stats = pd.read_csv(os.path.join(DATA_DIR, 'fight_stats.csv'))
columnas_X = pd.read_csv(os.path.join(DATA_DIR, 'columnas_X.csv'), header=None).squeeze().tolist()

X = fight_stats[columnas_X]
y_winner = fight_stats['Winner']
y_method = fight_stats['Method']

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train_winner, y_test_winner = y_winner.iloc[:split], y_winner.iloc[split:]
y_train_method, y_test_method = y_method.iloc[:split], y_method.iloc[split:]

models = {
    'XGBoost': (XGBClassifier(), {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }),
    'LightGBM': (LGBMClassifier(), {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }),
    'CatBoost': (CatBoostClassifier(verbose=0), {
        'learning_rate': [0.01, 0.1],
        'iterations': [100, 200],
        'depth': [3, 5, 7]
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }),
    'GradientBoosting': (GradientBoostingClassifier(), {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }),
    'SVC': (SVC(probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    })
}

def entrenar(tipo, y_train, y_test):
    mejores = []
    print(f"\n📊 Iniciando entrenamiento para: {tipo.upper()}\n")
    for nombre, (modelo, params) in models.items():
        print(f"⚙️ GridSearch para {nombre}...")
        grid = GridSearchCV(modelo, params, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        joblib.dump(grid.best_estimator_, os.path.join(MODELS_DIR, f"{nombre.lower()}_{tipo}.pkl"))
        mejores.append((nombre, grid.best_estimator_))
        print(f"✅ {nombre} mejores parámetros: {grid.best_params_}")

    stacking = StackingClassifier(estimators=mejores, final_estimator=LogisticRegression(), cv=3, n_jobs=-1)
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, os.path.join(MODELS_DIR, f"stacking_{tipo}.pkl"))

    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    if len(set(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    print(f"\n✅ MÉTRICAS PARA {tipo.upper()}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}\n")

entrenar('winner', y_train_winner, y_test_winner)
entrenar('method', y_train_method, y_test_method)
