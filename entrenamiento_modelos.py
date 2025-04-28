# Definir las características (X) y la variable objetivo (y)
X = fight_stats.drop(columns=['Winner', 'Method', 'Event', 'Fight', 'Fighter'])
y_winner = fight_stats['Winner']  
y_method = fight_stats['Method']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train_winner, y_test_winner = train_test_split(X, y_winner, test_size=0.2, random_state=42)
_, _, y_train_method, y_test_method = train_test_split(X, y_method, test_size=0.2, random_state=42)

# Definir los modelos y sus parámetros para GridSearch
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
        'depth': [3, 5, 7],
        'iterations': [100, 200]
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }),
    'GradientBoosting': (GradientBoostingClassifier(), {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }),
    'MLPClassifier': (MLPClassifier(), {
        'hidden_layer_sizes': [(100,), (150,)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [200, 300],
        'activation': ['relu', 'tanh']
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }),
    'KNeighbors': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    'SVC': (SVC(probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    'ExtraTrees': (ExtraTreesClassifier(), {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    })
}

# Función para realizar GridSearch y entrenar
def grid_search_and_evaluate(models, X_train, X_test, y_train, y_test, is_method=False):
    results = {}
    for name, (model, params) in models.items():
        print(f"Realizando GridSearch para {name}...")
        grid_search = GridSearchCV(model, param_grid=params, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Predecir las probabilidades
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Convertir a porcentajes
        percentage_predictions = pd.DataFrame(y_pred_proba, columns=[f'Prob_{i}' for i in best_model.classes_])

        # Predicción final basada en las probabilidades
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        if len(best_model.classes_) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        print(f"Mejores hiperparámetros para {name}: {grid_search.best_params_}")
        print(f"Accuracy para {name}: {accuracy:.4f}")
        print(f"ROC-AUC para {name}: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Guardar los resultados
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'percentage_predictions': percentage_predictions  # Aquí están los porcentajes
        }
    return results

# Entrenar y evaluar para predicción de ganador
print("\n--- Predicción del Ganador ---")
results_winner = grid_search_and_evaluate(models, X_train, X_test, y_train_winner, y_test_winner)

# Entrenar y evaluar para predicción del método de la pelea
print("\n--- Predicción del Método de la Pelea ---")
results_method = grid_search_and_evaluate(models, X_train, X_test, y_train_method, y_test_method, is_method=True)

# Mostrar resultados finales
print("\nResultados del modelo para predicción del ganador:")
for model, result in results_winner.items():
    print(f"{model}: Accuracy = {result['accuracy']:.4f}, ROC-AUC = {result['roc_auc']:.4f}, Mejores parámetros: {result['best_params']}")
    print(f"Porcentajes de predicción para ganador:\\n{result['percentage_predictions'].head()}")

print("\nResultados del modelo para predicción del método:")
for model, result in results_method.items():
    print(f"{model}: Accuracy = {result['accuracy']:.4f}, ROC-AUC = {result['roc_auc']:.4f}, Mejores parámetros: {result['best_params']}")
    print(f"Porcentajes de predicción para método:\\n{result['percentage_predictions'].head()}")

# **Modelos y parámetros para predicción del GANADOR**
base_models_winner = [
    ('XGBoost', XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)),
    ('LightGBM', LGBMClassifier(learning_rate=0.1, max_depth=3, n_estimators=200)),
    ('CatBoost', CatBoostClassifier(depth=5, iterations=200, learning_rate=0.1)),
    ('RandomForest', RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=100)),
    ('GradientBoosting', GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)),
    ('LogisticRegression', LogisticRegression(C=10, solver='liblinear',max_iter=500 )),  # C=10 confirmado
    ('ExtraTrees', ExtraTreesClassifier(max_depth=20, min_samples_split=2, n_estimators=200)),
    ('SVC', SVC(C=1, kernel='linear', probability=True))
]

# Metaclassifier para predicción del ganador
final_model_winner = LogisticRegression()

# Stacking ensemble para GANADOR
stacking_winner = StackingClassifier(estimators=base_models_winner, final_estimator=final_model_winner, cv=3)

# Entrenar el modelo para predicción del ganador
stacking_winner.fit(X_train, y_train_winner)

# Predicción y evaluación para GANADOR
y_pred_winner = stacking_winner.predict(X_test)
y_pred_proba_winner = stacking_winner.predict_proba(X_test)
accuracy_winner = accuracy_score(y_test_winner, y_pred_winner)

# Para el ROC-AUC, tomamos la segunda columna de probabilidades (la clase \"1\")
roc_auc_winner = roc_auc_score(y_test_winner, y_pred_proba_winner[:, 1])

print("\n--- Resultados del Stacking Ensemble (Ganador) ---")
print(f"Accuracy predicción ganador: {accuracy_winner:.4f}")
print(f"ROC-AUC predicción ganador: {roc_auc_winner:.4f}")
print(classification_report(y_test_winner, y_pred_winner))


# **Modelos y parámetros para predicción del MÉTODO**
base_models_method = [
    ('XGBoost', XGBClassifier(learning_rate=0.1, max_depth=7, n_estimators=100)),
    ('LightGBM', LGBMClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)),
    ('CatBoost', CatBoostClassifier(depth=3, iterations=200, learning_rate=0.1)),
    ('RandomForest', RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=200)),
    ('GradientBoosting', GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)),
    ('LogisticRegression', LogisticRegression(C=10, solver='liblinear',max_iter=500 )),
    ('SVC', SVC(C=10, kernel='linear', probability=True)),
    ('ExtraTrees', ExtraTreesClassifier(max_depth=20, min_samples_split=5, n_estimators=200))
]

# Metaclassifier para predicción del método
final_model_method = LogisticRegression()

# Stacking ensemble para MÉTODO
stacking_method = StackingClassifier(estimators=base_models_method, final_estimator=final_model_method, cv=3)

# Entrenar el modelo para predicción del método
stacking_method.fit(X_train, y_train_method)

# Predicción y evaluación para MÉTODO
y_pred_method = stacking_method.predict(X_test)
y_pred_proba_method = stacking_method.predict_proba(X_test)
accuracy_method = accuracy_score(y_test_method, y_pred_method)

# Para el ROC-AUC, tomamos las probabilidades de todas las clases (multiclase)
roc_auc_method = roc_auc_score(y_test_method, y_pred_proba_method, multi_class='ovr')

print("\n--- Resultados del Stacking Ensemble (Método) ---")
print(f"Accuracy predicción método: {accuracy_method:.4f}")
print(f"ROC-AUC predicción método: {roc_auc_method:.4f}")
print(classification_report(y_test_method, y_pred_method))

joblib.dump(stacking_winner,'stacking_winner.pkl')
joblib.dump(stacking_method,'stacking_method.pkl' )