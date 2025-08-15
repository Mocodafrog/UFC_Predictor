import streamlit as st
import joblib
import pandas as pd

from ufc_predictor.config import MODEL_VERSION


@st.cache_resource(show_spinner="Cargando modelo del ganador...")
def load_stacking_winner(version: str):
    try:
        return joblib.load(f"models/{version}/stacking_winner.pkl")
    except FileNotFoundError:
        st.error(f"No se encontró el modelo: models/{version}/stacking_winner.pkl")
        return None


@st.cache_resource(show_spinner="Cargando modelo del método...")
def load_stacking_method(version: str):
    try:
        return joblib.load(f"models/{version}/stacking_method.pkl")
    except FileNotFoundError:
        st.error(f"No se encontró el modelo: models/{version}/stacking_method.pkl")
        return None


# Cargar los modelos entrenados
stacking_winner = load_stacking_winner(MODEL_VERSION)
stacking_method = load_stacking_method(MODEL_VERSION)
if stacking_winner is None or stacking_method is None:
    st.stop()

# Cargar los datos preprocesados
try:
    df_estadisticas_ultimos_5 = pd.read_csv('data/df_estadisticas_ultimos_5.csv')
except FileNotFoundError:
    st.error("Archivo de estadísticas no encontrado: data/df_estadisticas_ultimos_5.csv")
    st.stop()

# Asegurarse de que las columnas del modelo se carguen correctamente
try:
    columnas_X = pd.read_csv('data/columnas_X.csv', header=None).squeeze().tolist()
except FileNotFoundError:
    st.error("Archivo de columnas no encontrado: data/columnas_X.csv")
    st.stop()

# Título de la app
st.title('Predicción de Resultados de Peleas de UFC')

# Selección de peleadores de la lista desplegable (completado automático)
fighters_list = df_estadisticas_ultimos_5['Fighter'].unique()

# Selección de peleadores de la lista desplegable
fighter_1 = st.selectbox('Selecciona el primer peleador:', fighters_list, key='fighter_1')
fighter_2 = st.selectbox('Selecciona el segundo peleador:', fighters_list, key='fighter_2')

# Verificar que los peleadores seleccionados sean diferentes
duplicate_selection = fighter_1 == fighter_2
if duplicate_selection:
    st.warning('Debes seleccionar dos peleadores distintos.')

# Obtener las estadísticas de los peleadores
stats_fighter_1 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['Fighter'] == fighter_1].drop(columns=['Fighter'])
stats_fighter_2 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['Fighter'] == fighter_2].drop(columns=['Fighter'])

if stats_fighter_1.empty or stats_fighter_2.empty:
    st.error("No se encontraron estadísticas para uno o ambos peleadores.")
else:
    # Obtener valores de forma de los últimos 5 combates desde las estadísticas
    form_last_5_fighter_1 = stats_fighter_1["form_last_5"].iloc[0]
    form_last_5_fighter_2 = stats_fighter_2["form_last_5"].iloc[0]

    # Asignar formato y clase de peso directamente desde las estadísticas filtradas
    format_input = stats_fighter_1["Format"].iloc[0]
    weight_class_input = stats_fighter_1["Weight Class"].iloc[0]

    # Mostrar valores seleccionados y calculados para transparencia
    st.write(f"Formato de la pelea: {format_input}")
    st.write(f"Clase de peso: {weight_class_input}")
    st.write(f"Forma últimos 5 de {fighter_1}: {form_last_5_fighter_1}")
    st.write(f"Forma últimos 5 de {fighter_2}: {form_last_5_fighter_2}")

    # Asegurarse de que las columnas estén en el orden correcto
    try:
        stats_fighter_1 = stats_fighter_1[columnas_X]
    except KeyError:
        missing_cols = [col for col in columnas_X if col not in stats_fighter_1.columns]
        st.error(f"Columnas faltantes en las estadísticas de {fighter_1}: {missing_cols}")
        st.stop()

    try:
        stats_fighter_2 = stats_fighter_2[columnas_X]
    except KeyError:
        missing_cols = [col for col in columnas_X if col not in stats_fighter_2.columns]
        st.error(f"Columnas faltantes en las estadísticas de {fighter_2}: {missing_cols}")
        st.stop()

    # Función para hacer la predicción del ganador
    def hacer_prediccion_winner(stacking_winner, stats_fighter_1, stats_fighter_2, fighter_1, fighter_2):
        # Realizar la predicción para ambos peleadores
        pred_proba_fighter_1 = stacking_winner.predict_proba(stats_fighter_1)
        pred_proba_fighter_2 = stacking_winner.predict_proba(stats_fighter_2)

        # Obtener las probabilidades de ganar para cada peleador
        proba_fighter_1 = pred_proba_fighter_1[0][1]  # Probabilidad de que fighter_1 gane
        proba_fighter_2 = pred_proba_fighter_2[0][1]  # Probabilidad de que fighter_2 gane

        # Normalizar las probabilidades para que sumen 100% solo si el total es mayor a 0
        total = proba_fighter_1 + proba_fighter_2
        if total > 0:
            proba_fighter_1_normalized = (proba_fighter_1 / total) * 100
            proba_fighter_2_normalized = (proba_fighter_2 / total) * 100

            st.write("\n--- Resultados de la Predicción del Ganador ---")
            st.write(f"{fighter_1}: {proba_fighter_1_normalized:.2f}%")
            st.write(f"{fighter_2}: {proba_fighter_2_normalized:.2f}%")
            st.write(f"Predicción del ganador: {fighter_1 if proba_fighter_1_normalized > proba_fighter_2_normalized else fighter_2}")
        else:
            st.error("Error: la suma de probabilidades es 0, se omite la normalización.")

    # Función para hacer la predicción del método de pelea
    def hacer_prediccion_method(stacking_method, stats_fighter_1, stats_fighter_2):
        # Realizar la predicción para ambos peleadores
        pred_method_fighter_1 = stacking_method.predict(stats_fighter_1)
        pred_method_fighter_2 = stacking_method.predict(stats_fighter_2)

        # Mapear los métodos
        method_mapping = {0: 'Decision', 1: 'KO/TKO', 2: 'Submission'}

        st.write("\n--- Resultados de la Predicción del Método de Pelea ---")
        st.write(f"Método predicho para {fighter_1}: {method_mapping[pred_method_fighter_1[0]]}")
        st.write(f"Método predicho para {fighter_2}: {method_mapping[pred_method_fighter_2[0]]}")

    # Botón para hacer la predicción
    if st.button('Hacer Predicción', disabled=duplicate_selection):
        hacer_prediccion_winner(stacking_winner, stats_fighter_1, stats_fighter_2, fighter_1, fighter_2)
        hacer_prediccion_method(stacking_method, stats_fighter_1, stats_fighter_2)
