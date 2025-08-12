import streamlit as st
import joblib
import pandas as pd

MODEL_VERSION = "1.0"  # Versión del modelo utilizada para construir las rutas


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

# Selección de formato (se convierte a valor numérico según tu tabla)
format_mapping = {'3 Rnd (5-5-5)': 0, '3 Rnd + OT (5-5-5-5)': 1, '5 Rnd (5-5-5-5-5)': 2}
format_selection = st.selectbox('Selecciona el formato de la pelea:', list(format_mapping.keys()), key='format_selection')
format_input = format_mapping[format_selection]

# Selección de clase de peso (se convierte a valor numérico según tu tabla)
weight_class_mapping = {
    'Bantamweight': 0, 'Catchweight': 1, 'Featherweight': 2, 'Flyweight': 3, 'Heavyweight': 4,
    'Light Heavyweight': 5, 'Lightweight': 6, 'Middleweight': 7, 'Welterweight': 8, 'Women\'s Bantamweight': 9,
    'Women\'s Featherweight': 10, 'Women\'s Flyweight': 11, 'Women\'s Strawweight': 12
}
weight_class_selection = st.selectbox('Selecciona la clase de peso:', list(weight_class_mapping.keys()), key='weight_class_selection')
weight_class_input = weight_class_mapping[weight_class_selection]

# Forma actual de los peleadores (se convierte a valor numérico según tu tabla)
form_mapping = {
    "": 0, "L": 1, "LL": 2, "LLL": 3, "LLLL": 4, "LLLLL": 5, "LLLLW": 6,
    "LLLW": 7, "LLLWL": 8, "LLLWW": 9, "LLW": 10, "LLWL": 11, "LLWLL": 12, "LLWLW": 13,
    "LLWW": 14, "LLWWL": 15, "LLWWW": 16, "LW": 17, "LWL": 18, "LWLL": 19, "LWLLL": 20,
    "LWLLW": 21, "LWLW": 22, "LWLWL": 23, "LWLWW": 24, "LWW": 25, "LWWL": 26, "LWWLL": 27,
    "LWWLW": 28, "LWWW": 29, "LWWWL": 30, "LWWWW": 31, "W": 32, "WL": 33, "WLL": 34,
    "WLLL": 35, "WLLLL": 36, "WLLLW": 37, "WLLW": 38, "WLLWL": 39, "WLLWW": 40, "WLW": 41,
    "WLWL": 42, "WLWLL": 43, "WLWLW": 44, "WLWW": 45, "WLWWL": 46, "WLWWW": 47, "WW": 48,
    "WWL": 49, "WWLL": 50, "WWLLL": 51, "WWLLW": 52, "WWLW": 53, "WWLWL": 54, "WWLWW": 55,
    "WWW": 56, "WWWL": 57, "WWWLL": 58, "WWWLW": 59, "WWWW": 60, "WWWWL": 61, "WWWWW": 62
}
# Forma actual de los peleadores (con clave única)
form_fighter_1 = st.selectbox(f'Introduce la forma de {fighter_1}:', list(form_mapping.keys()), key='form_fighter_1')
form_fighter_2 = st.selectbox(f'Introduce la forma de {fighter_2}:', list(form_mapping.keys()), key='form_fighter_2')

form_last_5_fighter_1 = form_mapping[form_fighter_1]
form_last_5_fighter_2 = form_mapping[form_fighter_2]

# Obtener las estadísticas de los peleadores
stats_fighter_1 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['Fighter'] == fighter_1].drop(columns=['Fighter'])
stats_fighter_2 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['Fighter'] == fighter_2].drop(columns=['Fighter'])

if stats_fighter_1.empty or stats_fighter_2.empty:
    st.error("No se encontraron estadísticas para uno o ambos peleadores.")
else:
    # Agregar los valores adicionales a las estadísticas de cada peleador
    stats_fighter_1['Format'] = format_input
    stats_fighter_1['form_last_5'] = form_last_5_fighter_1
    stats_fighter_1['Weight Class'] = weight_class_input

    stats_fighter_2['Format'] = format_input
    stats_fighter_2['form_last_5'] = form_last_5_fighter_2
    stats_fighter_2['Weight Class'] = weight_class_input

    # Asegurarse de que las columnas estén en el orden correcto
    stats_fighter_1 = stats_fighter_1[columnas_X]
    stats_fighter_2 = stats_fighter_2[columnas_X]

    # Función para hacer la predicción del ganador
    def hacer_prediccion_winner(stacking_winner, stats_fighter_1, stats_fighter_2, fighter_1, fighter_2):
        # Realizar la predicción para ambos peleadores
        pred_proba_fighter_1 = stacking_winner.predict_proba(stats_fighter_1)
        pred_proba_fighter_2 = stacking_winner.predict_proba(stats_fighter_2)

        # Obtener las probabilidades de ganar para cada peleador
        proba_fighter_1 = pred_proba_fighter_1[0][1]  # Probabilidad de que fighter_1 gane
        proba_fighter_2 = pred_proba_fighter_2[0][1]  # Probabilidad de que fighter_2 gane

        # Normalizar las probabilidades para que sumen 100%
        total = proba_fighter_1 + proba_fighter_2
        proba_fighter_1_normalized = (proba_fighter_1 / total) * 100
        proba_fighter_2_normalized = (proba_fighter_2 / total) * 100

        st.write("\n--- Resultados de la Predicción del Ganador ---")
        st.write(f"{fighter_1}: {proba_fighter_1_normalized:.2f}%")
        st.write(f"{fighter_2}: {proba_fighter_2_normalized:.2f}%")
        st.write(f"Predicción del ganador: {fighter_1 if proba_fighter_1_normalized > proba_fighter_2_normalized else fighter_2}")

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
