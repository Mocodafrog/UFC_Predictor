import streamlit as st
import joblib
import pandas as pd

# Cargar los modelos entrenados
stacking_winner = joblib.load('models/stacking_winner.pkl')
stacking_method = joblib.load('models/stacking_method.pkl')

# Cargar los datos preprocesados
df_estadisticas_ultimos_5 = pd.read_csv('data/df_estadisticas_ultimos_5.csv')

# Asegurarse de que las columnas del modelo se carguen correctamente
X = pd.read_csv('data/X_columns.csv')  # Aquí deberías tener las columnas de entrenamiento
X_columns = X.columns

# Crear DataFrame para la pelea futura basado en los datos de los últimos 5 combates
def crear_dataframe_pelea(fighter_1, fighter_2, df_estadisticas_ultimos_5):
    # Obtener las estadísticas de los dos peleadores
    stats_fighter_1 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['fighter'] == fighter_1]
    stats_fighter_2 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['fighter'] == fighter_2]

    if stats_fighter_1.empty or stats_fighter_2.empty:
        st.error("No se encontraron estadísticas para uno o ambos peleadores.")
        return None

    # Renombrar columnas para coincidir con fighter_1 y fighter_2
    stats_fighter_1 = stats_fighter_1.drop(columns=['fighter'])
    stats_fighter_1.columns = [col + '_figther_1' for col in stats_fighter_1.columns]
    
    stats_fighter_2 = stats_fighter_2.drop(columns=['fighter'])
    stats_fighter_2.columns = [col + '_figther_2' for col in stats_fighter_2.columns]

    # Unir las estadísticas de ambos peleadores en un solo DataFrame
    df_pelea_futura = pd.concat([stats_fighter_1.reset_index(drop=True), stats_fighter_2.reset_index(drop=True)], axis=1)

    return df_pelea_futura

# Título de la app
st.title('Predicción de Resultados de Peleas de UFC')

# Selección de peleadores de la lista desplegable (completado automático)
fighters_list = df_estadisticas_ultimos_5['fighter'].unique()

fighter_1 = st.selectbox('Selecciona el primer peleador:', fighters_list)
fighter_2 = st.selectbox('Selecciona el segundo peleador:', fighters_list)

# Selección de formato (se convierte a valor numérico según tu tabla)
format_mapping = {'3 Rnd (5-5-5)': 0, '3 Rnd + OT (5-5-5-5)': 1, '5 Rnd (5-5-5-5-5)': 2}
format_selection = st.selectbox('Selecciona el formato de la pelea:', list(format_mapping.keys()))
format_input = format_mapping[format_selection]

# Selección de clase de peso (se convierte a valor numérico según tu tabla)
weight_class_mapping = {
    'Bantamweight': 0, 'Catchweight': 1, 'Featherweight': 2, 'Flyweight': 3, 'Heavyweight': 4,
    'Light Heavyweight': 5, 'Lightweight': 6, 'Middleweight': 7, 'Welterweight': 8, 'Women\'s Bantamweight': 9,
    'Women\'s Featherweight': 10, 'Women\'s Flyweight': 11, 'Women\'s Strawweight': 12
}
weight_class_selection = st.selectbox('Selecciona la clase de peso:', list(weight_class_mapping.keys()))
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
form_fighter_1 = st.selectbox(f'Introduce la forma de {fighter_1}:', list(form_mapping.keys()))
form_fighter_2 = st.selectbox(f'Introduce la forma de {fighter_2}:', list(form_mapping.keys()))

form_last_5_fighter_1 = form_mapping[form_fighter_1]
form_last_5_fighter_2 = form_mapping[form_fighter_2]

# Crear el DataFrame de la pelea futura con estos valores
df_pelea_futura = crear_dataframe_pelea(fighter_1, fighter_2, df_estadisticas_ultimos_5)

if df_pelea_futura is not None:
    # Agregar los valores adicionales
    df_pelea_futura['Format'] = format_input
    df_pelea_futura['form_last_5_figther_1'] = form_last_5_fighter_1
    df_pelea_futura['form_last_5_figther_2'] = form_last_5_fighter_2
    df_pelea_futura['Weight Class'] = weight_class_input

    # Asegurarse de que el DataFrame tenga las mismas columnas en el mismo orden que el modelo espera
    columnas_faltantes = set(columnas_X) - set(df_pelea_futura.columns)
    columnas_adicionales = set(df_pelea_futura.columns) - set(columnas_X)

    if columnas_faltantes:
        st.error(f"Error: Faltan las siguientes columnas en el DataFrame: {columnas_faltantes}")
    elif columnas_adicionales:
        st.warning(f"Advertencia: Hay columnas adicionales en el DataFrame que no se esperaban: {columnas_adicionales}")
    else:
        # Asegurarse de que las columnas estén en el orden correcto
        df_pelea_futura = df_pelea_futura[columnas_X]



        # Botón para hacer la predicción
        if st.button('Hacer Predicción'):
            hacer_prediccion_winner(stacking_winner, df_pelea_futura, fighter_1, fighter_2)
            hacer_prediccion_method(stacking_method, df_pelea_futura)

# Función para hacer la predicción del ganador
def hacer_prediccion_winner(stacking_winner, df_pelea_futura, fighter_1, fighter_2):
    if df_pelea_futura is not None:
        # Realizar la predicción
        pred_winner = stacking_winner.predict(df_pelea_futura)
        pred_proba_winner = stacking_winner.predict_proba(df_pelea_futura)

        # Mostrar las probabilidades en porcentaje
        fighter_1_proba = pred_proba_winner[0][0] * 100
        fighter_2_proba = pred_proba_winner[0][1] * 100

        st.write("\n--- Resultados de la Predicción del Ganador ---")
        st.write(f"{fighter_1}: {fighter_1_proba:.2f}%")
        st.write(f"{fighter_2}: {fighter_2_proba:.2f}%")
        st.write(f"Predicción del ganador: {fighter_2 if pred_winner[0] == 1 else fighter_1}")

# Función para hacer la predicción del método de pelea
def hacer_prediccion_method(stacking_method, df_pelea_futura):
    if df_pelea_futura is not None:
        # Realizar la predicción
        pred_method = stacking_method.predict(df_pelea_futura)
        pred_proba_method = stacking_method.predict_proba(df_pelea_futura)

        # Mapear los métodos
        method_mapping = {0: 'Decision', 1: 'KO/TKO', 2: 'Submission'}
        
        # Mostrar las probabilidades en porcentaje
        method_0_proba = pred_proba_method[0][0] * 100
        method_1_proba = pred_proba_method[0][1] * 100
        method_2_proba = pred_proba_method[0][2] * 100

        st.write("\n--- Resultados de la Predicción del Método de Pelea ---")
        st.write(f"{method_mapping[0]}: {method_0_proba:.2f}%")
        st.write(f"{method_mapping[1]}: {method_1_proba:.2f}%")
        st.write(f"{method_mapping[2]}: {method_2_proba:.2f}%")
        st.write(f"Predicción del método: {method_mapping[pred_method[0]]}")



