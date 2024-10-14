import streamlit as st

# Aquí asumo que df_estadisticas_ultimos_5 ya existe y se genera directamente en tu pipeline

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
    pelea_df = pd.concat([stats_fighter_1.reset_index(drop=True), stats_fighter_2.reset_index(drop=True)], axis=1)

    return pelea_df

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
    'Bantamweight': 0, 'Catchweight': 1, 'Featherweight': 2, 'Flyweight': 3, 'Heavyweight': 4, 'Heavyweight Title': 5,
    'Light Heavyweight': 6, 'Lightweight': 7, 'Middleweight': 9, 'Welterweight': 11, 'Women\'s Bantamweight': 12,
    'Women\'s Featherweight': 13, 'Women\'s Flyweight': 14, 'Women\'s Strawweight': 15
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

    # Mostrar las estadísticas generadas
    st.write("--- Estadísticas generadas para la predicción ---")
    st.write(df_pelea_futura)

    # Botón para hacer la predicción
    if st.button('Hacer Predicción'):
        # Aquí debes llamar a la función que hace la predicción
        hacer_prediccion_winner(stacking_winner, df_pelea_futura)
        hacer_prediccion_method(stacking_method, df_pelea_futura)

# Función para hacer la predicción del ganador
def hacer_prediccion_winner(stacking_winner, df_pelea_futura):
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

