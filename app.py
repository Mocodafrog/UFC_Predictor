import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Cargar los modelos entrenados
stacking_winner = joblib.load('models/stacking_winner.pkl')
stacking_method = joblib.load('models/stacking_method.pkl')

# Cargar los datos preprocesados
df_estadisticas_ultimos_5 = pd.read_csv('data/df_estadisticas_ultimos_5.csv')
columnas_X = pd.read_csv('data/columnas_X.csv', header=None).squeeze().tolist()
fight_stats = pd.read_csv('data/fight_stats.csv')

def preparar_columnas(stats, prefix):
    stats = stats.drop(columns=['fighter']).reset_index(drop=True)
    stats.columns = [f"{col}_{prefix}" for col in stats.columns]
    return stats

def calcular_metricas(df):
    for fighter in [1, 2]:
        prefix = f'figther_{fighter}'
        
        # Golpes significativos
        df[f'Sig_Str_Acc_{prefix}'] = df[f'landed_sig. str._{prefix}'] / df[f'atmp_sig. str._{prefix}'].replace(0, 1)
        df[f'Sig_Str_LpM_{prefix}'] = df[f'landed_sig. str._{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes totales
        df[f'Total_Str_Acc_{prefix}'] = df[f'landed_total str._{prefix}'] / df[f'atmp_total str._{prefix}'].replace(0, 1)
        df[f'Total_Str_LpM_{prefix}'] = df[f'landed_total str._{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes a la cabeza
        df[f'Head_Str_Acc_{prefix}'] = df[f'landed_head_{prefix}'] / df[f'atmp_head_{prefix}'].replace(0, 1)
        df[f'Head_Str_LpM_{prefix}'] = df[f'landed_head_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes al cuerpo
        df[f'Body_Str_Acc_{prefix}'] = df[f'landed_body_{prefix}'] / df[f'atmp_body_{prefix}'].replace(0, 1)
        df[f'Body_Str_LpM_{prefix}'] = df[f'landed_body_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes a las piernas
        df[f'Leg_Str_Acc_{prefix}'] = df[f'landed_leg_{prefix}'] / df[f'atmp_leg_{prefix}'].replace(0, 1)
        df[f'Leg_Str_LpM_{prefix}'] = df[f'landed_leg_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes a distancia
        df[f'Distance_Str_Acc_{prefix}'] = df[f'landed_distance_{prefix}'] / df[f'atmp_distance_{prefix}'].replace(0, 1)
        df[f'Distance_Str_LpM_{prefix}'] = df[f'landed_distance_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes en el clinch
        df[f'Clinch_Str_Acc_{prefix}'] = df[f'landed_clinch_{prefix}'] / df[f'atmp_clinch_{prefix}'].replace(0, 1)
        df[f'Clinch_Str_LpM_{prefix}'] = df[f'landed_clinch_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Golpes en el suelo
        df[f'Ground_Str_Acc_{prefix}'] = df[f'landed_ground_{prefix}'] / df[f'atmp_ground_{prefix}'].replace(0, 1)
        df[f'Ground_Str_LpM_{prefix}'] = df[f'landed_ground_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Promedio de derribos por minuto
        df[f'TD_Avg_{prefix}'] = df[f'landed_td_{prefix}'] / (df[f'Total_fight_length_sec_{prefix}'] / 60)

        # Precisión de derribos
        df[f'TD_Acc_{prefix}'] = df[f'landed_td_{prefix}'] / df[f'atmp_td_{prefix}'].replace(0, 1)

        # Defensa de derribos
        df[f'TD_Def_{prefix}'] = 1 - (df.groupby('Fight')[f'landed_td_{prefix}']
                                       .transform(lambda x: x.shift()).fillna(0) / df[f'atmp_td_{prefix}'].replace(0, 1))

        # Ratio de control
        df[f'Control_Ratio_{prefix}'] = df[f'Control Time Sec_{prefix}'] / df[f'Total_fight_length_sec_{prefix}'].replace(0, 1)

    return df


def crear_dataframe_pelea(fighter_1, fighter_2, df_estadisticas_ultimos_5):
    stats_fighter_1 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['fighter'] == fighter_1]
    stats_fighter_2 = df_estadisticas_ultimos_5[df_estadisticas_ultimos_5['fighter'] == fighter_2]

    if stats_fighter_1.empty or stats_fighter_2.empty:
        st.error("No se encontraron estadísticas para uno o ambos peleadores.")
        return None

    stats_fighter_1 = preparar_columnas(stats_fighter_1, "figther_1")
    stats_fighter_2 = preparar_columnas(stats_fighter_2, "figther_2")

    df_pelea = pd.concat([stats_fighter_1, stats_fighter_2], axis=1)
    return df_pelea



st.title('Predicción de Resultados de Peleas de UFC')
# Crear un DataFrame con la forma correspondiente para cada peleador
forma_fighter_1 = fight_stats[['Fighter_figther_1', 'form_last_5_figther_1']].rename(
    columns={'Fighter_figther_1': 'fighter', 'form_last_5_figther_1': 'form'}
)
forma_fighter_2 = fight_stats[['Fighter_figther_2', 'form_last_5_figther_2']].rename(
    columns={'Fighter_figther_2': 'fighter', 'form_last_5_figther_2': 'form'}
)

# Combinar ambas formas en un solo DataFrame
formas_combined = pd.concat([forma_fighter_1, forma_fighter_2]).drop_duplicates(subset='fighter')

# Unir 'formas_combined' con 'df_estadisticas_ultimos_5' para tener la forma disponible por peleador
df_estadisticas_completo = pd.merge(
    df_estadisticas_ultimos_5, formas_combined, on='fighter', how='left'
)

# Actualizamos la lista de peleadores
fighters_list = df_estadisticas_completo['fighter'].unique()

# Selección de peleadores con completado automático
fighter_1 = st.selectbox('Selecciona el primer peleador:', fighters_list, key='fighter_1')
fighter_2 = st.selectbox('Selecciona el segundo peleador:', fighters_list, key='fighter_2')

# Obtener la forma correspondiente para los peleadores seleccionados
forma_fighter_1 = df_estadisticas_completo[df_estadisticas_completo['fighter'] == fighter_1]['form'].values[0]
forma_fighter_2 = df_estadisticas_completo[df_estadisticas_completo['fighter'] == fighter_2]['form'].values[0]



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
#form_fighter_1 = st.selectbox(f'Introduce la forma de {fighter_1}:', list(form_mapping.keys()), key='form_fighter_1')
#form_fighter_2 = st.selectbox(f'Introduce la forma de {fighter_2}:', list(form_mapping.keys()), key='form_fighter_2')

#form_last_5_fighter_1 = form_mapping[form_fighter_1]
#form_last_5_fighter_2 = form_mapping[form_fighter_2]


df_pelea_futura = crear_dataframe_pelea(fighter_1, fighter_2, df_estadisticas_ultimos_5)

if df_pelea_futura is not None:
    # Agregar los valores adicionales antes de la predicción
    df_pelea_futura['Format'] = format_input
    df_pelea_futura['Weight Class'] = weight_class_input
    df_pelea_futura['form_last_5_figther_1'] = forma_fighter_1
    df_pelea_futura['form_last_5_figther_2'] = forma_fighter_2


    # Asegurar el orden correcto de columnas
    df_pelea_futura = df_pelea_futura[columnas_X]

    # Crear DataFrame invertido (figther_1 y figther_2 intercambiados)
    df_pelea_invertida = df_pelea_futura.copy()

    # Intercambiar las estadísticas entre figther_1 y figther_2
    for col in df_pelea_futura.columns:
        if '_figther_1' in col:
            col_2 = col.replace('_figther_1', '_figther_2')
            df_pelea_invertida[col], df_pelea_invertida[col_2] = (
                df_pelea_invertida[col_2].values,
                df_pelea_invertida[col].values,
            )

    # Asegurar que ambas tengan el mismo orden de columnas
    df_pelea_invertida = df_pelea_invertida[columnas_X]

    # Botón para hacer la predicción
    if st.button('Hacer Predicción'):
        # ---------------------------
        # 1. Predicción del Ganador
        # ---------------------------
        # Primera predicción con el orden original
        proba_1 = stacking_winner.predict_proba(df_pelea_futura)

        # Segunda predicción con el orden invertido
        proba_2 = stacking_winner.predict_proba(df_pelea_invertida)

        # Promediar las probabilidades de ambas predicciones
        proba_final = (proba_1 + proba_2[:, ::-1]) / 2

        # Obtener la predicción final basada en el promedio de probabilidades
        pred_final = np.argmax(proba_final, axis=1)

        # Mostrar las probabilidades finales en porcentaje
        fighter_1_proba = proba_final[0][0] * 100
        fighter_2_proba = proba_final[0][1] * 100

        st.write("\n--- Predicción del Ganador ---")
        st.write(f"{fighter_1}: {fighter_1_proba:.2f}%")
        st.write(f"{fighter_2}: {fighter_2_proba:.2f}%")
        st.write(f"Ganador: {fighter_2 if pred_final[0] == 1 else fighter_1}")

        # --------------------------------------
        # 2. Predicción del Método de Pelea
        # --------------------------------------
        # Primera predicción del método con el orden original
        # Primera predicción del método con el orden original
        proba_method_1 = stacking_method.predict_proba(df_pelea_futura)

        # Segunda predicción del método con el orden invertido
        proba_method_2 = stacking_method.predict_proba(df_pelea_invertida)

        # Aseguramos que las probabilidades estén alineadas correctamente
        # No es necesario invertir las columnas aquí ya que los métodos (Decision, KO/TKO, Submission) no dependen del orden del peleador
        proba_method_final = (proba_method_1 + proba_method_2) / 2

        # Obtener la predicción final basada en el promedio de probabilidades
        pred_method_final = np.argmax(proba_method_final, axis=1)

        # Mapear los métodos
        method_mapping = {0: 'Decision', 1: 'KO/TKO', 2: 'Submission'}

        # Mostrar las probabilidades finales del método en porcentaje
        st.write("\n--- Predicción del Método de Pelea ---")
        for i, prob in enumerate(proba_method_final[0]):
            st.write(f"{method_mapping[i]}: {prob * 100:.2f}%")

        st.write(f"Método Predicho: {method_mapping[pred_method_final[0]]}")





