import streamlit as st
import joblib
import pandas as pd
import json

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


@st.cache_resource(show_spinner="Cargando codificador del método...")
def load_label_encoder_method(version: str):
    try:
        return joblib.load(f"models/{version}/label_encoder_method.pkl")
    except FileNotFoundError:
        st.error(
            f"No se encontró el codificador: models/{version}/label_encoder_method.pkl"
        )
        return None


# Cargar los modelos entrenados
stacking_winner = load_stacking_winner(MODEL_VERSION)
stacking_method = load_stacking_method(MODEL_VERSION)
label_encoder_method = load_label_encoder_method(MODEL_VERSION)
if stacking_winner is None or stacking_method is None or label_encoder_method is None:
    st.stop()

# Cargar los datos preprocesados
try:
    fight_stats = pd.read_csv("data/fight_stats.csv")
except FileNotFoundError:
    st.error("Archivo de estadísticas no encontrado: data/fight_stats.csv")
    st.stop()

# Cargar la lista de columnas utilizadas durante el entrenamiento
try:
    with open(
        f"models/{MODEL_VERSION}/features_winner.json", "r"
    ) as f:
        columnas_X = json.load(f)
except FileNotFoundError:
    st.error(
        f"No se encontró el archivo de características: models/{MODEL_VERSION}/features_winner.json"
    )
    st.stop()

faltantes = [c for c in columnas_X if c not in fight_stats.columns]
if faltantes:
    st.error(f"Columnas faltantes en datos: {faltantes}")
    st.stop()

# Remover de fight_stats las columnas que no se usaron en el entrenamiento
extra_cols = [
    c
    for c in ["Rounds", "Format", "Weight Class"]
    if c in fight_stats.columns and c not in columnas_X
]
if extra_cols:
    fight_stats = fight_stats.drop(columns=extra_cols)

# Replicar preprocesamiento utilizado durante el entrenamiento
weight_class_mapping: dict[str, int] = {}
if "Rounds" in columnas_X and "Rounds" in fight_stats.columns:
    fight_stats["Rounds"] = pd.to_numeric(
        fight_stats["Rounds"], errors="coerce"
    )
if "Format" in columnas_X and "Format" in fight_stats.columns:
    fight_stats["Format"] = (
        fight_stats["Format"].astype(str).str.extract(r"(\d+)").astype(float)
    )
if "Weight Class" in columnas_X and "Weight Class" in fight_stats.columns:
    weight_class_cat = pd.Categorical(fight_stats["Weight Class"])
    weight_class_mapping = {
        cat: code for code, cat in enumerate(weight_class_cat.categories)
    }
    fight_stats["Weight Class"] = weight_class_cat.codes

# Detectar columnas no numéricas tras el preprocesamiento
no_numericas = fight_stats[columnas_X].select_dtypes(exclude="number").columns.tolist()
if no_numericas:
    st.warning(f"Columnas no numéricas ignoradas: {no_numericas}")
    columnas_X = [c for c in columnas_X if c not in no_numericas]

# Título de la app
st.title('Predicción de Resultados de Peleas de UFC')

# Selección de peleadores de la lista desplegable (completado automático)
fighters_list = fight_stats["Fighter"].unique()

# Selección de peleadores de la lista desplegable
fighter_1 = st.selectbox('Selecciona el primer peleador:', fighters_list, key='fighter_1')
fighter_2 = st.selectbox('Selecciona el segundo peleador:', fighters_list, key='fighter_2')

# Verificar que los peleadores seleccionados sean diferentes
duplicate_selection = fighter_1 == fighter_2
if duplicate_selection:
    st.warning('Debes seleccionar dos peleadores distintos.')

# Obtener las estadísticas de los peleadores
stats_fighter_1 = (
    fight_stats[fight_stats["Fighter"] == fighter_1]
    .sort_values("date", ascending=False)
    .head(1)
)
stats_fighter_2 = (
    fight_stats[fight_stats["Fighter"] == fighter_2]
    .sort_values("date", ascending=False)
    .head(1)
)

# Listas de formatos y clases de peso disponibles
if "Format" in fight_stats.columns:
    format_list = fight_stats["Format"].dropna().unique().tolist()
elif "Rounds" in fight_stats.columns:
    format_list = (
        fight_stats["Rounds"].dropna().astype(int).astype(str).unique().tolist()
    )
else:
    format_list = []
inverse_weight_class_mapping = {v: k for k, v in weight_class_mapping.items()}
weight_class_list = list(weight_class_mapping.keys()) if weight_class_mapping else []

if stats_fighter_1.empty or stats_fighter_2.empty:
    st.error("No se encontraron estadísticas para uno o ambos peleadores.")
else:
    # Obtener valores de forma de los últimos 5 combates desde las estadísticas
    form_last_5_fighter_1 = stats_fighter_1["form_last_5"].iloc[0]
    form_last_5_fighter_2 = stats_fighter_2["form_last_5"].iloc[0]

    # Selección de formato y clase de peso
    default_format = (
        stats_fighter_1["Format"].iloc[0]
        if "Format" in stats_fighter_1.columns
        else stats_fighter_1["Rounds"].iloc[0]
        if "Rounds" in stats_fighter_1.columns
        else None
    )
    default_weight_class_code = (
        stats_fighter_1["Weight Class"].iloc[0]
        if "Weight Class" in stats_fighter_1.columns
        else None
    )
    default_weight_class = (
        inverse_weight_class_mapping.get(default_weight_class_code)
        if default_weight_class_code is not None
        else None
    )
    format_index = (
        format_list.index(default_format) if default_format in format_list else 0
    )
    if format_list:
        format_input = st.selectbox(
            "Selecciona el formato (rondas):",
            format_list,
            index=format_index,
        )
    else:
        format_input = default_format
    if weight_class_list:
        weight_class_index = (
            weight_class_list.index(default_weight_class)
            if default_weight_class in weight_class_list
            else 0
        )
        weight_class_label = st.selectbox(
            "Selecciona la clase de peso:",
            weight_class_list,
            index=weight_class_index,
        )
    else:
        weight_class_label = None
    # Mostrar valores calculados para transparencia
    st.write(f"Formato/Rounds de la pelea: {format_input}")
    if weight_class_label is not None:
        st.write(f"Clase de peso: {weight_class_label}")
    st.write(f"Forma últimos 5 de {fighter_1}: {form_last_5_fighter_1}")
    st.write(f"Forma últimos 5 de {fighter_2}: {form_last_5_fighter_2}")

    stats_features_1 = stats_fighter_1[columnas_X].copy()
    stats_features_2 = stats_fighter_2[columnas_X].copy()
    stats_features_1 = stats_features_1.drop(columns=["win"], errors="ignore")
    stats_features_2 = stats_features_2.drop(columns=["win"], errors="ignore")
    if "win" in stats_features_1.columns or "win" in stats_features_2.columns:
        st.error("La columna 'win' no debe estar presente en las características")
        st.stop()
    # Convertir entradas del usuario al mismo código utilizado en el entrenamiento
    # Parsear el formato seleccionado y asignarlo a las columnas correspondientes
    if ("Format" in columnas_X) or ("Rounds" in columnas_X):
        format_input_code = (
            pd.Series([format_input])
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
            .iloc[0]
        )
        if "Format" in columnas_X:
            stats_features_1["Format"] = format_input_code
            stats_features_2["Format"] = format_input_code
        if "Rounds" in columnas_X:
            stats_features_1["Rounds"] = format_input_code
            stats_features_2["Rounds"] = format_input_code
    if "Weight Class" in columnas_X and weight_class_label is not None:
        weight_class_input_code = weight_class_mapping[weight_class_label]
        stats_features_1["Weight Class"] = weight_class_input_code
        stats_features_2["Weight Class"] = weight_class_input_code

    # Alinear el orden de columnas con el usado durante el entrenamiento
    stats_features_1 = stats_features_1.reindex(columns=columnas_X)
    stats_features_2 = stats_features_2.reindex(columns=columnas_X)

    # Función para hacer la predicción del ganador
    def hacer_prediccion_winner(stacking_winner, stats_fighter_1, stats_fighter_2, fighter_1, fighter_2):
        # Realizar la predicción para ambos peleadores
        pred_proba_fighter_1 = stacking_winner.predict_proba(stats_fighter_1)
        pred_proba_fighter_2 = stacking_winner.predict_proba(stats_fighter_2)

        # Obtener las probabilidades de ganar para cada peleador
        proba_fighter_1 = pred_proba_fighter_1[0][1]  # Probabilidad de que fighter_1 gane
        proba_fighter_2 = pred_proba_fighter_2[0][1]  # Probabilidad de que fighter_2 gane

        st.write("\n--- Resultados de la Predicción del Ganador ---")

        # Mostrar probabilidades normalizadas
        total = proba_fighter_1 + proba_fighter_2
        if total > 0:
            proba_fighter_1_normalized = proba_fighter_1 / total * 100
            proba_fighter_2_normalized = proba_fighter_2 / total * 100
            st.write("Probabilidades normalizadas:")
            st.write(f"{fighter_1}: {proba_fighter_1_normalized:.2f}%")
            st.write(f"{fighter_2}: {proba_fighter_2_normalized:.2f}%")
        else:
            st.error(
                "Error: la suma de probabilidades es 0, se omite la normalización."
            )

        st.write(
            f"Predicción del ganador: {fighter_1 if proba_fighter_1 > proba_fighter_2 else fighter_2}"
        )

    # Función para hacer la predicción del método de pelea
    def hacer_prediccion_method(
        stacking_method, label_encoder_method, stats_fighter_1, stats_fighter_2
    ):
        # Obtener probabilidades de cada método para ambos peleadores
        proba_fighter_1 = stacking_method.predict_proba(stats_fighter_1)[0]
        proba_fighter_2 = stacking_method.predict_proba(stats_fighter_2)[0]

        # Obtener nombres de clases
        n_clases = len(proba_fighter_1)
        class_names = label_encoder_method.inverse_transform(range(n_clases))

        # Combinar clases con probabilidades y convertir a porcentajes
        df_fighter_1 = (
            pd.DataFrame(
                {
                    "Método": class_names,
                    "Probabilidad (%)": proba_fighter_1 * 100,
                }
            )
            .sort_values("Probabilidad (%)", ascending=False)
            .reset_index(drop=True)
        )
        df_fighter_2 = (
            pd.DataFrame(
                {
                    "Método": class_names,
                    "Probabilidad (%)": proba_fighter_2 * 100,
                }
            )
            .sort_values("Probabilidad (%)", ascending=False)
            .reset_index(drop=True)
        )

        st.write("\n--- Resultados de la Predicción del Método de Pelea ---")
        st.write(f"Probabilidades para {fighter_1}:")
        st.write(df_fighter_1)
        st.write(f"Probabilidades para {fighter_2}:")
        st.write(df_fighter_2)

    # Botón para hacer la predicción
    if st.button('Hacer Predicción', disabled=duplicate_selection):
        hacer_prediccion_winner(
            stacking_winner, stats_features_1, stats_features_2, fighter_1, fighter_2
        )
        hacer_prediccion_method(
            stacking_method, label_encoder_method, stats_features_1, stats_features_2
        )
