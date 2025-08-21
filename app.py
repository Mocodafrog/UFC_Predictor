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


def _sanitize(name: str) -> str:
    """Normaliza nombres de columnas para facilitar el cruce."""
    return name.lower().replace(" ", "_").replace(".", "").replace("/", "_")


# Asegurarse de que las columnas del modelo se carguen correctamente
try:
    columnas_X_raw = pd.read_csv("data/columnas_X.csv", header=None).squeeze().tolist()
except FileNotFoundError:
    st.error("Archivo de columnas no encontrado: data/columnas_X.csv")
    st.stop()

column_map = {_sanitize(col): col for col in fight_stats.columns}
columnas_X: list[str] = []
faltantes: list[str] = []
for col in columnas_X_raw:
    if col in fight_stats.columns:
        columnas_X.append(col)
    else:
        sanitized = _sanitize(col)
        if sanitized in column_map:
            columnas_X.append(column_map[sanitized])
        else:
            faltantes.append(col)

if faltantes:
    st.warning(f"Columnas ignoradas por no existir en datos: {faltantes}")

# Replicar preprocesamiento utilizado durante el entrenamiento
weight_class_mapping: dict[str, int] = {}
if "Rounds" in fight_stats.columns:
    fight_stats["Rounds"] = pd.to_numeric(
        fight_stats["Rounds"], errors="coerce"
    )
if "Format" in fight_stats.columns:
    fight_stats["Format"] = (
        fight_stats["Format"].str.extract(r"(\d+)").astype(float)
    )
if "Weight Class" in fight_stats.columns:
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

# Verificar que columnas críticas se mantengan en columnas_X
critical_cols = [
    c for c in ["Rounds", "Format", "Weight Class"] if c in fight_stats.columns
]
missing_critical = [c for c in critical_cols if c not in columnas_X]
if missing_critical:
    st.warning(f"Columnas no incluidas en columnas_X: {missing_critical}")

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
format_list = (
    fight_stats["Format"].dropna().unique().tolist()
    if "Format" in fight_stats.columns
    else []
)
weight_class_list = (
    fight_stats["Weight Class"].dropna().unique().tolist()
    if "Weight Class" in fight_stats.columns
    else []
)

if stats_fighter_1.empty or stats_fighter_2.empty:
    st.error("No se encontraron estadísticas para uno o ambos peleadores.")
else:
    # Obtener valores de forma de los últimos 5 combates desde las estadísticas
    form_last_5_fighter_1 = stats_fighter_1["form_last_5"].iloc[0]
    form_last_5_fighter_2 = stats_fighter_2["form_last_5"].iloc[0]

    # Selección de formato y clase de peso
    default_format = stats_fighter_1["Format"].iloc[0]
    default_weight_class = stats_fighter_1["Weight Class"].iloc[0]
    format_index = (
        format_list.index(default_format) if default_format in format_list else 0
    )
    weight_class_index = (
        weight_class_list.index(default_weight_class)
        if default_weight_class in weight_class_list
        else 0
    )
    format_input = st.selectbox(
        "Selecciona el formato (rondas):",
        format_list,
        index=format_index,
    )
    weight_class_input = st.selectbox(
        "Selecciona la clase de peso:",
        weight_class_list,
        index=weight_class_index,
    )

    # Mostrar valores calculados para transparencia
    st.write(f"Formato de la pelea: {format_input}")
    st.write(f"Clase de peso: {weight_class_input}")
    st.write(f"Forma últimos 5 de {fighter_1}: {form_last_5_fighter_1}")
    st.write(f"Forma últimos 5 de {fighter_2}: {form_last_5_fighter_2}")

    stats_features_1 = stats_fighter_1[columnas_X].copy()
    stats_features_2 = stats_fighter_2[columnas_X].copy()

    # Convertir entradas del usuario al mismo código utilizado en el entrenamiento
    if "Format" in stats_features_1.columns:
        format_input_code = (
            pd.Series([format_input]).str.extract(r"(\d+)").astype(float).iloc[0]
        )
        stats_features_1["Format"] = format_input_code
        stats_features_2["Format"] = format_input_code
    if "Weight Class" in stats_features_1.columns:
        weight_class_input_code = weight_class_mapping.get(weight_class_input, -1)
        stats_features_1["Weight Class"] = weight_class_input_code
        stats_features_2["Weight Class"] = weight_class_input_code

    # Verificar que las dimensiones coinciden con las esperadas por el modelo
    if stats_features_1.shape[1] != len(columnas_X) or stats_features_2.shape[1] != len(columnas_X):
        st.error(
            "Las columnas de las estadísticas no coinciden con las utilizadas en el modelo."
        )
        st.stop()

    # Verificar que las columnas coincidan con las esperadas por el modelo
    if stats_features_1.columns.tolist() != columnas_X:
        st.error(
            "Las columnas de las características no coinciden con las esperadas por el modelo."
        )
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
    def hacer_prediccion_method(
        stacking_method, label_encoder_method, stats_fighter_1, stats_fighter_2
    ):
        # Realizar la predicción para ambos peleadores
        pred_method_fighter_1 = stacking_method.predict(stats_fighter_1)
        pred_method_fighter_2 = stacking_method.predict(stats_fighter_2)

        method_fighter_1 = label_encoder_method.inverse_transform(
            pred_method_fighter_1
        )[0]
        method_fighter_2 = label_encoder_method.inverse_transform(
            pred_method_fighter_2
        )[0]

        st.write("\n--- Resultados de la Predicción del Método de Pelea ---")
        st.write(f"Método predicho para {fighter_1}: {method_fighter_1}")
        st.write(f"Método predicho para {fighter_2}: {method_fighter_2}")

    # Botón para hacer la predicción
    if st.button('Hacer Predicción', disabled=duplicate_selection):
        hacer_prediccion_winner(
            stacking_winner, stats_features_1, stats_features_2, fighter_1, fighter_2
        )
        hacer_prediccion_method(
            stacking_method, label_encoder_method, stats_features_1, stats_features_2
        )
