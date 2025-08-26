#  UFC Predictor – Predicción de Resultados en Combates de UFC

Este proyecto desarrolla un sistema completo de análisis y predicción de resultados en combates de UFC utilizando técnicas de machine learning, procesamiento de datos, y despliegue de aplicaciones web.

El flujo de trabajo incluye desde el web scraping de datos de peleadores y combates históricos, pasando por la ingeniería de características y la construcción de modelos predictivos, hasta la creación de una aplicación web interactiva desplegada en Streamlit.

## Tecnologías utilizadas

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- MySQL (AWS RDS)
- Streamlit
- BeautifulSoup4 (Web Scraping)

## Estructura del proyecto

- `ufc_predictor.scraping`: Script de extracción de datos desde UFCStats.com
- `ufc_predictor.preprocessing`: Limpieza y preparación de los datos
- `entrenamiento.py`: Entrenamiento y validación de modelos predictivos
- `app.py`: Código de la aplicación Streamlit
- `requirements.txt`: Librerías necesarias para ejecutar el proyecto
- `README.md`: Este documento

## Funcionalidades principales

- Extracción automática de estadísticas de peleas y peleadores.
- Limpieza y consolidación de datos en un esquema relacional.
- Entrenamiento de múltiples modelos de clasificación para:
  - Predicción del ganador del combate.
  - Predicción del método de victoria.
- Aplicación web interactiva que permite al usuario cargar parámetros de un combate y obtener predicciones.

## Diferencia de cálculo de probabilidades

La aplicación ahora muestra las probabilidades de victoria generadas por el modelo sin normalizarlas por la suma de ambos peleadores. Para fines informativos también se presentan porcentajes normalizados, pero la determinación del ganador se basa en las probabilidades originales.

## Demo de la Aplicación

La aplicación Streamlit está disponible públicamente en el siguiente enlace:

 [Acceder a la app](https://ufcpredictorwm.streamlit.app/)

## 📥 Instalación local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Mocodafrog/UFC_Predictor.git
   cd UFC_Predictor
   ```
2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv env
   source env/bin/activate  # en Linux/macOS
   env\Scripts\activate   # en Windows
   ```
3. Instala las dependencias principales:
   ```bash
   pip install -r requirements.txt
   ```
4. (Opcional) Instala dependencias de desarrollo como `pymysql` para `upload_to_sql.py`:
   ```bash
   pip install -r dev-requirements.txt
   ```
5. Corre la aplicación Streamlit:
   ```bash
   streamlit run app.py
   ```

### Dependencias opcionales para análisis

Algunas librerías se utilizan solo para análisis puntuales y no se instalan por defecto. Instálalas manualmente si las necesitas:

- matplotlib
- selenium
- webdriver-manager
- scipy
- tensorflow
- openpyxl

## 🏋️ Entrenamiento de modelos

Los modelos se entrenan mediante scripts ubicados en la raíz del proyecto.
Los artefactos generados se guardan en `models/{MODEL_VERSION}/`, donde la
versión se centraliza en `ufc_predictor/config.py`.

Antes de ejecutar estos scripts, asegúrate de que el archivo `data/fight_stats.csv` exista,
ya que contiene las características utilizadas para el entrenamiento.

Puedes ajustar el comportamiento mediante variables de entorno:

- `MODEL_NAMES`: lista separada por comas de modelos a entrenar.
- `FAST_MODE=1`: usa un subconjunto de datos y grids mínimos para ejecuciones rápidas.
- `EXTENDED_SEARCH=1`: activa grids de hiperparámetros más amplios para una búsqueda exhaustiva.
  Si ambos flags se usan, `FAST_MODE` tiene prioridad.

Opcionalmente puedes suministrar un archivo JSON o YAML con ajustes de
hiperparámetros por modelo. Pásalo como primer argumento al script y sus
valores se combinarán con los grids por defecto. Ejemplo:

```json
// grid.json
{
  "RandomForest": {"n_estimators": [150, 200]},
  "LogisticRegression": {"C": [0.5, 2.0]}
}
```

```bash
python entrenamiento.py grid.json
```

Para entrenar el modelo del método de victoria:

```bash
python entrenamiento.py
```

Para entrenar el modelo del ganador del combate:

```bash
python entrenamiento_winner.py
```

### Ajustes rápidos en CI

Para acelerar la integración continua, los grids de hiperparámetros en
`ufc_predictor/train.py` usan valores reducidos.  Por ejemplo,
`n_estimators` se limita a `[50, 100]`, `max_depth` a `[3, 5]` y
`learning_rate` queda fijo en `0.1`, lo que implica menos de 50 "fits" por
modelo con validación cruzada de 3 pliegues.  Fuera del entorno de CI puedes
activar `EXTENDED_SEARCH=1` para ampliar estas combinaciones.

## 📊 Agregación vs Dataset Rolling

- `ufc_predictor.analysis.aggregate_last_five_stats`: produce un único registro por
  peleador calculando la media de las estadísticas de sus últimas cinco
  peleas.
- `ufc_predictor.fight_stats.compute_last_five_stats`: genera un dataset con una fila por
  combate e incluye medias móviles y la racha de victorias de los últimos
  cinco enfrentamientos.

