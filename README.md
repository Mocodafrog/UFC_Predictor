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

- `scraping.py`: Script de extracción de datos desde UFCStats.com
- `preprocessing.py`: Limpieza y preparación de los datos
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
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Corre la aplicación Streamlit:
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
Los artefactos generados se guardan en `models/{MODEL_VERSION}/`.

Para entrenar el modelo del método de victoria:

```bash
python entrenamiento.py
```

Para entrenar el modelo del ganador del combate:

```bash
python entrenamiento_winner.py
```

