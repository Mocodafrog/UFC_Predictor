import pandas as pd
import sqlalchemy
import os
from sqlalchemy import create_engine

# Cargar credenciales desde variables de entorno (secretos de GitHub)
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# Crear la cadena de conexión a la base de datos
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

# Cargar el archivo CSV generado por el pipeline
df = pd.read_csv('data/fight_stats.csv')

# Subir el DataFrame a la base de datos
try:
    df.to_sql('fight_stats', con=engine, if_exists='replace', index=False)
    print("Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f"Error al cargar datos en la base de datos: {e}")
