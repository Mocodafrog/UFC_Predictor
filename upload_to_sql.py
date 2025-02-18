import pandas as pd
import sqlalchemy
import os

# Cargar credenciales desde variables de entorno
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# Revisar si las variables están cargadas correctamente
if not all([db_user, db_password, db_host, db_port, db_name]):
    raise ValueError("❌ ERROR: Faltan variables de entorno. Revisa GitHub Secrets.")

# Crear cadena de conexión para SQL Server
connection_string = (
    f"mssql+pyodbc://{db_user}:{db_password}@{db_host},{db_port}/{db_name}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)

# Crear la conexión
engine = sqlalchemy.create_engine(connection_string)

# Cargar el archivo CSV generado por el pipeline
df = pd.read_csv('data/fight_stats.csv')

# Subir el DataFrame a la base de datos
try:
    df.to_sql('fight_stats', con=engine, if_exists='replace', index=False)
    print("✅ Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f"❌ Error al cargar datos en la base de datos: {e}")
