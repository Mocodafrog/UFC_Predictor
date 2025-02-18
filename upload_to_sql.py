import pandas as pd
import sqlalchemy
import pyodbc
import os

# 1️⃣ Cargar credenciales desde variables de entorno
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# 2️⃣ Verificar si todas las variables están definidas
if not all([db_user, db_password, db_host, db_port, db_name]):
    raise ValueError(" ERROR: Faltan variables de entorno. Configúralas antes de ejecutar el script.")

# 3️⃣ Crear cadena de conexión para SQL Server
connection_string = (
    f"mssql+pyodbc://{db_user}:{db_password}@{db_host},{db_port}/{db_name}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)

# 4️⃣ Crear la conexión con SQLAlchemy
try:
    engine = sqlalchemy.create_engine(connection_string)
    conn = engine.connect()
    print(" Conexión exitosa a SQL Server.")
except Exception as e:
    print(f" Error de conexión a SQL Server: {e}")
    exit()

# 5️⃣ Cargar el archivo CSV con los datos de peleas
try:
    df = pd.read_csv('data/fight_stats.csv')
    print(f"📊 Datos cargados del CSV con {df.shape[0]} filas y {df.shape[1]} columnas.")
except Exception as e:
    print(f" Error al cargar el CSV: {e}")
    exit()

# 6️⃣ Subir el DataFrame a la base de datos
try:
    df.to_sql('fight_stats', con=engine, if_exists='replace', index=False)
    print(" Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f" Error al cargar datos en la base de datos: {e}")
