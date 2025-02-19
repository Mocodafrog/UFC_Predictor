import pandas as pd
import sqlalchemy
import os

# 1️⃣ Cargar credenciales desde variables de entorno
db_user = os.getenv('DB_USER')  # Usuario de la base de datos
db_password = os.getenv('DB_PASSWORD')  # Contraseña
db_host = os.getenv('DB_HOST')  # Host del servidor
db_port = os.getenv('DB_PORT')  # Puerto de conexión (10047)
db_name = os.getenv('DB_NAME')  # Nombre de la base de datos

# 2️⃣ Crear cadena de conexión para SQL Server (NO MySQL)
connection_string = (
    f"mssql+pyodbc://{db_user}:{db_password}@{db_host},{db_port}/{db_name}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)

# 3️⃣ Conectar a la base de datos
try:
    engine = sqlalchemy.create_engine(connection_string)
    conn = engine.connect()
    print("✅ Conexión exitosa a SQL Server.")
except Exception as e:
    print(f"❌ Error de conexión a SQL Server: {e}")
    exit()

# 4️⃣ Cargar el CSV
try:
    df = pd.read_csv('data/fight_stats.csv')
    print(f"📊 CSV cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
except Exception as e:
    print(f"❌ Error al leer el CSV: {e}")
    exit()

# 5️⃣ Subir datos a SQL Server
try:
    df.to_sql('fight_stats', con=engine, if_exists='replace', index=False)
    print("✅ Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f"❌ Error al cargar datos en la base de datos: {e}")
