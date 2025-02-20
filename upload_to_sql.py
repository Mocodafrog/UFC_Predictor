import pandas as pd
import os
import sqlalchemy
from sqlalchemy import create_engine

# 1️⃣ Cargar credenciales desde variables de entorno
db_user = os.getenv('DB_USER')  # Usuario de la base de datos
db_password = os.getenv('DB_PASSWORD')  # Contraseña
db_host = os.getenv('DB_HOST')  # Host del servidor
db_port = os.getenv('DB_PORT')  # Puerto (1433 por defecto en SQL Server)
db_name = os.getenv('DB_NAME')  # Nombre de la base de datos

# 2️⃣ Crear cadena de conexión con SQLAlchemy
connection_string = f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)

# 3️⃣ Cargar el archivo CSV
csv_file = "data/fight_stats.csv"
df = pd.read_csv(csv_file)

# 4️⃣ Subir los datos a SQL Server
try:
    df.to_sql("fight_stats", con=engine, if_exists="replace", index=False)
    print("✅ Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f"❌ Error al cargar datos: {e}")
