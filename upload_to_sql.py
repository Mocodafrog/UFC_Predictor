import os
import pandas as pd
from sqlalchemy import create_engine

# Obtener credenciales desde variables de entorno
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# Verificar que las variables están configuradas correctamente
if None in [db_user, db_password, db_host, db_port, db_name]:
    raise ValueError("Faltan variables de entorno para la conexión a la base de datos.")

# Crear la cadena de conexión
connection_string = f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"

# Crear el motor de conexión con SQLAlchemy
try:
    engine = create_engine(connection_string)
    conn = engine.connect()
    print("✅ Conexión exitosa a la base de datos.")
    conn.close()
except Exception as e:
    print(f"❌ Error al conectar a la base de datos: {e}")

# Cargar el archivo CSV generado por el pipeline
df = pd.read_csv('data/fight_stats.csv')

# Subir los datos a SQL Server
try:
    df.to_sql('fight_stats', con=engine, if_exists='replace', index=False)
    print("✅ Datos cargados exitosamente en la base de datos.")
except Exception as e:
    print(f"❌ Error al cargar datos en la base de datos: {e}")
