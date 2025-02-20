import pyodbc
import os

# 1️⃣ Cargar credenciales desde variables de entorno
server = os.getenv('DB_HOST')  # Host de Amazon RDS
database = os.getenv('DB_NAME')  # Base de datos (asegúrate de que existe)
username = os.getenv('DB_USER')  # Usuario
password = os.getenv('DB_PASSWORD')  # Contraseña

# 2️⃣ Cadena de conexión
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

# 3️⃣ Intentar la conexión
try:
    conn = pyodbc.connect(conn_str)
    print("✅ Conexión exitosa a SQL Server en Amazon RDS.")
    conn.close()
except Exception as e:
    print(f"❌ Error de conexión: {e}")
