import pyodbc
import os

# Cargar credenciales desde variables de entorno
server = os.getenv('DB_HOST')  # Endpoint de Amazon RDS
database = os.getenv('DB_NAME')  # Nombre de la base de datos en SQL Server
username = os.getenv('DB_USER')  # Usuario de SQL Server
password = os.getenv('DB_PASSWORD')  # Contraseña

# Cadena de conexión para SQL Server
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

try:
    conn = pyodbc.connect(conn_str)
    print("✅ Conexión exitosa a SQL Server en Amazon RDS.")
    conn.close()
except Exception as e:
    print(f"❌ Error de conexión: {e}")
