import pyodbc  # Usa pyodbc para SQL Server
import os
import pandas as pd

# Cargar credenciales desde variables de entorno
server = os.getenv('DB_SERVER')
database = os.getenv('DB_NAME')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

# Cadena de conexión para SQL Server
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password}"
)

try:
    conn = pyodbc.connect(conn_str)
    print("✅ Conexión exitosa a SQL Server en Amazon RDS.")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    exit(1)

# Cargar CSV
csv_path = "data/fight_stats.csv"  # Ajusta la ruta si es necesario
df = pd.read_csv(csv_path)

# Insertar datos en la tabla
cursor = conn.cursor()

columns = df.columns.tolist()
placeholders = ', '.join(['?' for _ in columns])
query = f"INSERT INTO fight_stats ({', '.join(columns)}) VALUES ({placeholders})"

for _, row in df.iterrows():
    cursor.execute(query, tuple(row))

conn.commit()
cursor.close()
conn.close()

print("✅ Datos cargados exitosamente.")
