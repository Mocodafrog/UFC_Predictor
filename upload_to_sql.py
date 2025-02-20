import os
import pandas as pd
import pyodbc

# 🔹 Configurar conexión a SQL Server en Amazon RDS
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')}"
)

# 🔹 Leer el CSV

df = pd.read_csv(data/fight_stats.csv)

# 🔹 Conectar a la base de datos
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# 🔹 Obtener nombres de columnas automáticamente
columns = df.columns.tolist()
placeholders = ', '.join(['?' for _ in columns])
query = f"INSERT INTO fight_stats ({', '.join(columns)}) VALUES ({placeholders})"

# 🔹 Insertar datos en la tabla
for _, row in df.iterrows():
    cursor.execute(query, tuple(row))

conn.commit()
cursor.close()
conn.close()

print("✅ Datos cargados exitosamente en SQL Server RDS.")

