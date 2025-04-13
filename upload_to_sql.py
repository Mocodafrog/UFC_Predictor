import pymysql
import os
import pandas as pd
import sys

# Leer argumentos: CSV y tabla
csv_path = sys.argv[1]  # por ejemplo: data/fight_stats.csv
table_name = sys.argv[2]  # por ejemplo: fight_stats

# Conexión a MySQL usando variables de entorno
conn = pymysql.connect(
    host=os.environ['DB_SERVER'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD'],
    database=os.environ['DB_NAME'],
    port=3306
)
print("✅ Conexión exitosa a MySQL")

# Cargar el CSV
df = pd.read_csv(csv_path)

# Limpieza de nombres de columnas para que coincidan con la tabla SQL
df.columns = [col.strip().replace(" ", "_").replace(".", "").replace("-", "_") for col in df.columns]

# Preparar query
columns = df.columns.tolist()
placeholders = ', '.join(['%s'] * len(columns))
column_names = ', '.join([f"`{col}`" for col in columns])
query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

# Insertar en lote
cursor = conn.cursor()
try:
    cursor.executemany(query, df.values.tolist())
    conn.commit()
    print(f"✅ {cursor.rowcount} filas insertadas exitosamente en `{table_name}`.")
except Exception as e:
    print(f"❌ Error al insertar datos: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()
