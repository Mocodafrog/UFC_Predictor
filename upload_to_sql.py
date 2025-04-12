import pymysql
import os
import pandas as pd

# Cargar credenciales desde variables de entorno
host = os.getenv('DB_SERVER')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')
port = 3306  # Puedes cambiarlo si tu RDS usa otro puerto

# Leer CSV (usando la ruta desde argumentos)
import sys
csv_path = sys.argv[1]  # data/fight_stats.csv
table_name = sys.argv[2]  # fight_stats

# Cargar el archivo CSV
df = pd.read_csv(csv_path)

# Crear conexión a MySQL
try:
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
    print(" Conexión exitosa a MySQL")
except Exception as e:
    print(f" Error al conectar a MySQL: {e}")
    exit(1)

# Insertar datos
cursor = conn.cursor()

columns = df.columns.tolist()
placeholders = ', '.join(['%s' for _ in columns])
column_names = ', '.join([f"`{col}`" for col in columns])  # usar backticks por si hay espacios

query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

try:
    for _, row in df.iterrows():
        cursor.execute(query, tuple(row))
    conn.commit()
    print(" Datos insertados correctamente.")
except Exception as e:
    print(f" Error al insertar datos: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()
