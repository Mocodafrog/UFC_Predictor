import pymysql
import os
import pandas as pd
import sys

# Cargar credenciales desde variables de entorno
host = os.getenv('DB_SERVER')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')
port = 3306

# Leer CSV desde argumentos
csv_path = sys.argv[1]          # data/fight_stats.csv
table_name = sys.argv[2]        # fight_stats

# Segundo CSV (versión raw)
csv_raw_path = "data/fight_stats_raw.csv"
table_raw_name = "fight_stats_raw"

def insert_csv(csv_path, table_name, clean_strings=False):
    print(f"\n Cargando archivo: {csv_path} → tabla: {table_name}")
    
    try:
        df = pd.read_csv(csv_path)

        if clean_strings:
            # Limpiar saltos de línea y espacios extra
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: x.replace('\n', ' ').strip() if isinstance(x, str) else x)

        # Reemplazar NaN por None (valor por valor)
        df = df.where(pd.notnull(df), None)

    except Exception as e:
        print(f" Error al leer {csv_path}: {e}")
        return

    cursor = conn.cursor()
    columns = df.columns.tolist()
    placeholders = ', '.join(['%s' for _ in columns])
    column_names = ', '.join([f"`{col}`" for col in columns])
    query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

    try:
        # DELETE solo si la tabla es la versión principal limpia
    if table_name == "fight_stats":
        cursor.execute(f"DELETE FROM {table_name}")
        print(f" Tabla {table_name} vaciada antes de insertar (con DELETE).")

        for _, row in df.iterrows():
            values = [None if pd.isna(val) else val for val in row]
            cursor.execute(query, values)

        conn.commit()
        print(f"Datos insertados correctamente en: {table_name}")
    except Exception as e:
        print(f" Error al insertar en {table_name}: {e}")
        conn.rollback()
    finally:
        cursor.close()

# Conexión a MySQL
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
    sys.exit(1)

# Insertar archivos
insert_csv(csv_path, table_name, clean_strings=False)        # fight_stats limpio (se trunca)
insert_csv(csv_raw_path, table_raw_name, clean_strings=True) # fight_stats_raw (no se trunca)

# Cerrar conexión
conn.close()
