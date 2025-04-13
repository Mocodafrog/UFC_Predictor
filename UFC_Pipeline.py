import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re 
import numpy as np
from scipy.stats import mstats
import openpyxl 
from sklearn.preprocessing import LabelEncoder

import os




# Inicializar listas para almacenar los datos de todos los peleadores
full_names = []
heights = []
weights = []
reaches = []
stances = []
wins = []
losses = []
draws = []
birthdates = []  # Nueva lista para almacenar las fechas de nacimiento

# Iterar sobre cada letra del alfabeto
for char in 'abcdefghijklmnopqrstuvwxyz':
    # URL de la página de peleadores para cada letra
    url = f"http://ufcstats.com/statistics/fighters?char={char}&page=all"
    
    # Realizar la solicitud a la página
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Seleccionar la tabla con los datos de los peleadores
    table = soup.select_one('table')
    
    # Verificar si la tabla existe en la página
    if table:
        # Iterar sobre cada fila (excluyendo la primera fila que es el encabezado)
        for row in table.select('tr:nth-of-type(n+2)'):
            columns = row.find_all('td')
            
            # Obtener el enlace al perfil del peleador
            last_name_column = columns[1].find('a')
            if last_name_column:
                fighter_profile_url = last_name_column['href']
                
                # Hacer una solicitud a la página del perfil del peleador
                profile_response = requests.get(fighter_profile_url)
                profile_soup = BeautifulSoup(profile_response.content, 'html.parser')
                
                # Extraer el nombre completo desde el perfil
                full_name_tag = profile_soup.select_one('span.b-content__title-highlight')
                full_name = full_name_tag.text.strip() if full_name_tag else 'N/A'
                
                # Extraer la fecha de nacimiento (quinto elemento de la lista)
                birthdate_tag = profile_soup.select_one('.b-list__info-box_style_small-width li:nth-of-type(5)')
                birthdate = birthdate_tag.text.strip() if birthdate_tag else 'N/A'
                
                # Agregar el nombre completo, fecha de nacimiento y otros detalles a las listas
                full_names.append(full_name)
                birthdates.append(birthdate)
                heights.append(columns[3].text.strip())      # Ht. (Height)
                weights.append(columns[4].text.strip())      # Wt. (Weight)
                reaches.append(columns[5].text.strip())      # Reach
                stances.append(columns[6].text.strip())      # Stance
                wins.append(columns[7].text.strip())         # W (Wins)
                losses.append(columns[8].text.strip())       # L (Losses)
                draws.append(columns[9].text.strip())        # D (Draws)
                
                # Agregar un pequeño retraso entre solicitudes para evitar sobrecargar el servidor
                time.sleep(1)

# Crear un DataFrame con los datos extraídos de todas las letras
fighters_df = pd.DataFrame({
    'full_name': full_names,
    'birthdate': birthdates,  # Incluir la fecha de nacimiento
    'height': heights,
    'weight': weights,
    'reach': reaches,
    'stance': stances,
    'wins': wins,
    'losses': losses,
    'draws': draws
})


# URL de la página de eventos completados
base_url = "http://ufcstats.com/statistics/events/completed?page=all"
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Listas para almacenar datos
event_names = []
fight_keys = []
fighter_names = []
winner_flags = []
stats = []  # Lista para almacenar todas las estadísticas de las peleas

# Extraer todos los enlaces de los eventos sin limitación
event_links = [a['href'] for a in soup.select('a.b-link.b-link_style_black')]

# Recorrer cada evento
for event_link in event_links:
    event_response = requests.get(event_link)
    soup_event = BeautifulSoup(event_response.content, 'html.parser')

    # Extraer todos los enlaces de las peleas sin limitación
    #fight_links = [a['href'] for a in soup_event.select('a.b-flag.b-flag_style_green')]
    fight_links = list(set([a['href'] for a in soup_event.select('a.b-flag.b-flag_style_green, a.b-flag.b-flag_style_bordered')]))

    for fight_link in fight_links:
        try:
            time.sleep(1)
            fight_response = requests.get(fight_link, timeout=10)
            soup_fight = BeautifulSoup(fight_response.content, 'html.parser')

            # Extraer el nombre del evento
            event_name_elem = soup_fight.select_one('h2.b-content__title')
            event_name = event_name_elem.text.strip() if event_name_elem else "N/A"

            # Extraer nombres de los peleadores
            fighter_1_name = soup_fight.select('p.b-fight-details__table-text a')[0].text.strip()
            fighter_2_name = soup_fight.select('p.b-fight-details__table-text a')[1].text.strip()
            fighters_info = soup_fight.select('.b-fight-details__persons .b-fight-details__person')

            fighter_1_result = fighters_info[0].select_one('i.b-fight-details__person-status').text.strip()
            fighter_2_result = fighters_info[1].select_one('i.b-fight-details__person-status').text.strip()

            # Extraer el título de la pelea (Weight Class)
            fight_title_elem = soup_fight.select_one('.b-fight-details__fight-title')
            weight_class = fight_title_elem.get_text(strip=True) if fight_title_elem else "N/A"
            # Crear la clave de pelea (Fight Key)
            fight_key = f"{fighter_1_name} - {fighter_2_name}"

            # Extraer las tablas de estadísticas generales ("Totals") y significant strikes
            tables = soup_fight.select('.b-fight-details__table-body')

            # **IMPORTANTE**: Asegurar que estamos seleccionando las tablas correctas por peleador
            # Totals (KD, Sig. Str., Total Str., TD, etc.)
            totals_table = tables[:2]  # Las primeras dos tablas son para Totals
            fighter_1_stats = [row.text.strip() for row in totals_table[0].select('p.b-fight-details__table-text')]
            fighter_2_stats = [row.text.strip() for row in totals_table[0].select('p.b-fight-details__table-text')]

            # Significant strikes (se encuentran al final)
            sig_strikes_table = tables[-2:]  # Las dos últimas tablas son para significant strikes
            fighter_1_striking_stats = [row.text.strip() for row in sig_strikes_table[0].select('p.b-fight-details__table-text')]
            fighter_2_striking_stats = [row.text.strip() for row in sig_strikes_table[0].select('p.b-fight-details__table-text')]

            # Extraer 'method', 'rounds', 'format', y 'referee'
            method = soup_fight.select_one('.b-fight-details__text-item_first i:nth-of-type(2)').text.strip() if soup_fight.select_one('.b-fight-details__text-item_first i:nth-of-type(2)') else "N/A"
            rounds = soup_fight.select_one('i.b-fight-details__text-item:nth-of-type(2)').text.strip() if soup_fight.select_one('i.b-fight-details__text-item:nth-of-type(2)') else "N/A"
            length = soup_fight.select_one('i.b-fight-details__text-item:nth-of-type(3)').text.strip() if soup_fight.select_one('i.b-fight-details__text-item:nth-of-type(3)') else "N/A"
            fight_format = soup_fight.select_one('i:nth-of-type(4)').text.strip() if soup_fight.select_one('i:nth-of-type(4)') else "N/A"
            referee = soup_fight.select_one('span').text.strip() if soup_fight.select_one('span') else "N/A"
            # Agregar todos los datos en un diccionario para cada peleador
            stats.append({
                'Event': event_name,
                'Fight': fight_key,
                'Fighter': fighter_1_name,
                'Weight Class': weight_class,
                'Winner': fighter_1_result, 
                'KD': fighter_1_stats[2],
                'Sig. Str.': fighter_1_stats[4],
                'Total Str.': fighter_1_stats[8],
                'TD': fighter_1_stats[10],
                'Sub. Att': fighter_1_stats[14],
                'Reversal': fighter_1_stats[16],
                'Control Time': fighter_1_stats[18],
                'Head': fighter_1_striking_stats[6],
                'Body': fighter_1_striking_stats[8],
                'Leg': fighter_1_striking_stats[10],
                'Distance': fighter_1_striking_stats[12],
                'Clinch': fighter_1_striking_stats[14],
                'Ground': fighter_1_striking_stats[16],
                'Method': method,
                'Fight_lenght': length,
                'Rounds': rounds,
                'Format': fight_format,
                'Referee': referee
            })

            # Ajuste importante para asegurarnos de que Fighter 2 está correctamente alineado con las tablas adecuadas
            stats.append({
                'Event': event_name,
                'Fight': fight_key,
                'Fighter': fighter_2_name,
                'Weight Class': weight_class,
                'Winner': fighter_2_result,
                'KD': fighter_2_stats[3],
                'Sig. Str.': fighter_2_stats[5],
                'Total Str.': fighter_2_stats[9],
                'TD': fighter_2_stats[11],
                'Sub. Att': fighter_2_stats[15],
                'Reversal': fighter_2_stats[17],
                'Control Time': fighter_2_stats[19],
                'Head': fighter_2_striking_stats[7],
                'Body': fighter_2_striking_stats[9],
                'Leg': fighter_2_striking_stats[11],
                'Distance': fighter_2_striking_stats[13],
                'Clinch': fighter_2_striking_stats[15],
                'Ground': fighter_2_striking_stats[17],
                'Method': method,
                'Fight_lenght': length,
                'Rounds': rounds,
                'Format': fight_format,
                'Referee': referee
            })

        except Exception as e:
            print(f"Error procesando la pelea {fight_link}: {e}")

# Convertir los datos en un DataFrame
df = pd.DataFrame(stats)

# URL de la página de eventos completados
base_url = "http://ufcstats.com/statistics/events/completed?page=all"
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Listas para almacenar los datos
event_names = []
event_dates = []
event_locations = []

# Seleccionar la tabla de eventos
event_table = soup.select('table.b-statistics__table-events tbody tr')
# Recorrer cada fila de la tabla
for row in event_table:
    try:
        # Extraer el nombre del evento
        event_name = row.select_one('a.b-link.b-link_style_black').text.strip()
        event_names.append(event_name)

        # Extraer la fecha del evento
        event_date = row.select_one('span.b-statistics__date').text.strip()
        event_dates.append(event_date)

        # Extraer la ubicación del evento
        event_location = row.select_one('td.b-statistics__table-col.b-statistics__table-col_style_big-top-padding').text.strip()
        event_locations.append(event_location)
    except Exception as e:
        print(f"Error procesando una fila: {e}")

# Crear un DataFrame con los datos
events = pd.DataFrame({
    'Event Name': event_names,
    'Date': event_dates,
    'Location': event_locations
    }
)

# URL de la página de eventos futuros
base_url = "http://ufcstats.com/statistics/events/upcoming?page=all"
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Listas para almacenar los datos
event_names = []
event_dates = []
event_locations = []

# Seleccionar la tabla de eventos
event_table = soup.select('table.b-statistics__table-events tbody tr')

# Recorrer cada fila de la tabla
for row in event_table:
    try:
        # Extraer el nombre del evento
        event_name = row.select_one('a.b-link.b-link_style_black').text.strip()
        event_names.append(event_name)

        # Extraer la fecha del evento
        event_date = row.select_one('span.b-statistics__date').text.strip()
        event_dates.append(event_date)

        # Extraer la ubicación del evento
        event_location = row.select_one('td.b-statistics__table-col.b-statistics__table-col_style_big-top-padding').text.strip()
        event_locations.append(event_location)
    except Exception as e:
        print(f"Error procesando una fila: {e}")

# Crear un DataFrame con los datos
upcoming_envents = pd.DataFrame({
    'Event Name': event_names,
    'Date': event_dates,
    'Location': event_locations
})


# URL de la página de eventos futuros
base_url = "http://ufcstats.com/statistics/events/upcoming?page=all"
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Listas para almacenar datos
events_data = []

# Extraer los enlaces de los eventos futuros
event_links = [a['href'] for a in soup.select('a.b-link.b-link_style_black')]

# Recorrer cada evento
for event_link in event_links:
    try:
        time.sleep(1)
        event_response = requests.get(event_link)
        soup_event = BeautifulSoup(event_response.content, 'html.parser')

        # Extraer el nombre del evento
        event_name_elem = soup_event.select_one('h2.b-content__title')
        event_name = event_name_elem.text.strip() if event_name_elem else "N/A"

        # Extraer la fecha y ubicación del evento
        event_info = soup_event.select_one('ul.b-list__box-list')
        event_date = event_info.select('li')[0].text.replace('Date:', '').strip() if event_info else "N/A"
        event_location = event_info.select('li')[1].text.replace('Location:', '').strip() if event_info else "N/A"

        # Extraer la tabla que contiene los datos de las peleas y la clase de peso
        fight_table = soup_event.select_one('table.b-fight-details__table.b-fight-details__table_style_margin-top.b-fight-details__table_type_event-details')

        if fight_table:
            rows = fight_table.select('tr.b-fight-details__table-row')
            for row in rows:
                try:
                    # Extraer los peleadores
                    fighters = row.select('p.b-fight-details__table-text a')
                    if len(fighters) >= 2:
                        fighter_1 = fighters[0].text.strip()
                        fighter_2 = fighters[1].text.strip()
                    else:
                        fighter_1 = fighter_2 = "N/A"
                    
                    # Crear la combinación de peleadores
                    fight_pair = f"{fighter_1} vs. {fighter_2}"

                    # Extraer la clase de peso desde el selector que proporcionaste
                    weight_class_elem = row.select_one('td:nth-of-type(7) p')
                    weight_class = weight_class_elem.text.strip() if weight_class_elem else "N/A"

                    # Añadir los datos a la lista
                    if fighter_1 != "N/A" and fighter_2 != "N/A":
                        events_data.append({
                            'Event Name': event_name,
                            'Date': event_date,
                            'Location': event_location,
                            'Fight': fight_pair,
                            'Fighter 1': fighter_1,
                            'Fighter 2': fighter_2,
                            'Weight Class': weight_class
                        })
                except Exception as e:
                    print(f"Error procesando una pelea: {e}")
        else:
            print(f"No se encontró la tabla de peleas para el evento {event_name}")

    except Exception as e:
        print(f"Error procesando un evento: {e}")

# Convertir los datos en un DataFrame
upcoming_fights = pd.DataFrame(events_data)


# Identificar los nombres duplicados
duplicated_fighters = fighters_df[fighters_df.duplicated(subset=['full_name'], keep=False)]

# Eliminar todos los duplicados (tanto originales como duplicados)
fighters_df = fighters_df[~fighters_df['full_name'].isin(duplicated_fighters['full_name'])]

# Generar una clave única numérica para cada luchador basada en su nombre completo
fighters_df['fighter_id'] = pd.factorize(fighters_df['full_name'])[0] + 1  # Sumar 1 para evitar IDs en 0

# Función para convertir altura de pies y pulgadas a centímetros
def convert_height_to_cm(height):
    if height != '--':
        feet, inches = height.split("' ")
        inches = inches.replace('\"', '')
        return int(feet) * 30.48 + int(inches) * 2.54
    else:
        return None  # Si no hay datos, dejamos como None

# Función para convertir peso de libras a kilogramos
def convert_weight_to_kg(weight):
    if weight != '--':
        return float(weight.replace(' lbs.', '')) * 0.453592
    else:
        return None  # Si no hay datos, dejamos como None

# Función para convertir alcance de pulgadas a centímetros
def convert_reach_to_cm(reach):
    if reach != '--':
        return float(reach.replace('\"', '')) * 2.54
    else:
        return None  # Si no hay datos, dejamos como None

# Aplicar las funciones de conversión
fighters_df['height_cm'] = fighters_df['height'].apply(convert_height_to_cm)
fighters_df['weight_kg'] = fighters_df['weight'].apply(convert_weight_to_kg)
fighters_df['reach_cm'] = fighters_df['reach'].apply(convert_reach_to_cm)

# Reemplazar los valores NaN por ceros para las columnas de altura, peso y alcance
fighters_df['height_cm'].fillna(0, inplace=True)
fighters_df['weight_kg'].fillna(0, inplace=True)
fighters_df['reach_cm'].fillna(0, inplace=True)
fighters_df['birthdate'] = fighters_df['birthdate'].apply(lambda x: re.search(r'\w{3} \d{2}, \d{4}', x).group(0) if re.search(r'\w{3} \d{2}, \d{4}', x) else 'N/A')


# Reordenar el DataFrame con 'fighter_id', 'full_name', seguido de las medidas y luego victorias, derrotas y empates
fighters_df = fighters_df[['fighter_id', 'full_name', 'height_cm', 'weight_kg', 'reach_cm', 'stance', 'wins', 'losses', 'draws','birthdate']]

# Crear una copia del DataFrame original para mantenerlo intacto
fight_stats = df.copy()

events.rename(columns={'Event Name': 'Event'}, inplace=True)




# Convertir la columna 'Winner' a valores booleanos en el nuevo DataFrame
#fight_stats['Winner'] = fight_stats['Winner'].apply(lambda x: 1 if x == 'W' else 0)
fight_stats = fight_stats[fight_stats['Winner'] != 'NC']
fight_stats = fight_stats[fight_stats['Winner'] != 'D']
fight_stats = fight_stats[fight_stats['Method'] != 'DQ']


# Crear una nueva columna para almacenar la forma de los últimos 5 encuentros antes de la pelea actual
# Diccionario actualizado de normalización
category_normalization = {
    # Pesos estándar
    'Bantamweight Bout': 'Bantamweight',
    'Catch Weight Bout': 'Catchweight',
    'Featherweight Bout': 'Featherweight',
    'Flyweight Bout': 'Flyweight',
    'Heavyweight Bout': 'Heavyweight',
    'Light Heavyweight Bout': 'Light Heavyweight',
    'Lightweight Bout': 'Lightweight',
    'Middleweight Bout': 'Middleweight',
    'Welterweight Bout': 'Welterweight',
    'Women\'s Bantamweight Bout': 'Women\'s Bantamweight',
    'Women\'s Featherweight Bout': 'Women\'s Featherweight',
    'Women\'s Flyweight Bout': 'Women\'s Flyweight',
    'Women\'s Strawweight Bout': 'Women\'s Strawweight',
    
    'UFC Bantamweight Title Bout': 'Bantamweight',
    'UFC Featherweight Title Bout': 'Featherweight',
    'UFC Flyweight Title Bout': 'Flyweight',
    'UFC Heavyweight Title Bout': 'Heavyweight',
    'UFC Light Heavyweight Title Bout': 'Light Heavyweight',
    'UFC Lightweight Title Bout': 'Lightweight',
    'UFC Middleweight Title Bout': 'Middleweight',
    'UFC Welterweight Title Bout': 'Welterweight',
    'UFC Women\'s Bantamweight Title Bout': 'Women\'s Bantamweight',
    'UFC Women\'s Featherweight Title Bout': 'Women\'s Featherweight',
    'UFC Women\'s Flyweight Title Bout': 'Women\'s Flyweight',
    'UFC Women\'s Strawweight Title Bout': 'Women\'s Strawweight',

    # Títulos Interinos
    'UFC Interim Bantamweight Title Bout': 'Bantamweight',
    'UFC Interim Featherweight Title Bout': 'Featherweight',
    'UFC Interim Flyweight Title Bout': 'Flyweight',
    'UFC Interim Heavyweight Title Bout': 'Heavyweight',
    'UFC Interim Light Heavyweight Title Bout': 'Light Heavyweight',
    'UFC Interim Lightweight Title Bout': 'Lightweight',
    'UFC Interim Middleweight Title Bout': 'Middleweight',
    'UFC Interim Welterweight Title Bout': 'Welterweight',

    # Torneos UFC y Ultimate Fighter
    'Road To UFC 1 Bantamweight Tournament Title Bout': 'Bantamweight',
    'Road To UFC 1 Featherweight Tournament Title Bout': 'Featherweight',
    'Road To UFC 1 Lightweight Tournament Title Bout': 'Lightweight',
    'Road to UFC 1 Flyweight Tournament Title Bout': 'Flyweight',
    'TUF Nations Canada vs. Australia Middleweight Tournament Title Bout': 'Middleweight',
    'TUF Nations Canada vs. Australia Welterweight Tournament Title Bout': 'Welterweight',

    
    # Ultimate Fighter Torneos
    'Ultimate Fighter 2 Welterweight Tournament Title Bout' : 'Welterweight',
    'Ultimate Fighter 1 Light Heavyweight Tournament Title Bout': 'Light Heavyweight',
    'Ultimate Fighter 1 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 10 Heavyweight Tournament Title Bout': 'Heavyweight',
    'Ultimate Fighter 11 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 12 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 13 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter 14 Bantamweight Tournament Title Bout': 'Bantamweight',
    'Ultimate Fighter 14 Featherweight Tournament Title Bout': 'Featherweight',
    'Ultimate Fighter 15 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 16 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter 17 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 19 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 22 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 23 Light Heavyweight Tournament Title Bout': 'Light Heavyweight',
    'Ultimate Fighter 23 Women\'s Strawweight Tournament Title Bout': 'Women\'s Strawweight',
    'Ultimate Fighter 25 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter 27 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 28 Heavyweight Tournament Title Bout': 'Heavyweight',
    'Ultimate Fighter 28 Women\'s Featherweight Tournament Title Bout': 'Women\'s Featherweight',
    'Ultimate Fighter 3 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 4 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 4 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter 5 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 6 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter 7 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter 8 Light Heavyweight Tournament Title Bout': 'Light Heavyweight',
    'Ultimate Fighter 8 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 9 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter 9 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter Australia vs. UK Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter Brazil 1 Featherweight Tournament Title Bout': 'Featherweight',
    'Ultimate Fighter Brazil 1 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter Brazil 2 Welterweight Tournament Title Bout': 'Welterweight',
    'Ultimate Fighter Brazil 3 Heavyweight Tournament Title Bout': 'Heavyweight',
    'Ultimate Fighter Brazil 3 Middleweight Tournament Title Bout': 'Middleweight',
    'Ultimate Fighter Latin America 3 Lightweight Tournament Title Bout': 'Lightweight',
    'Ultimate Fighter Latin America Bantamweight Tournament Title Bout': 'Bantamweight',
}

# Aplicamos la normalización al DataFrame
fight_stats['Weight Class'] = fight_stats['Weight Class'].replace(category_normalization)


category_normalization_2 = {
    'Decision - Majority':'Decision',
    'Decision - Split':'Decision',
    'Decision - Unanimous':'Decision',
    'TKO - Doctor\'s Stoppage':'KO/TKO'
}
fight_stats['Method'] = fight_stats['Method'].replace(category_normalization_2)
fight_stats['Method'].unique
# Función para generar la secuencia de las 5 peleas siguientes
def get_next_5_results(group):
    results = group['Winner'].apply(lambda x: 'W' if x == 'W' else ('L' if x == 'L' else 'D')).tolist()
    next_5_results = []
    
    # Recorremos el grupo y generamos las secuencias de las siguientes 5 peleas después de la pelea actual
    for i in range(len(results)):
        # Tomar los 5 resultados siguientes a la pelea actual (i.e., del futuro)
        next_5_str = ''.join(results[i+1:i+6])  # Tomamos las siguientes 5 peleas
        next_5_results.append(next_5_str)
    
    return pd.Series(next_5_results, index=group.index)

# Aplicar la función agrupando por peleador (sin modificar el orden)
fight_stats['form_last_5'] = fight_stats.groupby('Fighter', group_keys=False).apply(get_next_5_results)




# Función para separar los valores 'X of Y'
def split_landed_attempted(col):
    # Manejar casos donde el formato 'X of Y' no está presente
    return col.str.split(' of ', expand=True).fillna(0).astype(float)

# Aplicar la función de separación a las columnas relevantes en el nuevo DataFrame, con manejo de errores
columns_to_split = ['Sig. Str.', 'Total Str.', 'TD', 'Head', 'Body', 'Leg', 'Distance', 'Clinch', 'Ground']

for col in columns_to_split:
    if col in fight_stats.columns:
        try:
            fight_stats[[f'landed_{col.lower()}', f'atmp_{col.lower()}']] = split_landed_attempted(fight_stats[col])
        except Exception as e:
            print(f"Error procesando la columna {col}: {e}")

# Eliminar las columnas originales que fueron transformadas
fight_stats.drop(columns=columns_to_split, inplace=True)

fight_stats['Rounds'] = fight_stats['Rounds'].str.extract('(\\d+)')[0].astype(int)

fight_stats['Format'] = fight_stats['Format'].str.extract('(\\d+ Rnd.*\\([\\d-]+\\)|no time limit)', flags=re.IGNORECASE)
# Mostrar el DataFrame normalizado
fight_stats['Control Time'] = fight_stats['Control Time'].str.extract(r'(\d+:\d+)$')
fight_stats['Fight_lenght'] = fight_stats['Fight_lenght'].str.extract(r'(\d+:\d+)$')
fight_stats['Control Time'] = '00:' + fight_stats['Control Time'].fillna('00:00')
fight_stats['Fight_lenght'] = '00:' + fight_stats['Fight_lenght'].fillna('00:00')

fight_stats['Control Time Sec'] = pd.to_timedelta(fight_stats['Control Time'], errors='raise').dt.total_seconds()
fight_stats['Fight_lenght Sec'] = pd.to_timedelta(fight_stats['Fight_lenght'], errors='raise').dt.total_seconds()
fight_stats['Fight_lenght Sec'] = pd.to_timedelta(fight_stats['Fight_lenght'], errors='raise').dt.total_seconds()

fight_stats = fight_stats.merge(events, on='Event', how='left')
fight_stats['Date'] = pd.to_datetime(fight_stats['Date'], format='%B %d, %Y')
cutoff_date = pd.Timestamp('1999-05-07')
fight_stats = fight_stats[fight_stats['Date'] > cutoff_date]



fight_stats['Total_fight_length_sec'] = (fight_stats['Rounds'] - 1) * 300 + fight_stats['Fight_lenght Sec']






# Golpes significativos
fight_stats['Sig_Str_Acc'] = fight_stats['landed_sig. str.'] / fight_stats['atmp_sig. str.'].replace(0, 1)  # Precisión golpes significativos
fight_stats['Sig_Str_LpM'] = fight_stats['landed_sig. str.'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes totales
fight_stats['Total_Str_Acc'] = fight_stats['landed_total str.'] / fight_stats['atmp_total str.'].replace(0, 1)  # Precisión golpes totales
fight_stats['Total_Str_LpM'] = fight_stats['landed_total str.'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes a la cabeza
fight_stats['Head_Str_Acc'] = fight_stats['landed_head'] / fight_stats['atmp_head'].replace(0, 1)  # Precisión golpes a la cabeza
fight_stats['Head_Str_LpM'] = fight_stats['landed_head'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes al cuerpo
fight_stats['Body_Str_Acc'] = fight_stats['landed_body'] / fight_stats['atmp_body'].replace(0, 1)  # Precisión golpes al cuerpo
fight_stats['Body_Str_LpM'] = fight_stats['landed_body'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes a las piernas
fight_stats['Leg_Str_Acc'] = fight_stats['landed_leg'] / fight_stats['atmp_leg'].replace(0, 1)  # Precisión golpes a las piernas
fight_stats['Leg_Str_LpM'] = fight_stats['landed_leg'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes a distancia
fight_stats['Distance_Str_Acc'] = fight_stats['landed_distance'] / fight_stats['atmp_distance'].replace(0, 1)  # Precisión golpes a distancia
fight_stats['Distance_Str_LpM'] = fight_stats['landed_distance'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes en el clinch
fight_stats['Clinch_Str_Acc'] = fight_stats['landed_clinch'] / fight_stats['atmp_clinch'].replace(0, 1)  # Precisión golpes en el clinch
fight_stats['Clinch_Str_LpM'] = fight_stats['landed_clinch'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

# Golpes en el suelo (ground)
fight_stats['Ground_Str_Acc'] = fight_stats['landed_ground'] / fight_stats['atmp_ground'].replace(0, 1)  # Precisión golpes en el suelo
fight_stats['Ground_Str_LpM'] = fight_stats['landed_ground'] / (fight_stats['Total_fight_length_sec'] / 60)  # Aterrizados por minuto

fight_stats['TD_Avg.'] = (fight_stats['landed_td'] / (fight_stats['Total_fight_length_sec'] / 60))
fight_stats['TD_Acc.'] = fight_stats['landed_td'] / fight_stats['atmp_td'].replace(0, 1)
fight_stats['TD_Def.'] = 1 - (fight_stats.groupby('Fight')['landed_td'].transform(lambda x: x.shift()).fillna(0) / fight_stats['atmp_td'].replace(0, 1))
fight_stats['Control_Ratio'] = fight_stats['Control Time Sec'] / fight_stats['Total_fight_length_sec'].replace(0, 1)

# Renombrar columnas en fighters_df para asegurar consistencia
fighters_df.rename(columns={
    'full_name': 'Fighter',  # Asegurarse de que el nombre del luchador en fighters_df coincide con fight_stats
    'wins': 'total_wins',
    'losses': 'total_losses',
    'draws': 'total_draws'
}, inplace=True)

fight_stats = fight_stats.merge(fighters_df, on='Fighter', how='left')
fight_stats['birthdate'] = pd.to_datetime(fight_stats['birthdate'], format="%b %d, %Y", errors='coerce')



# Paso 1: Crear una llave única combinando 'Event', 'Fight' y 'Fighter'
fight_stats['fight_key'] = fight_stats['Event'] + ' - ' + fight_stats['Fight'] + ' - ' + fight_stats['Fighter']

# Paso 2: Identificar todas las llaves donde 'fighter_id' sea NaN
keys_to_remove_fighter_id = fight_stats['fight_key'][fight_stats['fighter_id'].isna()].unique()

# Paso 3: Identificar todas las llaves donde height_cm, weight_kg, reach_cm o stance sean cero o NaN
keys_to_remove_measurements = fight_stats['fight_key'][
    (fight_stats['height_cm'] == 0) | 
    (fight_stats['weight_kg'] == 0) | 
    (fight_stats['reach_cm'] == 0) | 
    (fight_stats['stance'].isna()) |
    (fight_stats['stance'] == '')  |
    (fight_stats['birthdate'] == '')
].unique()

# Combinar ambas listas de llaves a eliminar
keys_to_remove = set(keys_to_remove_fighter_id).union(set(keys_to_remove_measurements))

# Paso 4: Eliminar todas las filas que tengan esas llaves
fight_stats = fight_stats[~fight_stats['fight_key'].isin(keys_to_remove)]

fight_stats['age_at_fight_days'] = (fight_stats['Date'] - fight_stats['birthdate']).dt.days
columns_to_drop = ['Fight_lenght', 'Location', 'total_wins', 'total_losses', 'total_draws', 'fight_key','birthdate' ]
fight_stats.drop(columns=columns_to_drop, inplace=True)


#fight_stats = fight_stats.groupby(['Event', 'Fight']).apply(
#    lambda x: pd.concat(
#        [x.iloc[0].add_suffix('_figther_1'), x.iloc[1].add_suffix('_figther_2')]
#    ) if len(x) == 2 else None  # Aseguramos que solo opere si hay exactamente dos peleadores
#)

# Eliminamos filas que son None
fight_stats = fight_stats.dropna()
fight_stats = fight_stats.sort_values(by='Date', ascending=False)

# Reiniciamos el índice para evitar problemas con índices agrupados
fight_stats = fight_stats.reset_index(drop=True)

# Eliminar solo las columnas innecesarias y mantener Event y Fight
columns_to_drop = ['Date', 'Referee', 'fighter_id',  'weight_kg']
fight_stats = fight_stats.drop(columns=columns_to_drop)



# Eliminar columnas redundantes, dejando solo una versión de cada atributo
#fight_stats = fight_stats.drop(columns=[
#    'Event_figther_2', 'Fight_figther_2',  'Weight Class_figther_2',
#    'Method_figther_2', 'Format_figther_2'
#])

# Unir los valores de Event, Fight, Winner, etc. en una sola columna
#fight_stats['Event'] = fight_stats['Event_figther_1']
#fight_stats['Fight'] = fight_stats['Fight_figther_1']
#fight_stats['Weight Class'] = fight_stats['Weight Class_figther_1']
#fight_stats['Method'] = fight_stats['Method_figther_1']
#fight_stats['Format'] = fight_stats['Format_figther_1']

# Eliminar las columnas originales ahora redundantes
#fight_stats = fight_stats.drop(columns=[
#    'Event_figther_1', 'Fight_figther_1', 'Weight Class_figther_1',
#    'Method_figther_1', 'Format_figther_1'
#])

# Creamos un LabelEncoder para cada columna categórica
label_encoder = LabelEncoder()
category_mappings = {}
# Lista de columnas que serán codificadas
columns_to_encode = ['Method', 'Winner', 'form_last_5','Format', 'stance','Weight Class']
# Aplicamos LabelEncoder a cada columna y guardamos los mapeos
for col in columns_to_encode:
    le = LabelEncoder()
    fight_stats[col] = le.fit_transform(fight_stats[col].astype(str))
    # Guardamos los mapeos en un diccionario: categoría -> número asignado
    category_mappings[col] = pd.DataFrame({'Category': le.classes_, 'Encoded_Value': le.transform(le.classes_)})

# Aplicamos LabelEncoder a cada columna
for col in columns_to_encode:
    fight_stats[col] = label_encoder.fit_transform(fight_stats[col])




# Reordenar las columnas para que las principales estén al inicio
cols_order = ['Event', 'Fight', 'Weight Class', 'Method', 'Format'] + \
             [col for col in fight_stats.columns if col not in ['Event', 'Fight', 'Weight Class', 'Method', 'Format']]


# Reemplazar valores no numéricos y convertir a enteros
columns_to_convert = ['KD', 'Sub. Att', 'Reversal',]

# Convertir las columnas a tipo numérico, manejando posibles errores o valores no numéricos
for col in columns_to_convert:
    # Convertir a NaN los valores no numéricos
    fight_stats[col] = pd.to_numeric(fight_stats[col], errors='coerce')
    
    # Rellenar NaN con ceros si corresponde (esto es opcional dependiendo de tu lógica)
    fight_stats[col].fillna(0, inplace=True)
    
    # Convertir la columna a entero
    fight_stats[col] = fight_stats[col].astype(int)

fight_stats = fight_stats[cols_order]


# Eliminar las columnas innecesarias
columns_to_drop = ['Control Time','Fight_lenght Sec']

# Actualizamos el DataFrame eliminando las columnas
fight_stats = fight_stats.drop(columns=columns_to_drop)

with pd.ExcelWriter('category_mappings.xlsx') as writer:
    for col, mapping_df in category_mappings.items():
        mapping_df.to_excel(writer, sheet_name=col, index=False)
        

# Listamos las columnas categóricas (que no queremos escalar)
categorical_columns = ['Weight Class', 'Method', 'Rounds', 'Format',  'Winner', 
                       'stance', 'form_last_5']

# Identificamos las columnas numéricas que no son categóricas
numeric_columns = fight_stats.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [col for col in numeric_columns if col not in categorical_columns]

# Aplicamos Winsorizing para manejar los outliers
fight_stats_winsorized = fight_stats.copy()
for col in numeric_columns:
    fight_stats_winsorized[col] = mstats.winsorize(fight_stats[col], limits=[0.05, 0.05])

# Escalar solo las columnas numéricas no categóricas
#scaler = StandardScaler()
#fight_stats_winsorized[numeric_columns] = scaler.fit_transform(fight_stats_winsorized[numeric_columns])




def calcular_ultimos_5_combates(df):
    datos_ultimos_5 = []
    peleadores = df['Fighter'].unique()
    
    for peleador in peleadores:
        peleas_peleador = df[df['Fighter'] == peleador]
        ultimas_5 = peleas_peleador.head(5)
        
        # Eliminar las columnas 'Winner' y 'Method' si existen
        if 'Winner' in ultimas_5.columns:
            ultimas_5 = ultimas_5.drop(columns=['Winner'])
        if 'Method' in ultimas_5.columns:
            ultimas_5 = ultimas_5.drop(columns=['Method'])
        
        # Filtrar solo las columnas numéricas y mantener columnas relevantes como 'Rounds'
        columnas_numericas = ultimas_5.select_dtypes(include=[np.number])
        promedio_estadisticas = columnas_numericas.mean()

        # Añadir el nombre del peleador para que no se pierda
        promedio_estadisticas['Fighter'] = peleador
        datos_ultimos_5.append(promedio_estadisticas)
    
    return pd.DataFrame(datos_ultimos_5)

# Calcular las estadísticas de los últimos 5 combates por peleador
df_estadisticas_ultimos_5 = calcular_ultimos_5_combates(fight_stats)


X = fight_stats.drop(columns=['Winner', 'Method', 'Event', 'Fight', 'Fighter'])

# Definir rutas absolutas
DATA_DIR = os.path.abspath('data')
MODELS_DIR = os.path.abspath('models')

# Crear los directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# Guardar las columnas del DataFrame X en 'data/columnas_X.csv'
columnas_X = X.columns  # Aquí usas las columnas de tu DataFrame de características X
columnas_X_path = os.path.join(DATA_DIR, 'columnas_X.csv')
try:
    pd.DataFrame(columnas_X).to_csv(columnas_X_path, index=False, header=False)
    print(f"Archivo guardado en: {columnas_X_path}")
except Exception as e:
    print(f"Error al guardar columnas_X: {e}")


# Guardar las estadísticas de los últimos 5 combates en 'data/df_estadisticas_ultimos_5.csv'
estadisticas_path = os.path.join(DATA_DIR, 'df_estadisticas_ultimos_5.csv')
try:
    df_estadisticas_ultimos_5.to_csv(estadisticas_path, index=False)
    print(f"Estadísticas guardadas en: {estadisticas_path}")
except Exception as e:
    print(f"Error al guardar estadísticas: {e}")

fight_stats_path = os.path.join(DATA_DIR, 'fight_stats.csv')
try:
    fight_stats.to_csv(fight_stats_path, index=False)
    print(f"Archivo de estadísticas actualizado en: {fight_stats_path}")
except Exception as e:
    print(f"Error al guardar fight_stats: {e}")

fight_stats_raw_path = os.path.join(DATA_DIR, 'fight_stats_raw.csv')
try:
    df.to_csv(fight_stats_raw_path, index=False)
    print(f"Archivo RAW guardado en: {fight_stats_raw_path}")
except Exception as e:
    print(f"Error al guardar fight_stats_raw: {e}")
