import pandas as pd
import sqlite3
import requests  # Используется для получения данных из API
from bs4 import BeautifulSoup  # Используется для парсинга HTML

# Пример функции для сбора данных
def collect_data():
    url = 'https://www.ebi.ac.uk/chembl/' 
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Предположим, что данные находятся в таблице HTML
    data = []
    for row in soup.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) >= 12:  
            # Извлекаем данные из первых трех столбцов (не числовые)
            non_numeric_col1 = cols[0].text.strip()
            non_numeric_col2 = cols[1].text.strip()
            non_numeric_col3 = cols[2].text.strip()

            # Извлекаем числовые данные из оставшихся столбцов
            numeric_data = [float(col.text.strip()) for col in cols[3:12]]  # Предполагается, что это числовые данные
            
            # Собираем все данные в одну строку
            data.append([non_numeric_col1, non_numeric_col2, non_numeric_col3] + numeric_data)

    return data

# Сбор данных
data = collect_data()

# Создание DataFrame с 12 столбцами
columns = ['Col1', 'Col2', 'Col3', 'NumCol1', 'NumCol2', 'NumCol3', 'NumCol4', 'NumCol5', 'NumCol6', 'NumCol7', 'NumCol8', 'NumCol9']
df = pd.DataFrame(data, columns=columns)

# Подключение к базе данных SQLite
conn = sqlite3.connect('your_database.db')

# Запись DataFrame в таблицу в базе данных
df.to_sql('miRNA_information', conn, if_exists='replace', index=False)

# Закрытие соединения
conn.close()

print("Данные успешно собраны и сохранены в таблице miRNA_information.")
