# создание базы данных
import sqlite3

# Подключаемся к базе данных (или создаем ее)
conn = sqlite3.connect('dataset.db')
# Создаем курсор
cursor = conn.cursor()
df.to_sql('miRNA_information', conn, if_exists='replace', index=False)
# Сохраняем изменения и закрываем соединение
conn.commit()
conn.close()
# Подключение к базе данных
conn = sqlite3.connect('dataset.db')
# Чтение данных в DataFrame
df = pd.read_sql_query(query, conn)
# Закрытие соединения
conn.close()
