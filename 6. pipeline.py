import requests
from bs4 import BeautifulSoup
import pandas as pd

class DataPipeline:
    def __init__(self, search_term):
        self.search_term = search_term
        self.data = None

    def fetch_data(self):
        # Формируем URL для поиска
        url = f"https://www.chemspider.com/Chemical-Structure.{self.search_term}.html"
        
        # Выполняем запрос к сайту
        response = requests.get(url)
        
        if response.status_code == 200:
            self.parse_data(response.text)
        else:
            print(f"Failed to retrieve data: {response.status_code}")

    def parse_data(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        compound_name = soup.find().text.strip()
        properties = {}
        property_elements = soup.find_all('div', class_='property')
        
        for prop in property_elements:
            label = prop.find('span', class_='property-label').text.strip()
            value = prop.find('span', class_='property-value').text.strip()
            properties[label] = value
        
        # Сохраняем данные в DataFrame
        self.data = pd.DataFrame({
            'Compound Name': [compound_name],
            **properties
        })

    def preprocess_data(self):
        # Предобработка данных
        if self.data is not None:
            # Преобразуем типы данных, обрабатываем пропуски и т.д.
            self.data.fillna('N/A', inplace=True)

    def save_to_csv(self, output_file):
        if self.data is not None:
            self.data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")

if __name__ == "__main__":
    search_term = "474454"  # идентификатор соединения 2-гидроксиэтилпропионат
    pipeline = DataPipeline(search_term)
    
    pipeline.fetch_data()
    pipeline.preprocess_data()
    pipeline.save_to_csv('output.csv')

    import pandas as pd

df = pd.read_csv('output.csv')

# Выводим содержимое DataFrame
print(df)

