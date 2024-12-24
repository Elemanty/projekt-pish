
# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import lazypredict
from sklearn.datasets import load_breast_cancer
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle

# Настройка стиля для визуализации
sns.set(style="whitegrid")

# Загрузка данных
data = pd.read_csv(r'C:\Users\User\Desktop\ИТМО\111.csv', sep=';')

data.drop_duplicates(inplace=True)  

# 1. Обзор данных
print("Первые 5 строк датасета:")
print(data.head())

print("\nИнформация о датасете:")
print(data.info())

print("\nСтатистическое описание:")
print(data.describe())

# Получение названия столбца по индексу 0
column_name_at_index_0 = data.columns[0]
print(column_name_at_index_0)

# переименование столбцов
data.rename(columns={'hsa-let-7a': 'miRNA ', 'A1BG': 'Gene symbol', 'NM_130786' : 'mRNA ID', '2949' : 'Target strategy', '2969' : 'Target end', '-22.6' : 'miRNA/mRNA hybridization energy (kcal/mol)', '-1381.6' : 'mRNA folding energy with the target site is open (kcal/mol)', '-1390.7' : 'mRNA folding energy (kcal/mol)', '0.02' : 'miRNA concentration (normalized)', '0.02.1' : 'mRNA concentration (normalized)', '0.98' : 'miRNA/mRNA concentration (normalized)', '-5.00' : 'Net free energy change (kcal/mol)'}, inplace=True)


# 2. Проверка на пропуски
print("\nПроверка на пропуски:")
print(data.isnull().sum())

# Визуализация пропусков
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Пропуски в данных')
plt.show()


# Убедимся, что в DataFrame достаточно столбцов
if len(data.columns) >= 12:
    column_name_at_index_11 = data.columns[11]  # Индекс 11 соответствует 12-му столбцу
    column_data = data[column_name_at_index_11].dropna()  # Удаляем NaN значения

    # Построение графика логарифмированных данных
    plt.figure(figsize=(10, 6))
    sns.histplot(column_data, kde=True, bins=30)  # kde=True добавляет линию KDE
    plt.title(f'Логарифмированное распределение данных в столбце "{column_name_at_index_11}"')
    plt.xlabel(f'Логарифмированные данные ({column_name_at_index_11})')
    plt.ylabel('Частота')
    plt.grid()
    plt.show()

    # Применение Yeo-Johnson преобразования
    pt = PowerTransformer(method='yeo-johnson')
    transformed_data = pt.fit_transform(column_data.values.reshape(-1, 1))
    # Построение графика исходного распределения
    plt.figure(figsize=(10, 6))
    sns.histplot(column_data, kde=True, bins=30)
    plt.title(f'Исходное распределение данных в столбце "{column_name_at_index_11}"')
    plt.xlabel('Данные')
    plt.ylabel('Частота')
    plt.grid()
    plt.show()

    # Построение графика преобразованных данных
    plt.figure(figsize=(10, 6))
    sns.histplot(transformed_data, kde=True, bins=30)
    plt.title(f'Преобразованные данные (Yeo-Johnson) из данных в столбце "{column_name_at_index_11}"')
    plt.xlabel('Преобразованные данные')
    plt.ylabel('Частота')
    plt.grid()
    plt.show()

    # Вывод параметров преобразования
    print(f'Параметры Yeo-Johnson преобразования: {pt.lambdas_}')
print(column_data)
# Удаление выбросов
mean = column_data.mean()
std_dev = column_data.std()
threshold = 3
outliers = column_data[(column_data < mean - threshold * std_dev) | (column_data > mean + threshold * std_dev)]
print("Выбросы:")
print(outliers)

# 3. Визуализация распределения целевой переменной
if 'data.columns[11]' in data.columns:  
    plt.figure(figsize=(10, 6))
    sns.histplot(data['data.columns[11]'], bins=30, kde=True)
    plt.title('Распределение целевой переменной')
    plt.xlabel('Значение')
    plt.ylabel('Частота')
    plt.show()

# 4. Визуализация корреляций между числовыми переменными
plt.figure(figsize=(12, 8))
data_numeric = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = data_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()

# 5. Анализ категориальных переменных
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data[col])
    plt.title(f'Распределение по категории: {col}')
    plt.xticks(rotation=45)
    plt.show()

# 6. Выявление выбросов с помощью boxplot
numerical_cols = data.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Выбросы в переменной: {col}')
    plt.show()

# 7. Дополнительные визуализации
sns.pairplot(data)
plt.show()
