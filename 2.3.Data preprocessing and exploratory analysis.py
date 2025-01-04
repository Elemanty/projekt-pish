# предобработка и исследовательский анализ данных
# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.utils import shuffle
from scipy import stats

# Настройка стиля для визуализации
sns.set(style="whitegrid")

# Загрузка данных
df = pd.read_csv(r'/content/111.csv', sep=';')

# Создание DataFrame
data = pd.DataFrame(df)  

# Обзор данных
print("Первые 5 строк датасета:")
print(data.head())

print("Информация о датасете:")
print(data.info())

print("Статистическое описание:")
print(data.describe())

# Получение названия столбца по индексу 0
column_name_at_index_0 = data.columns[0]
print(column_name_at_index_0)

# переименование столбцов
data.rename(columns={'hsa-let-7a': 'miRNA ', 'A1BG': 'Gene symbol', 'NM_130786' : 'mRNA ID', '2949' : 'Target strategy', '2969' : 'Target end', '-22.6' : 'miRNA/mRNA hybridization energy (kcal/mol)', '-1381.6' : 'mRNA folding energy with the target site is open (kcal/mol)', '-1390.7' : 'mRNA folding energy (kcal/mol)', '0.02' : 'miRNA concentration (normalized)', '0.02.1' : 'mRNA concentration (normalized)', '0.98' : 'miRNA/mRNA concentration (normalized)', '-5.00' : 'Net free energy change (kcal/mol)'}, inplace=True)

# Проверка на пропуски
print("Проверка на пропуски:")
print(data.isnull().sum())
# Визуализация пропусков
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Пропуски в данных')
plt.show()

# Удаление полных дубликатов
data = data.drop_duplicates()

# настройка визуализации
from plotly.subplots import make_subplots
import plotly.express as px
colors = ['#082040', '#175073', '#3285A6', '#B8D0D9', '#6CC5D9']

column_names = data.columns.tolist()
print(column_names)

# Преобразование таргетной величины в числовой формат
data['Net free energy change (kcal/mol)'] = pd.to_numeric(data['Net free energy change (kcal/mol)'], errors='coerce')

# Анализ распределения целевой величины Net free energy change (kcal/mol)
# График распределения, скрипичные диаграммы (Violin plot)
fig = make_subplots(rows=1, cols=2, subplot_titles=['Распределение изменения свободной энергии', 'Violin Plot для изменения свободной энергии'])

# Гистограмма
hist_fig = px.histogram(df, x="-5.00", nbins=60,
                         color_discrete_sequence=colors,
                         opacity=0.7)


fig.add_trace(hist_fig['data'][0], row=1, col=1) # Добавление графика с указанием расположения

# Violin plot
violin_fig = px.violin(df, y="-5.00", color_discrete_sequence = colors, box = True)  
fig.add_trace(violin_fig['data'][0], row=1, col=2) # Добавление графика с указанием расположения

# Настройка макета
fig.update_layout(showlegend=False, title_text="Гистограмма и Violin Plot") 

# Отображение графика
fig.show()


# Проведем тест Андерсона-Дарлинга на нормальность
#В параметре dist указываем необходимое нам распределение - нормальное
result = stats.anderson(df['-5.00'], dist='norm')

# Выводи весь результат теста
print(f"Результат теста: {result}")

# Результат теста будет содержать статистику и критические значения
print('Статистика теста:', result.statistic)
print('Критические значения:', result.critical_values)

# Оценка уровня значимости на основе статистики теста, статистики в тесте считаются для конкретных уровней значимости
print('Уровень значимости:', result.significance_level)

# Оценим результат теста на нормальность
if result.statistic < result.critical_values[2]: #Данная статистика считается для уровня значимости 0.05 (5%)
    print("Данные похожи на нормальное распределение (гипотеза о нормальности не отвергается).")
else:
    print("Данные не похожи на нормальное распределение (гипотеза о нормальности отвергается).")

# Выявление выбросов с помощью boxplot
# (Для наглядности полезна визуализация выбросов, но в данном датасете нельзя удалять выбросы, так как все значения важны для дальнейшего анализа.)
numerical_cols = data.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data[col])
    plt.title(f'Выбросы в переменной: {col}')
    plt.show()

# Приведем распределение таргетной величины к нормальному
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

# MинMакс нормализация
# Создадим датафрейм, содержащий только числовые переменные
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_numeric.head()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1)) # Создание экземпляра MinMaxScaler с заданным диапазоном

# Применение масштабирования к числовым данным в DataFrame
# и сохранение результатов в новом DataFrame
df_numeric_sc = pd.DataFrame(sc.fit_transform(df_numeric), columns=df_numeric.columns) 
df_numeric_sc.head()

plt.figure(figsize=(10, 6))
sns.histplot(df_numeric_sc['-5.00'], bins=30, kde=True)  # kde=True добавляет линию плотности
plt.title('Распределение значений в столбце таргетной величины после нормализации')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.show()


# Изучение взаимосвязей между переменными
sns.pairplot(data)
plt.show()

# Визуализация корреляций между числовыми переменными
plt.figure(figsize=(12, 8))
data_numeric = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = data_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Матрица корреляции')
plt.show()
