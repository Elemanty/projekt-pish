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


# Считывание CSV файла с разделителем ';'
data = pd.read_csv(r'C:\Users\User\Desktop\ИТМО\111.csv', sep=';')


# Вывод всего DataFrame
print(data)

# Получение названия столбца по индексу 0
column_name_at_index_0 = data.columns[0]
print(column_name_at_index_0)

# Проверка количества столбцов
print("Количество столбцов:", len(data.columns))

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








# Применяем one-hot encoding для других столбцов (пример)
one_hot_encoded_df = pd.get_dummies(data, columns=['hsa-let-7a'], prefix='miRNA', drop_first=True)
one_hot_encoded_df.iloc[:, 5:] = one_hot_encoded_df.iloc[:, 5:].astype(int)

print("miRNA после One-Hot Encoding:")
print(one_hot_encoded_df)




from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# Создание DataFrame
df = pd.DataFrame(data)

# Вывод оригинального DataFrame
print("Оригинальный DataFrame:")
print(df)

# Инициализация OneHotEncoder
encoder = OneHotEncoder(sparse_output=True)

# Применение OneHotEncoder к первым трем столбцам
encoded_columns = encoder.fit_transform(df.iloc[:, :3])

# Получение имен новых столбцов
encoded_columns_names = encoder.get_feature_names_out(df.columns[:3])

# Создание DataFrame с закодированными столбцами
encoded_df = pd.DataFrame(encoded_columns, columns=encoded_columns_names)

# Объединение закодированных столбцов с остальными данными
final_df = pd.concat([encoded_df, df.iloc[:, 3]], axis=1)

# Вывод итогового DataFrame
print("Итоговый DataFrame после OneHotEncoding:")
print(final_df)






X, y = shuffle(final_df, column_data, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print("результат")

print(models)
