# Метрики качества данных
# Полнота
completeness = df.notnull().mean() * 100
# Уникальность
uniqueness_gene = df['A1BG'].nunique() / df['A1BG'].count() * 100

# Консистентность (количество дубликатов)
consistency_duplicates = df.duplicated().sum()

# Актуальность (количество NaN)
timeliness_nan_count = df.isna().sum()

# Корректность (Чистое изменение свободной энергии не может быть положительным)
accuracy_value = (df['-5.00'] < 0).mean() * 100

# Вывод метрик
print(f"Полнота:\n{completeness}\n")
print(f"Уникальность Gene: {uniqueness_gene:.2f}%\n")
print(f"Количество дубликатов: {consistency_duplicates}\n")
print(f"Количество NaN в каждом столбце:\n{timeliness_nan_count}\n")
print (accuracy_value)
