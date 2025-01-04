# Метрики качества данных
# Полнота
completeness = df.notnull().mean() * 100
# Уникальность
uniqueness_gene = df['A1BG'].nunique() / df['A1BG'].count() * 100

# Консистентность (количество дубликатов)
consistency_duplicates = df.duplicated().sum()

# Актуальность (количество NaN)
timeliness_nan_count = df.isna().sum()

# Вывод метрик
print(f"Полнота:\n{completeness}\n")
print(f"Уникальность Gene: {uniqueness_gene:.2f}%\n")
print(f"Количество дубликатов: {consistency_duplicates}\n")
print(f"Количество NaN в каждом столбце:\n{timeliness_nan_count}\n")
