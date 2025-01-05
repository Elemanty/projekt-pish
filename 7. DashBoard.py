!pip install dash
# Создание дашборда
import pandas as pd
from dash import Dash, dcc, html
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Метрики качества данных
completeness = df.notnull().mean() * 100
accuracy_value1 = (df['-5.00'] < 0).mean() * 100
uniqueness_gene = df['A1BG'].nunique() / df['A1BG'].count() * 100
consistency_duplicates = df.duplicated().sum()
timeliness_nan_count = df.isna().sum()

# Создание приложения Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Дашборд качества данных для miRNA database"),
    
    dcc.Graph(
        id='completeness-bar',
        figure={
            'data': [
                go.Bar(
                    x=completeness.index,
                    y=completeness.values,
                    name='Полнота',
                    marker=dict(color='royalblue')
                )
            ],
            'layout': go.Layout(
                title='Полнота данных по столбцам',
                xaxis={'title': 'Столбцы'},
                yaxis={'title': 'Процент заполненности (%)'},
                hovermode='closest'
            )
        }
    ),
    
    dcc.Graph(
        id='accuracy-bar',
        figure={
            'data': [
                go.Bar(
                    x=['-5.00'],
                    y=[accuracy_value1],
                    name='Корректность',
                    marker=dict(color='orange')
                )
            ],
            'layout': go.Layout(
                title='Корректность данных (	Net free energy change > 0)',
                xaxis={'title': 'Столбцы'},
                yaxis={'title': 'Процент корректности (%)'},
                hovermode='closest'
            )
        }
    ),
    
    dcc.Graph(
        id='uniqueness-pie',
        figure={
            'data': [
                go.Pie(
                    labels=['Уникальные', 'Дубликаты'],
                    values=[df['A1BG'].nunique(), df['A1BG'].count() - df['A1BG'].nunique()],
                    marker=dict(colors=['lightgreen', 'lightcoral'])
                )
            ],
            'layout': go.Layout(
                title='Уникальность значений в столбце Gene'
            )
        }
    ),
    
    html.Div([
        html.H3("Количество дубликатов:"),
        html.P(f"{consistency_duplicates}"),
        
        html.H3("Количество NaN в каждом столбце:"),
        html.Pre(timeliness_nan_count.to_string())
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)


# дашборд с частотой значений таргетной величины
# Подсчитываем частоту повторяющихся значений
value_counts = df['-5.00'].value_counts().reset_index()
value_counts.columns = ['Value', 'Frequency']

# Создаем экземпляр приложения Dash
app = dash.Dash(__name__)

# Определяем макет приложения
app.layout = html.Div([
    dcc.Dropdown(
        id='value-selector',
        options=[{'label': str(val), 'value': val} for val in value_counts['Value']],
        value=value_counts['Value'][0],  # Значение по умолчанию
        multi=False
    ),
    dcc.Graph(id='frequency-graph')
])

# Определяем обратный вызов для обновления графика
@app.callback(
    Output('frequency-graph', 'figure'),
    Input('value-selector', 'value')  # Добавляем Input элемент
)
def update_graph(selected_value):
    # Фильтруем данные на основе выбранного значения (если необходимо)
    filtered_df = df[df['-5.00'] == selected_value]
    
    # Обновляем частоту для выбранного значения
    filtered_counts = filtered_df['-5.00'].value_counts().reset_index()
    filtered_counts.columns = ['Value', 'Frequency']
    
    # Создаем график с использованием Plotly Express
    fig = px.bar(filtered_counts, x='Value', y='Frequency', 
                  title=f'Частота значений в столбце 	Net free energy change для {selected_value}',
                  labels={'Value': 'Значение', 'Frequency': 'Частота'})
    
    return fig

# Запускаем приложение
if __name__ == '__main__':
    app.run_server(debug=True)

