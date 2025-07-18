import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

# Inicializar la app Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'RouteAnalysys - MVP'

# Ruta de datos de ejemplo
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'import')

def cargar_datos():
    archivos = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if archivos:
        return pd.read_csv(os.path.join(data_dir, archivos[0]))
    else:
        return pd.DataFrame({'evento': [], 'usuario': [], 'timestamp': []})

df = cargar_datos()

# Layout con Bootstrap
app.layout = dbc.Container([
    html.H1('RouteAnalysys - MVP', className='my-3'),
    dbc.Row([
        dbc.Col([
            html.Label('Selecciona usuario:'),
            dcc.Dropdown(
                id='dropdown-usuario',
                options=[{'label': u, 'value': u} for u in sorted(df['usuario'].unique())] if not df.empty else [],
                value=df['usuario'].unique()[0] if not df.empty else None,
                clearable=True,
                placeholder='Selecciona un usuario',
            ),
            html.Br(),
            html.Label('Filtra por número mínimo de eventos:'),
            dcc.Slider(
                id='slider-num-eventos',
                min=1,
                max=int(df['evento'].value_counts().max()) if not df.empty else 1,
                step=1,
                value=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], md=4),
        dbc.Col([
            html.H5('Vista previa de datos filtrados'),
            dash_table.DataTable(
                id='tabla-filtrada',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'},
            ),
        ], md=8),
    ]),
    html.Hr(),
    html.H3('Gráfico de eventos filtrados'),
    dcc.Graph(id='grafico-filtrado'),
], fluid=True)

# Callback para filtrar datos y actualizar tabla y gráfico
@app.callback(
    Output('tabla-filtrada', 'data'),
    Output('grafico-filtrado', 'figure'),
    Input('dropdown-usuario', 'value'),
    Input('slider-num-eventos', 'value')
)
def actualizar_tabla_grafico(usuario, min_eventos):
    dff = df.copy()
    if usuario:
        dff = dff[dff['usuario'] == usuario]
    # Filtrar por número mínimo de eventos (por usuario)
    conteo = dff['evento'].value_counts()
    eventos_validos = conteo[conteo >= min_eventos].index
    dff = dff[dff['evento'].isin(eventos_validos)]
    fig = px.histogram(dff, x='evento', color='usuario', barmode='group') if not dff.empty else {}
    return dff.to_dict('records'), fig

if __name__ == '__main__':
    app.run(debug=True)