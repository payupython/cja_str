import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yaml
import os
from datetime import datetime
import sys

# Importar funciones de test_kpis.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_kpis import (
    KPI_FUNCTIONS, parsear_eventos, cargar_insights,
    kpi_most_common_events, kpi_most_common_2grams,
    kpi_post_conversion_navigation, kpi_rapid_conversion_anomaly,
    kpi_circular_navigation, kpi_excessive_event_repetition,
    kpi_abandonment_after_key_event
)

# Configuración de la app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "RouteAnalysys - Dashboard de KPIs"

# Cargar configuración
def cargar_configuracion():
    with open('.rules_mapping_kpis', 'r') as f:
        mapping = yaml.safe_load(f)
    return mapping

def cargar_datos():
    mapping = cargar_configuracion()
    col_map = mapping['column_mapping']
    df = pd.read_csv('data/import/ejemplo.csv')
    
    user_col = col_map.get('user_id', 'user_id')
    ruta_col = col_map.get('ruta', 'ruta')
    if ruta_col not in df.columns:
        ruta_col = col_map.get('Interactions', 'Interactions')
    
    return df, user_col, ruta_col, mapping

# Layout principal
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📊 RouteAnalysys Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Controles
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("⚙️ Configuración", className="card-title"),
                    dbc.Button("🔄 Calcular KPIs", id="btn-calcular", color="primary", className="mb-3"),
                    html.Div(id="status-calculo", className="text-muted")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # KPIs en tarjetas
    html.Div(id="kpi-cards", className="mb-4"),
    
    # Tabla de resultados
    dbc.Row([
        dbc.Col([
            html.H4("📋 Resultados Detallados", className="mb-3"),
            html.Div(id="tabla-resultados")
        ])
    ]),
    
    # Gráficos
    dbc.Row([
        dbc.Col([
            html.H4("📈 Visualizaciones", className="mb-3"),
            html.Div(id="graficos")
        ])
    ])
], fluid=True)

# Callback para calcular KPIs
@app.callback(
    [Output("kpi-cards", "children"),
     Output("tabla-resultados", "children"),
     Output("graficos", "children"),
     Output("status-calculo", "children")],
    [Input("btn-calcular", "n_clicks")]
)
def calcular_kpis(n_clicks):
    if n_clicks is None:
        return [], [], [], "Presiona el botón para calcular KPIs"
    
    try:
        # Cargar datos y configuración
        df, user_col, ruta_col, mapping = cargar_datos()
        kpi_map = mapping['kpi_specific_mapping']
        ayuda_kpis = cargar_insights()
        
        # Calcular KPIs
        resultados = []
        for kpi_name, params in kpi_map.items():
            func = KPI_FUNCTIONS.get(kpi_name)
            evento_conversion = params.get('evento_conversion', '')
            if func:
                func_params = {k: params[k] for k in params if k not in ['usuario', 'secuencia_eventos']}
                try:
                    result = func(df, user_col, ruta_col, **func_params)
                    if result is not None:
                        ayuda = ayuda_kpis.get(kpi_name, "")
                        resultados.append({
                            "KPI": kpi_name,
                            "Resultado": result,
                            "Evento de conversión": evento_conversion,
                            "Guía de interpretación": ayuda
                        })
                except Exception as e:
                    resultados.append({
                        "KPI": kpi_name,
                        "Resultado": f"[ERROR] {e}",
                        "Evento de conversión": evento_conversion,
                        "Guía de interpretación": ayuda_kpis.get(kpi_name, "")
                    })
        
        df_resultados = pd.DataFrame(resultados)
        
        # Generar tarjetas de KPIs
        kpi_cards = generar_tarjetas_kpis(df_resultados)
        
        # Generar tabla de resultados
        tabla = generar_tabla_resultados(df_resultados)
        
        # Generar gráficos
        graficos = generar_graficos(df_resultados)
        
        status = f"✅ KPIs calculados exitosamente - {len(resultados)} métricas generadas"
        
        return kpi_cards, tabla, graficos, status
        
    except Exception as e:
        return [], [], [], f"❌ Error: {str(e)}"

def generar_tarjetas_kpis(df_resultados):
    """Generar tarjetas de Bootstrap para cada KPI"""
    cards = []
    
    # Agrupar KPIs por categoría
    categorias = {
        "📊 Métricas Básicas": ["drop_off_rate", "backtracking_rate", "conversion_rate"],
        "🎯 Análisis Avanzado": ["most_common_entry_event", "most_common_exit_point", "first_key_action"],
        "🔍 Detección de Anomalías": ["post_conversion_navigation", "rapid_conversion_anomaly", "circular_navigation"],
        "📈 Descriptivos": ["most_common_events", "most_common_2grams"]
    }
    
    for categoria, kpis in categorias.items():
        categoria_cards = []
        for kpi in kpis:
            row = df_resultados[df_resultados['KPI'] == kpi]
            if not row.empty:
                resultado = row.iloc[0]['Resultado']
                interpretacion = row.iloc[0]['Guía de interpretación']
                
                # Determinar color basado en el tipo de resultado
                color = "primary"
                if isinstance(resultado, str) and "ERROR" in resultado:
                    color = "danger"
                elif isinstance(resultado, (int, float)) and resultado > 0:
                    color = "success"
                
                card = dbc.Card([
                    dbc.CardBody([
                        html.H6(kpi.replace("_", " ").title(), className="card-title"),
                        html.H4(str(resultado), className=f"text-{color}"),
                        html.P(interpretacion[:100] + "..." if len(interpretacion) > 100 else interpretacion,
                               className="card-text text-muted small")
                    ])
                ], className="mb-3")
                categoria_cards.append(card)
        
        if categoria_cards:
            cards.append(html.H5(categoria, className="mt-4 mb-3"))
            cards.append(dbc.Row([dbc.Col(card, width=4) for card in categoria_cards]))
    
    return cards

def generar_tabla_resultados(df_resultados):
    """Generar tabla de resultados con formato mejorado"""
    # Preparar datos para la tabla
    df_table = df_resultados.copy()
    df_table['Resultado'] = df_table['Resultado'].astype(str)
    
    # Limitar longitud de interpretación para la tabla
    df_table['Guía de interpretación'] = df_table['Guía de interpretación'].apply(
        lambda x: x[:150] + "..." if len(x) > 150 else x
    )
    
    return dash_table.DataTable(
        id='tabla-kpis',
        columns=[
            {"name": "KPI", "id": "KPI"},
            {"name": "Resultado", "id": "Resultado"},
            {"name": "Evento de Conversión", "id": "Evento de conversión"},
            {"name": "Interpretación", "id": "Guía de interpretación"}
        ],
        data=df_table.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'minWidth': '100px',
            'maxWidth': '300px',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{Resultado} contains ERROR'},
                'backgroundColor': '#ffebee',
                'color': '#c62828'
            }
        ],
        page_size=10,
        sort_action='native',
        filter_action='native'
    )

def generar_graficos(df_resultados):
    """Generar gráficos interactivos"""
    graficos = []
    
    # Gráfico 1: Distribución de tipos de KPIs
    df_resultados['Tipo'] = df_resultados['KPI'].apply(categorizar_kpi)
    tipo_counts = df_resultados['Tipo'].value_counts()
    
    fig1 = px.pie(
        values=tipo_counts.values,
        names=tipo_counts.index,
        title="Distribución de KPIs por Categoría"
    )
    
    # Gráfico 2: KPIs con errores vs exitosos
    df_resultados['Estado'] = df_resultados['Resultado'].apply(
        lambda x: 'Error' if 'ERROR' in str(x) else 'Exitoso'
    )
    estado_counts = df_resultados['Estado'].value_counts()
    
    fig2 = px.bar(
        x=estado_counts.index,
        y=estado_counts.values,
        title="Estado de Cálculo de KPIs",
        color=estado_counts.index,
        color_discrete_map={'Exitoso': '#4CAF50', 'Error': '#F44336'}
    )
    
    # Gráfico 3: Top KPIs por resultado numérico
    df_numericos = df_resultados[
        df_resultados['Resultado'].apply(lambda x: isinstance(x, (int, float)) or 
                                       (isinstance(x, str) and x.replace('.', '').replace('%', '').isdigit()))
    ].copy()
    
    if not df_numericos.empty:
        df_numericos['Valor_Numerico'] = df_numericos['Resultado'].apply(
            lambda x: float(str(x).replace('%', '')) if isinstance(x, str) else float(x)
        )
        df_numericos = df_numericos.nlargest(10, 'Valor_Numerico')
        
        fig3 = px.bar(
            df_numericos,
            x='KPI',
            y='Valor_Numerico',
            title="Top 10 KPIs por Valor Numérico",
            color='Valor_Numerico',
            color_continuous_scale='viridis'
        )
        fig3.update_layout(xaxis_tickangle=-45)
    else:
        fig3 = go.Figure()
        fig3.add_annotation(text="No hay KPIs numéricos para mostrar", xref="paper", yref="paper")
    
    graficos.append(dbc.Row([
        dbc.Col(dcc.Graph(figure=fig1), width=6),
        dbc.Col(dcc.Graph(figure=fig2), width=6)
    ]))
    
    graficos.append(dbc.Row([
        dbc.Col(dcc.Graph(figure=fig3), width=12)
    ]))
    
    return graficos

def categorizar_kpi(kpi_name):
    """Categorizar KPI basado en su nombre"""
    if any(word in kpi_name.lower() for word in ['rate', 'conversion', 'drop']):
        return "Métricas Básicas"
    elif any(word in kpi_name.lower() for word in ['common', 'entry', 'exit', 'first']):
        return "Análisis Avanzado"
    elif any(word in kpi_name.lower() for word in ['anomaly', 'navigation', 'circular', 'excessive']):
        return "Detección de Anomalías"
    else:
        return "Descriptivos"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 