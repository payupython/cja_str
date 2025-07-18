import pandas as pd
import yaml

# 1. Leer el mapping de columnas
with open('RouteAnalysys/.rules_mapping_kpis', 'r') as f:
    mapping = yaml.safe_load(f)

col_map = mapping['column_mapping']
kpi_map = mapping['kpi_specific_mapping']["Usuarios con más de X eventos antes de la conversión"]

# 2. Leer los datos de ejemplo
# Ajusta la ruta si tu archivo se llama diferente
df = pd.read_csv('RouteAnalysys/data/import/ejemplo.csv')

# 3. Extraer columnas relevantes según el mapping
user_col = kpi_map['usuario']
ruta_col = kpi_map['secuencia_eventos']
evento_conversion = kpi_map['evento_conversion']
param_x = 3  # Puedes parametrizarlo

# 4. Función para contar eventos antes de la conversión
def eventos_antes_conversion(ruta, evento_conversion, sep=' - ', min_x=3):
    eventos = []
    for bloque in ruta.split(sep):
        partes = bloque.strip().split(' ', 1)
        if len(partes) == 2:
            eventos.append(partes[1].strip())
        else:
            eventos.append(partes[0].strip())
    try:
        idx = next(i for i, ev in enumerate(eventos) if evento_conversion in ev)
        return idx  # Número de eventos antes de la conversión
    except StopIteration:
        return None  # No hay conversión

# 5. Calcular el KPI
usuarios_con_mas_x = set()
for _, row in df.iterrows():
    n_antes = eventos_antes_conversion(str(row[ruta_col]), evento_conversion)
    if n_antes is not None and n_antes > param_x:
        usuarios_con_mas_x.add(row[user_col])

print(f"Número de usuarios con más de {param_x} eventos antes de la conversión: {len(usuarios_con_mas_x)}")
print("Usuarios:", usuarios_con_mas_x) 