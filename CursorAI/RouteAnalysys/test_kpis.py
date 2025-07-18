import pandas as pd
import yaml
import numpy as np
import re

# --- Funciones para cada KPI ---
def parsear_eventos(ruta, sep=' - '):
    eventos = []
    for bloque in str(ruta).split(sep):
        partes = bloque.strip().split(' ', 1)
        if len(partes) == 2:
            evento = partes[1].strip()
        else:
            evento = partes[0].strip()
        # Eliminar números y símbolos no alfanuméricos al principio y final
        evento = re.sub(r'^[\W\d\s]+|[\W\s]+$', '', evento)
        eventos.append(evento)
    return eventos

def kpi_drop_off_rate(df, user_col, ruta_col, step, evento_conversion="purchase"):
    usuarios_en_step = set()
    usuarios_abandonan = set()
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if step in eventos:
            usuarios_en_step.add(row[user_col])
            # Verificar si tiene conversión
            tiene_conversion = any(evento_conversion in ev for ev in eventos)
            if not tiene_conversion:
                usuarios_abandonan.add(row[user_col])
    if not usuarios_en_step:
        return 0
    return (len(usuarios_abandonan) / len(usuarios_en_step)) * 100

def kpi_backtracking_rate(df, user_col, ruta_col):
    sesiones_con_backtrack = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        for i in range(1, len(eventos)):
            if eventos[i] in eventos[:i]:
                sesiones_con_backtrack += 1
                break
    return (sesiones_con_backtrack / len(df)) * 100 if len(df) else 0

def kpi_average_path_length(df, user_col, ruta_col):
    return np.mean([len(parsear_eventos(row[ruta_col])) for _, row in df.iterrows()])

def kpi_excessive_steps(df, user_col, ruta_col, threshold):
    sesiones_exceso = 0
    for _, row in df.iterrows():
        if len(parsear_eventos(row[ruta_col])) > threshold:
            sesiones_exceso += 1
    return (sesiones_exceso / len(df)) * 100 if len(df) else 0

def kpi_avg_time_between_events(df, user_col, ruta_col, timestamp):
    if timestamp not in df.columns:
        return None
    tiempos = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        ts = str(row[timestamp]).split(' - ')
        if len(ts) > 1:
            try:
                ts = [pd.to_datetime(t) for t in ts]
                diffs = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
                tiempos.extend(diffs)
            except Exception:
                continue
    return np.mean(tiempos) if tiempos else None

def kpi_conversion_rate_per_path(df, user_col, ruta_col, evento_conversion):
    total = len(df)
    con_conversion = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if any(evento_conversion in ev for ev in eventos):
            con_conversion += 1
    return (con_conversion / total) * 100 if total else 0

def kpi_unique_paths(df, user_col, ruta_col):
    rutas = set()
    for _, row in df.iterrows():
        rutas.add(row[ruta_col])
    return len(rutas)

def kpi_path_entropy(df, user_col, ruta_col):
    from math import log
    rutas = [row[ruta_col] for _, row in df.iterrows()]
    total = len(rutas)
    from collections import Counter
    counts = Counter(rutas)
    return -sum((c/total) * log(c/total) for c in counts.values()) if total else 0

def kpi_most_common_entry_event(df, user_col, ruta_col, top_n=5):
    entradas = [parsear_eventos(row[ruta_col])[0] for _, row in df.iterrows() if parsear_eventos(row[ruta_col])]
    from collections import Counter
    counter = Counter(entradas)
    # Devolver tabla con top N eventos más frecuentes
    top_events = counter.most_common(top_n)
    if not top_events:
        return "No hay eventos de entrada"
    
    # Crear tabla formateada
    tabla = []
    for i, (evento, frecuencia) in enumerate(top_events, 1):
        porcentaje = (frecuencia / len(entradas)) * 100
        tabla.append(f"{i}. {evento}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

def kpi_last_event_before_dropoff_or_conversion(df, user_col, ruta_col, evento_conversion, top_n=5):
    ultimos = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if any(evento_conversion in ev for ev in eventos):
            idx = [i for i, ev in enumerate(eventos) if evento_conversion in ev]
            if idx and idx[0] > 0:
                ultimos.append(eventos[idx[0]-1])
    
    if not ultimos:
        return "No se encontraron eventos previos a conversión"
    
    from collections import Counter
    counter = Counter(ultimos)
    # Devolver tabla con top N eventos más frecuentes
    top_events = counter.most_common(top_n)
    
    # Crear tabla formateada
    tabla = []
    for i, (evento, frecuencia) in enumerate(top_events, 1):
        porcentaje = (frecuencia / len(ultimos)) * 100
        tabla.append(f"{i}. {evento}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

def kpi_most_common_exit_point(df, user_col, ruta_col, top_n=5):
    salidas = [parsear_eventos(row[ruta_col])[-1] for _, row in df.iterrows() if parsear_eventos(row[ruta_col])]
    from collections import Counter
    counter = Counter(salidas)
    # Devolver tabla con top N eventos más frecuentes
    top_events = counter.most_common(top_n)
    if not top_events:
        return "No hay eventos de salida"
    
    # Crear tabla formateada
    tabla = []
    for i, (evento, frecuencia) in enumerate(top_events, 1):
        porcentaje = (frecuencia / len(salidas)) * 100
        tabla.append(f"{i}. {evento}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

def kpi_event_repetition_rate(df, user_col, ruta_col):
    repeticiones = 0
    total = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        total += len(eventos)
        repeticiones += len(eventos) - len(set(eventos))
    return (repeticiones / total) * 100 if total else 0

def kpi_average_click_depth(df, user_col, ruta_col):
    # No implementado: requiere jerarquía de URLs
    return None

def kpi_key_event_ratio(df, user_col, ruta_col, key_event):
    sesiones_con_evento = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if any(key_event in ev for ev in eventos):
            sesiones_con_evento += 1
    total_sesiones = len(df)
    return (sesiones_con_evento / total_sesiones) * 100 if total_sesiones else 0

def kpi_total_time_on_path(df, user_col, ruta_col, timestamp):
    # No implementado: requiere timestamps por evento
    return None

def kpi_path_standard_deviation(df, user_col, ruta_col):
    longitudes = [len(parsear_eventos(row[ruta_col])) for _, row in df.iterrows()]
    return np.std(longitudes) if longitudes else 0

def kpi_first_key_action(df, user_col, ruta_col, key_events_list, top_n=5):
    acciones = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        for ev in eventos:
            if any(key in ev for key in key_events_list):
                acciones.append(ev)
                break
    
    if not acciones:
        return "No se encontraron acciones clave"
    
    from collections import Counter
    counter = Counter(acciones)
    # Devolver tabla con top N eventos más frecuentes
    top_events = counter.most_common(top_n)
    
    # Crear tabla formateada
    tabla = []
    for i, (evento, frecuencia) in enumerate(top_events, 1):
        porcentaje = (frecuencia / len(acciones)) * 100
        tabla.append(f"{i}. {evento}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

def kpi_technical_bounce_rate(df, user_col, ruta_col):
    bounces = 0
    for _, row in df.iterrows():
        if len(parsear_eventos(row[ruta_col])) <= 1:
            bounces += 1
    return (bounces / len(df)) * 100 if len(df) else 0

# KPIs avanzados (ya implementados antes)
def kpi_usuarios_mas_x_eventos(df, user_col, ruta_col, evento_conversion, parametro_x):
    usuarios_con_mas_x = set()
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        try:
            idx = next(i for i, ev in enumerate(eventos) if evento_conversion in ev)
            if idx > parametro_x:
                usuarios_con_mas_x.add(row[user_col])
        except StopIteration:
            continue
    return len(usuarios_con_mas_x), usuarios_con_mas_x

def kpi_sesiones_multiples_conversiones(df, user_col, ruta_col, evento_conversion):
    sesiones = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if sum(1 for ev in eventos if evento_conversion in ev) > 1:
            sesiones += 1
    return sesiones

def kpi_rutas_multiples_conversiones(df, user_col, ruta_col, evento_conversion):
    rutas = 0
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        if sum(1 for ev in eventos if evento_conversion in ev) > 1:
            rutas += 1
    return rutas

def kpi_pasos_hasta_primera_conversion(df, user_col, ruta_col, evento_conversion):
    pasos = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        try:
            idx = next(i for i, ev in enumerate(eventos) if evento_conversion in ev)
            pasos.append(idx)
        except StopIteration:
            continue
    return sum(pasos) / len(pasos) if pasos else 0

def kpi_pasos_entre_conversiones(df, user_col, ruta_col, evento_conversion):
    pasos_entre = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        indices = [i for i, ev in enumerate(eventos) if evento_conversion in ev]
        if len(indices) > 1:
            for i in range(1, len(indices)):
                pasos_entre.append(indices[i] - indices[i-1] - 1)
    return sum(pasos_entre) / len(pasos_entre) if pasos_entre else 0

def kpi_most_common_events(df, user_col, ruta_col, top_n=10):
    """Eventos más frecuentes en general (todos los eventos de todas las rutas)"""
    todos_eventos = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        todos_eventos.extend(eventos)
    
    if not todos_eventos:
        return "No se encontraron eventos"
    
    from collections import Counter
    counter = Counter(todos_eventos)
    # Devolver tabla con top N eventos más frecuentes
    top_events = counter.most_common(top_n)
    
    # Crear tabla formateada
    tabla = []
    for i, (evento, frecuencia) in enumerate(top_events, 1):
        porcentaje = (frecuencia / len(todos_eventos)) * 100
        tabla.append(f"{i}. {evento}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

def kpi_most_common_2grams(df, user_col, ruta_col, top_n=10):
    """Secuencias de 2 eventos más frecuentes"""
    bigrams = []
    for _, row in df.iterrows():
        eventos = parsear_eventos(row[ruta_col])
        for i in range(len(eventos) - 1):
            bigram = f"{eventos[i]} → {eventos[i+1]}"
            bigrams.append(bigram)
    
    if not bigrams:
        return "No se encontraron secuencias de 2 eventos"
    
    from collections import Counter
    counter = Counter(bigrams)
    # Devolver tabla con top N secuencias más frecuentes
    top_sequences = counter.most_common(top_n)
    
    # Crear tabla formateada
    tabla = []
    for i, (secuencia, frecuencia) in enumerate(top_sequences, 1):
        porcentaje = (frecuencia / len(bigrams)) * 100
        tabla.append(f"{i}. {secuencia}: {frecuencia} veces ({porcentaje:.1f}%)")
    
    return "\n".join(tabla)

# --- Diccionario de funciones KPI ---
KPI_FUNCTIONS = {
    "drop_off_rate": kpi_drop_off_rate,
    "backtracking_rate": kpi_backtracking_rate,
    "average_path_length": kpi_average_path_length,
    "excessive_steps": kpi_excessive_steps,
    "avg_time_between_events": kpi_avg_time_between_events,
    "conversion_rate_per_path": kpi_conversion_rate_per_path,
    "unique_paths": kpi_unique_paths,
    "path_entropy": kpi_path_entropy,
    "most_common_entry_event": kpi_most_common_entry_event,
    "last_event_before_dropoff_or_conversion": kpi_last_event_before_dropoff_or_conversion,
    "most_common_exit_point": kpi_most_common_exit_point,
    "event_repetition_rate": kpi_event_repetition_rate,
    "average_click_depth": kpi_average_click_depth,
    "key_event_ratio": kpi_key_event_ratio,
    "total_time_on_path": kpi_total_time_on_path,
    "path_standard_deviation": kpi_path_standard_deviation,
    "first_key_action": kpi_first_key_action,
    "technical_bounce_rate": kpi_technical_bounce_rate,
    "Usuarios con más de X eventos antes de la conversión": kpi_usuarios_mas_x_eventos,
    "Sesiones con múltiples conversiones": kpi_sesiones_multiples_conversiones,
    "Rutas con múltiples conversiones": kpi_rutas_multiples_conversiones,
    "Número de pasos medio hasta la primera conversión": kpi_pasos_hasta_primera_conversion,
    "Número de pasos medio entre conversiones sucesivas": kpi_pasos_entre_conversiones,
    "Eventos más frecuentes": kpi_most_common_events,
    "Secuencias de 2 eventos más frecuentes": kpi_most_common_2grams,
}

def cargar_insights():
    ayuda = {}
    try:
        with open('.rules_insights', 'r') as f:
            contenido = f.read()
        # Parsear ayuda por KPI
        bloques = contenido.split('KPI: ')
        for bloque in bloques[1:]:
            lineas = bloque.split('\n')
            nombre = lineas[0].strip()
            texto = '\n'.join([l for l in lineas[1:] if l.strip()])
            ayuda[nombre] = texto
    except Exception:
        pass
    return ayuda

if __name__ == '__main__':
    with open('.rules_mapping_kpis', 'r') as f:
        mapping = yaml.safe_load(f)
    col_map = mapping['column_mapping']
    kpi_map = mapping['kpi_specific_mapping']
    df = pd.read_csv('data/import/ejemplo.csv')

    user_col = col_map.get('user_id', 'user_id')
    ruta_col = col_map.get('ruta', 'ruta')
    if ruta_col not in df.columns:
        ruta_col = col_map.get('Interactions', 'Interactions')

    print("\n[DEBUG] Columnas del DataFrame:", list(df.columns))

    ayuda_kpis = cargar_insights()

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
        else:
            resultados.append({
                "KPI": kpi_name,
                "Resultado": "No implementado en el script",
                "Evento de conversión": evento_conversion,
                "Guía de interpretación": ayuda_kpis.get(kpi_name, "")
            })

    # Mostrar resultados en formato tabla
    df_resultados = pd.DataFrame(resultados)
    print("\nRESULTADOS DE KPIs EN FORMATO TABLA:\n")
    print(df_resultados.to_string(index=False))

    # Exportar resultados a CSV
    export_path_csv = 'data/export/resultados_kpis.csv'
    df_resultados.to_csv(export_path_csv, index=False)
    print(f"\n[INFO] Resultados exportados a: {export_path_csv}")

    # Exportar resultados a Excel
    export_path_xlsx = 'data/export/resultados_kpis.xlsx'
    df_resultados.to_excel(export_path_xlsx, index=False)
    print(f"[INFO] Resultados exportados a: {export_path_xlsx}") 