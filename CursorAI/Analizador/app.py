import streamlit as st
import pandas as pd
import io
import numpy as np
import re

st.set_page_config(page_title="Analizador de Rutas", layout="wide")

st.title("🔎 Analizador de Rutas de Navegación")

# --- Sidebar: Configuración de entrada ---
st.sidebar.header("Configuración de entrada")
modo_entrada = st.sidebar.radio("Modo de entrada", ["Texto directo", "Archivo CSV"], index=1)
separador = st.sidebar.text_input("Separador de eventos", value="-")

# Inputs para textos de conversión
st.sidebar.markdown("**Textos que indican conversión (pueden contener números, letras, @, etc.)**")
texto_conversion1 = st.sidebar.text_input("Texto conversión 1", value="purchase")
texto_conversion2 = st.sidebar.text_input("Texto conversión 2", value="", help="Opcional")
texto_conversion3 = st.sidebar.text_input("Texto conversión 3", value="", help="Opcional")
conversion_keywords = [t for t in [texto_conversion1, texto_conversion2, texto_conversion3] if t.strip()]

# Nuevo: Selector de separador de CSV
separador_csv = ","
if modo_entrada == "Archivo CSV":
    separador_csv = st.sidebar.text_input("Separador de columnas en CSV (sugerido ,)", value=separador_csv, help="Por defecto se sugiere coma (,), pero puedes escribir cualquier otro separador.")

# Selector de tipo de filtro
opciones_filtro = [
    "Empieza por (exacto)",
    "Empieza por (contiene texto)",
    "Contiene en cualquier parte"
]
tipo_filtro = st.sidebar.selectbox("Tipo de filtro de evento inicial", opciones_filtro)

# Input para filtrar rutas por evento inicial
filtro_inicio = st.sidebar.text_input("Filtrar rutas por evento (puede tener números, letras, espacios)", value="")

# --- Limpieza y parsing de rutas ---
def parsear_ruta(ruta, sep):
    # Si el separador es coma, separar por coma y eliminar espacios alrededor
    if sep.strip() == ",":
        pasos = [p.strip() for p in ruta.split(",") if p.strip()]
    else:
        pasos = [p.strip() for p in ruta.split(sep) if p.strip()]
    return pasos

# --- Entrada de datos ---
rutas = []
df_csv = None
columna_rutas = None

if modo_entrada == "Texto directo":
    texto = st.text_area("Pega aquí las rutas de navegación (una por línea)")
    if texto:
        rutas = [line.strip() for line in texto.splitlines() if line.strip()]
else:
    archivo = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if archivo:
        try:
            df_csv = pd.read_csv(archivo, sep=separador_csv)
            posibles_columnas = [col for col in df_csv.columns if any(x in col.lower() for x in ["route", "path", "navigation", "journey"])]
            if not posibles_columnas:
                posibles_columnas = list(df_csv.columns)
            columna_rutas = st.selectbox("Selecciona la columna de rutas", posibles_columnas)
            # Nuevo: Selección de formato de columnas
            st.markdown("**Selecciona el formato de cada columna:**")
            formatos_columnas = {}
            for col in df_csv.columns:
                tipo = st.selectbox(f"Formato para '{col}'", ["string", "numeric"], key=f"formato_{col}")
                formatos_columnas[col] = tipo
            # Aplicar conversión de tipo
            for col, tipo in formatos_columnas.items():
                if tipo == "numeric":
                    df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce")
                else:
                    df_csv[col] = df_csv[col].astype(str)
            # Añadir columna de número de eventos al DataFrame principal
            df_csv['numero_eventos'] = df_csv[columna_rutas].astype(str).apply(lambda x: len(parsear_ruta(x, separador)))
            # Añadir columna booleana de conversión al DataFrame principal
            def es_convertida_str(r):
                pasos = parsear_ruta(str(r), separador)
                return any(any(kw.lower() in paso.lower() for kw in conversion_keywords) for paso in pasos)
            df_csv['es_conversion'] = df_csv[columna_rutas].astype(str).apply(es_convertida_str)
            # Crear DataFrame auxiliar para métricas
            df_metricas = pd.DataFrame({
                'ruta': df_csv[columna_rutas].astype(str),
                'numero_eventos': df_csv['numero_eventos'],
                'es_conversion': df_csv['es_conversion']
            })
            st.info(f"Archivo: {archivo.name} | Filas: {len(df_csv)}")
            # Mostrar todas las columnas y permitir filtrado interactivo
            st.dataframe(df_csv, use_container_width=True)
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}\nVerifica el separador y el formato del archivo.")

# Aplicar filtro a los DataFrames si se indica un filtro
if 'df_csv' in locals() and df_csv is not None and not df_csv.empty and filtro_inicio.strip():
    if tipo_filtro == "Empieza por (exacto)":
        df_csv = df_csv[df_csv[columna_rutas].astype(str).apply(lambda x: parsear_ruta(x, separador)[0] if parsear_ruta(x, separador) else "") == filtro_inicio.strip()]
    elif tipo_filtro == "Empieza por (contiene texto)":
        df_csv = df_csv[df_csv[columna_rutas].astype(str).apply(lambda x: filtro_inicio.strip().lower() in (parsear_ruta(x, separador)[0].lower() if parsear_ruta(x, separador) else ""))]
    elif tipo_filtro == "Contiene en cualquier parte":
        df_csv = df_csv[df_csv[columna_rutas].astype(str).apply(lambda x: any(filtro_inicio.strip().lower() in paso.lower() for paso in parsear_ruta(x, separador)))]

if 'df_metricas' in locals() and df_metricas is not None and not df_metricas.empty and filtro_inicio.strip():
    if tipo_filtro == "Empieza por (exacto)":
        df_metricas = df_metricas[df_metricas['ruta'].apply(lambda x: parsear_ruta(x, separador)[0] if parsear_ruta(x, separador) else "") == filtro_inicio.strip()]
    elif tipo_filtro == "Empieza por (contiene texto)":
        df_metricas = df_metricas[df_metricas['ruta'].apply(lambda x: filtro_inicio.strip().lower() in (parsear_ruta(x, separador)[0].lower() if parsear_ruta(x, separador) else ""))]
    elif tipo_filtro == "Contiene en cualquier parte":
        df_metricas = df_metricas[df_metricas['ruta'].apply(lambda x: any(filtro_inicio.strip().lower() in paso.lower() for paso in parsear_ruta(x, separador)))]

# --- Depuración: mostrar parsing de rutas ---
# (Eliminado para limpiar la interfaz)

# --- Mostrar resumen de datos cargados ---
if 'df_metricas' in locals() and df_metricas is not None and not df_metricas.empty:
    total_rutas = len(df_metricas)
else:
    total_rutas = len(rutas)
st.subheader("Resumen de datos cargados")
st.write(f"Total de rutas: {total_rutas}")
if total_rutas > 0:
    # --- Métricas clave ---
    st.subheader("Métricas clave")
    # Calcular métricas SIEMPRE sobre el DataFrame auxiliar df_metricas
    if 'df_metricas' in locals() and df_metricas is not None and not df_metricas.empty:
        df_metricas['numero_eventos'] = pd.to_numeric(df_metricas['numero_eventos'], errors='coerce')
        serie_eventos = df_metricas['numero_eventos'].dropna()
        longitudes = []
        try:
            longitudes = serie_eventos.astype(int).tolist()
            if not serie_eventos.empty:
                serie_eventos_float = serie_eventos.astype(float)
                min_val = serie_eventos_float.min()
                max_val = serie_eventos_float.max()
                if isinstance(min_val, (int, float, np.integer, np.floating)):
                    longitud_min = int(min_val)
                else:
                    longitud_min = 0
                if isinstance(max_val, (int, float, np.integer, np.floating)):
                    longitud_max = int(max_val)
                else:
                    longitud_max = 0
                longitud_promedio = float(serie_eventos_float.mean())
            else:
                longitud_min = 0
                longitud_max = 0
                longitud_promedio = 0
        except Exception:
            longitud_min = 0
            longitud_max = 0
            longitud_promedio = 0
            longitudes = []
        # Calcular conversiones y tasa usando la columna booleana
        num_convertidas = int(df_metricas['es_conversion'].sum())
        num_abandonadas = int((~df_metricas['es_conversion']).sum())
        tasa_conversion = num_convertidas / len(df_metricas) if len(df_metricas) > 0 else 0
    else:
        longitudes = [len(parsear_ruta(r, separador)) for r in rutas if r]
        longitud_promedio = sum(longitudes) / len(longitudes) if longitudes else 0
        longitud_min = min(longitudes) if longitudes else 0
        longitud_max = max(longitudes) if longitudes else 0
        num_convertidas = 0
        num_abandonadas = 0
        tasa_conversion = 0

    # Sección temporal de depuración para comparar con los datos del usuario
    if longitudes:
        st.info(f"[Depuración] Máximo eventos detectados: {longitud_max} | Promedio: {longitud_promedio:.4f} | Mínimo: {longitud_min}")
        # Mostrar ejemplos de rutas con máximo y mínimo número de eventos
        if 'df_metricas' in locals() and df_metricas is not None and not df_metricas.empty:
            st.write("### Ejemplos de rutas con máximo número de eventos:")
            st.dataframe(df_metricas[df_metricas['numero_eventos'] == longitud_max].head(5), use_container_width=True)
            st.write("### Ejemplos de rutas con mínimo número de eventos:")
            st.dataframe(df_metricas[df_metricas['numero_eventos'] == longitud_min].head(5), use_container_width=True)
    def es_convertida(r):
        return any(any(kw.lower() in paso.lower() for kw in conversion_keywords) for paso in r)
    rutas_convertidas = [r for r in rutas if es_convertida(r)]
    rutas_abandonadas = [r for r in rutas if not es_convertida(r)]
    # tasa_conversion = len(rutas_convertidas) / total_rutas if total_rutas else 0 # This line is now redundant as it's calculated above

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total rutas", total_rutas)
    col2.metric("Longitud mín.", f"{longitud_min}")
    col3.metric("Longitud máx.", f"{longitud_max}")
    col4.metric("Longitud prom.", f"{longitud_promedio:.2f}")
    col5.metric("Tasa conversión", f"{tasa_conversion*100:.1f}%")
    col6.metric("Convertidas", num_convertidas)
    col7.metric("Abandonadas", num_abandonadas)

# --- Sección de análisis de patrones y visualizaciones ---
st.subheader("Análisis de patrones y visualizaciones")
if 'df_metricas' in locals() and df_metricas is not None and not df_metricas.empty:
    # Selector para número de resultados a mostrar
    top_n = st.number_input("¿Cuántos resultados mostrar en las tablas de eventos?", min_value=1, max_value=100, value=10, step=1)
    # Selector para tamaño de n-grama
    n_grama = st.number_input("Tamaño de secuencia de eventos consecutivos (n-grama)", min_value=2, max_value=10, value=2, step=1)

    # Páginas/eventos más visitados (conteo global de todos los eventos)
    from collections import Counter
    eventos = []
    for ruta in df_metricas['ruta']:
        eventos.extend(parsear_ruta(ruta, separador))
    conteo_eventos = Counter(eventos)
    eventos_mas_visitados = conteo_eventos.most_common(top_n)
    st.write(f"**Eventos más visitados (Top {top_n}):**")
    st.table(pd.DataFrame(eventos_mas_visitados, columns=["Evento", "Frecuencia"]))

    # Selector para Top X de entradas y salidas
    top_entradas_salidas = st.number_input("¿Cuántos puntos de entrada/salida mostrar?", min_value=1, max_value=100, value=5, step=1)

    # Puntos de entrada y salida
    entradas = [parsear_ruta(r, separador)[0] for r in df_metricas['ruta'] if parsear_ruta(r, separador)]
    salidas = [parsear_ruta(r, separador)[-1] for r in df_metricas['ruta'] if parsear_ruta(r, separador)]
    st.write(f"**Puntos de entrada más frecuentes (Top {top_entradas_salidas}):**")
    st.table(pd.DataFrame(Counter(entradas).most_common(top_entradas_salidas), columns=["Entrada", "Frecuencia"]))
    st.write(f"**Puntos de salida más frecuentes (Top {top_entradas_salidas}):**")
    st.table(pd.DataFrame(Counter(salidas).most_common(top_entradas_salidas), columns=["Salida", "Frecuencia"]))

    # Bucles: rutas donde algún evento se repite
    def tiene_bucle(r):
        eventos = parsear_ruta(r, separador)
        return len(eventos) != len(set(eventos))
    df_metricas['tiene_bucle'] = df_metricas['ruta'].apply(tiene_bucle)
    num_bucles = df_metricas['tiene_bucle'].sum()
    st.write(f"**Rutas con bucles:** {num_bucles} ({num_bucles/len(df_metricas)*100:.1f}%)")
    st.write("Ejemplo de rutas con bucles:")
    st.dataframe(df_metricas[df_metricas['tiene_bucle']].head(top_n), use_container_width=True)

    # Anomalías: rutas con conversión y más eventos después
    def es_anomalia(r):
        eventos = parsear_ruta(r, separador)
        indices = [i for i, paso in enumerate(eventos) if any(kw.lower() in paso.lower() for kw in conversion_keywords)]
        return bool(indices) and indices[-1] < len(eventos) - 1
    df_metricas['es_anomalia'] = df_metricas['ruta'].apply(es_anomalia)
    num_anomalias = df_metricas['es_anomalia'].sum()
    st.write(f"**Rutas anómalas (con conversión y más eventos después):** {num_anomalias}")
    st.dataframe(df_metricas[df_metricas['es_anomalia']].head(top_n), use_container_width=True)

    # Eventos más repetidos en cada ruta (tabla resumen)
    def evento_mas_repetido(r):
        eventos = parsear_ruta(r, separador)
        if not eventos:
            return None
        return Counter(eventos).most_common(1)[0][0]
    df_metricas['evento_mas_repetido'] = df_metricas['ruta'].apply(evento_mas_repetido)
    st.write(f"**Eventos más repetidos por ruta (Top {top_n}):**")
    st.table(df_metricas['evento_mas_repetido'].value_counts().head(top_n).reset_index().rename(columns={'index':'Evento','evento_mas_repetido':'Frecuencia'}))

    # Pares de eventos más comunes (ahora n-gramas)
    ngramas = []
    for ruta in df_metricas['ruta']:
        eventos = parsear_ruta(ruta, separador)
        ngramas += [tuple(eventos[i:i+n_grama]) for i in range(len(eventos)-n_grama+1)]
    conteo_ngramas = Counter(ngramas)
    ngramas_mas_comunes = conteo_ngramas.most_common(top_n)
    st.write(f"**Secuencias de {n_grama} eventos consecutivos más comunes (Top {top_n}):**")
    st.table(pd.DataFrame([(" → ".join(ng), freq) for ng, freq in ngramas_mas_comunes], columns=[f"{n_grama}-grama de eventos", "Frecuencia"]))

    # Función para limpiar el evento (eliminar prefijos numéricos, espacios y símbolos)
    def limpiar_evento(ev):
        if not isinstance(ev, str):
            return ""
        return re.sub(r'^\W*\d*\s*', '', ev).strip()

    # Obtener eventos parseados únicos (limpios)
    eventos_parseados = [limpiar_evento(ev) for ev in eventos]
    eventos_unicos_parseados = pd.Series(eventos_parseados).value_counts().head(30).index.tolist() if eventos_parseados else []
    evento_abandono = st.selectbox("Selecciona el evento para analizar abandono tras ese evento (parseado)", eventos_unicos_parseados, index=eventos_unicos_parseados.index('search') if 'search' in eventos_unicos_parseados else 0)

    # Abandono tras evento seleccionado: rutas que terminan en ese evento parseado y no tienen más eventos después
    def abandono_tras_evento_parseado(r):
        eventos_ruta = parsear_ruta(r, separador)
        if not eventos_ruta:
            return False
        evento_final = limpiar_evento(eventos_ruta[-1])
        return evento_final == evento_abandono
    df_metricas['abandono_tras_evento'] = df_metricas['ruta'].apply(abandono_tras_evento_parseado)
    num_abandono_evento = df_metricas['abandono_tras_evento'].sum()
    st.write(f"**Rutas con abandono tras '{evento_abandono}':** {num_abandono_evento} ({num_abandono_evento/len(df_metricas)*100:.1f}%)")
    st.dataframe(df_metricas[df_metricas['abandono_tras_evento']].head(top_n), use_container_width=True)

# --- Placeholder para análisis y visualización ---
st.subheader("Análisis y visualización (en desarrollo)")
st.info("Aquí se mostrarán las métricas, patrones y visualizaciones del análisis de rutas.") 