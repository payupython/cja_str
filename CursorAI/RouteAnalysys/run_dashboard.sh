#!/bin/bash

# Script para ejecutar el dashboard de RouteAnalysys
# Autor: RouteAnalysys Team
# Fecha: $(date)

echo "🚀 Iniciando RouteAnalysys Dashboard..."

# Cambiar al directorio del script
cd "$(dirname "$0")"

# Verificar si el entorno virtual está activo
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "📦 Activando entorno virtual..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "❌ Error: No se encontró el entorno virtual 'venv'"
        echo "💡 Ejecuta primero: ./setup.sh"
        exit 1
    fi
fi

# Verificar dependencias
echo "🔍 Verificando dependencias..."
python -c "import dash, plotly, pandas, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Instalando dependencias faltantes..."
    pip install dash plotly pandas pyyaml dash-bootstrap-components
fi

# Verificar archivos necesarios
if [ ! -f "dashboard.py" ]; then
    echo "❌ Error: No se encontró dashboard.py"
    exit 1
fi

if [ ! -f ".rules_mapping_kpis" ]; then
    echo "❌ Error: No se encontró .rules_mapping_kpis"
    exit 1
fi

if [ ! -f "data/import/ejemplo.csv" ]; then
    echo "❌ Error: No se encontró data/import/ejemplo.csv"
    exit 1
fi

# Ejecutar dashboard
echo "🌐 Iniciando servidor web..."
echo "📊 Dashboard disponible en: http://localhost:8050"
echo "🛑 Para detener: Ctrl+C"
echo ""

python dashboard.py 