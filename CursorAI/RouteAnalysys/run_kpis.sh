#!/bin/bash
# Script para ejecutar test_kpis.py, activando el entorno virtual si no está activo

cd "$(dirname "$0")"

if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "[INFO] El entorno virtual NO está activado. Activando..."
  source venv/bin/activate
fi

if [[ -z "$VIRTUAL_ENV" ]]; then
  echo "[ERROR] No se pudo activar el entorno virtual."
  exit 1
else
  echo "[OK] Entorno virtual activado: $VIRTUAL_ENV"
  python3 test_kpis.py
fi 