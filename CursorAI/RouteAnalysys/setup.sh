#!/bin/bash
# Script para activar el entorno virtual e instalar dependencias
cd "$(dirname "$0")"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt 