# Práctica Sistemas Inteligentes: Sistema Multiagente para detección de noticias falsas

## Instrucciones para crear el venv

Si no se dispone de uv, ejecutar el siguiente comando para instalarlo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Una vez instalado uv en el sistema, ejecutar el siguiente comando en el directorio `pade/` para crear el entorno virtual:

```bash
uv sync
```

Para activar el entorno, ejecutar en `pade/`:

```bash
source .venv/bin/activate
```

## Instrucciones para ejecutar el agente de visualización de PADE y verlo en web

Primero se iniciará el dashboad. Para ello, ejecutar el siguiente comando en el directorio `practica-ssii/`:

```bash
streamlit run fake_news_system.py
```

A continuación, se iniciará el runtime del agente PADE. Para ello, ejecutar el siguiente comando en el directorio `practica-ssii/`:

```bash
pade start-runtime fake_news_system.py
```

## Datos que llegan al agente visualizador

Los datos que llegan al agente visualizador son los siguientes:

- Titulo de la noticia
- Fuente de la noticia
- Si es fake o no
- Probabilidad de que sea fake (confianza)
- Fecha de la noticia

Los gráficos que se generan son los siguientes:

- Piechart que muestra la distribución de noticias fake y reales.
- Bar chart que muestra la cantidad de noticias fake y reales por fuente.

Con estos datos se genera gráficos y se almacenan en un archivo JSON: `fake_news_data.json`.
El GUI_Agent actúa como un puente (Bridge Agent). Gestiona la persistencia de los datos recibidos de la red multiagente para garantizar que no se pierda información durante los ciclos de renderizado de la interfaz de usuario, cumpliendo así con la responsabilidad de visualización del sistema.
