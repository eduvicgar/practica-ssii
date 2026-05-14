# practica-ssii

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
