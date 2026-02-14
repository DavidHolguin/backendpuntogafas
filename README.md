# AI Order Worker — Punto Gafas CRM

Worker Python que procesa jobs de `ai_order_jobs` en Supabase, extrae datos de conversaciones y fórmulas ópticas usando Gemini 2.0 Flash, y crea pedidos borrador automáticamente.

## Arquitectura

```
ai_order_jobs (pending)
       │
       ▼
┌──────────────────────────────────────┐
│           Worker (polling 5s)         │
│                                      │
│  ┌─────────────────────────────────┐ │
│  │ Agent 1: Vision Extractor       │ │
│  │   → OCR fórmulas con Gemini     │ │
│  ├─────────────────────────────────┤ │
│  │ Agent 2: Conversation Analyzer  │ │
│  │   → Intenciones de compra       │ │
│  ├─────────────────────────────────┤ │
│  │ Agent 3: Catalog Matcher        │ │
│  │   → Cruce con lens_catalog &    │ │
│  │     products (fuzzy)            │ │
│  ├─────────────────────────────────┤ │
│  │ Agent 4: Order Builder          │ │
│  │   → Ensamblaje final            │ │
│  └─────────────────────────────────┘ │
│                                      │
│  DB Writer → prescriptions, orders,  │
│              order_items, customers,  │
│              notifications, messages  │
└──────────────────────────────────────┘
       │
       ▼
  orders (status='borrador')
```

## Variables de Entorno

| Variable | Requerida | Default | Descripción |
|----------|-----------|---------|-------------|
| `SUPABASE_URL` | ✅ | — | URL del proyecto Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | — | Service role key de Supabase |
| `GEMINI_API_KEY` | ✅ | — | API key de Google Gemini |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Modelo de Gemini a usar |
| `POLL_INTERVAL_SECONDS` | No | `5` | Intervalo de polling (segundos) |
| `JOB_TIMEOUT_SECONDS` | No | `180` | Timeout por job (segundos) |
| `MAX_RETRIES` | No | `2` | Reintentos máximos |

## Ejecución Local

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 4. Ejecutar
python -m app.worker
```

## Docker

```bash
# Build
docker build -t ai-order-worker .

# Run
docker run -d \
  --name ai-order-worker \
  -e SUPABASE_URL=https://xxx.supabase.co \
  -e SUPABASE_SERVICE_ROLE_KEY=eyJ... \
  -e GEMINI_API_KEY=AIza... \
  ai-order-worker
```

## Filosofía de Tolerancia a Fallos

El sistema **SIEMPRE** crea un pedido borrador, incluso con datos incompletos:

- **Sin fórmula** → Pedido sin prescription + warning
- **Sin imágenes** → Extrae solo del texto
- **Sin match en catálogo** → Item con `unit_price=0`, logística asigna
- **Sin cliente** → Usa `customer_id` del payload
- **Sin mensajes** → Extrae de notas internas
- **Sin nada** → Pedido vacío con `needs_manual_review=true`
- **Error en agente** → Continúa con los demás

Los pedidos se clasifican por completitud:
- **completo**: Todos los datos y precios presentes
- **parcial**: Faltan algunos datos o precios
- **minimo**: Solo datos del header, items pendientes

## Estructura del Proyecto

```
app/
├── __init__.py
├── __main__.py           # python -m app
├── config.py             # Variables de entorno (Pydantic Settings)
├── worker.py             # Loop de polling + signal handlers
├── agents/
│   ├── vision_extractor.py       # OCR de fórmulas ópticas
│   ├── conversation_analyzer.py  # Análisis de intenciones
│   ├── catalog_matcher.py        # Cruce con catálogos
│   ├── order_builder.py          # Ensamblaje del pedido
│   └── pipeline.py               # Orquestación secuencial
├── models/
│   ├── job.py             # ai_order_jobs + payload
│   ├── prescription.py    # rx_data, PrescriptionFound
│   ├── extraction.py      # Outputs intermedios
│   └── order_draft.py     # Resultado final
└── tools/
    ├── supabase_client.py # Cliente Supabase singleton
    ├── lens_catalog.py    # Búsqueda fuzzy en lens_catalog
    ├── products.py        # Búsqueda en products
    └── db_writer.py       # Inserts transaccionales
```
