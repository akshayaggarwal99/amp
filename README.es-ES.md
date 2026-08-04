

# AMP: El Protocolo de Memoria para Agentes 🧠

**El Estándar Abierto para la Memoria de Agentes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Native-purple)](https://modelcontextprotocol.io/introduction)
[![PyPI](https://img.shields.io/pypi/v/amp-memory.svg)](https://pypi.org/project/amp-memory/)

---

## La (Breve) Historia

Estaba cansado de crear agentes de IA que **olvidaban todo** en el momento en que cerraba la terminal. 

RAG (Generación Aumentada con Recuperación) es excelente para documentos, pero terrible para la *experiencia*. Divide el texto a ciegas, perdiendo la narrativa. Cuando le preguntaba a mis agentes "¿Por qué decidimos esto ayer?", me daban respuestas alucinadas y sin sentido.

Así que creé **AMP**. No es solo una base de datos; es un **Hipocampo** para tus agentes. Imita la distinción del cerebro humano entre la **Memoria de Trabajo** (contexto a corto plazo) y la **Memoria a Largo Plazo Episódica**, dándole a tus agentes un sentido de identidad continuo y en evolución.

## ¿Por qué los desarrolladores están cambiando a AMP?

### 🌌 Vista Galaxia (Visualización)
No te limites a *adivinar* lo que sabe tu agente. **Míralo.**
AMP incluye un impresionante panel local a 60 fps. Observa cómo los recuerdos forman constelaciones en tiempo real. Los nodos se agrupan por significado semántico: si dos ideas están relacionadas, se mueven juntas físicamente.

![Galaxy View](assets/Galaxy.png)

---

### 🕸️ Modo Fuerza (Física)
Activa el **Modo Fuerza** para ver las conexiones topológicas entre tus recuerdos. Utiliza una simulación física (D3.js) para mostrarte cómo diferentes clústeres de memoria son "atraídos" juntos por un contexto compartido.

![Force Mode](assets/Force.png)

---

### 🔍 Consulta Semántica
Deja de adivinar palabras clave. Consulta la memoria de tu agente usando lenguaje natural. Creé una interfaz dedicada que no solo encuentra recuerdos relevantes, sino que te muestra la **Puntuación de Relevancia** (0-100%) para que sepas exactamente por qué se recuperó un recuerdo.

![Semantic Query](assets/sementic_query.png)

---

### 🔌 Nativo para MCP (Conectar y Usar)
Diseñado desde el primer día para el **Model Context Protocol**.
*   **Claude Desktop**: Añade AMP a tu configuración y Claude te recordará para siempre.
*   **Cursor**: Dale a tu asistente de código un contexto persistente sobre el historial de tu proyecto.

### 🧠 El Cerebro de "3 Capas"
No solo vuelco texto en un almacén vectorial. Lo estructuro:
1.  **⚡ STM (Corto Plazo)**: Búfer de alta fidelidad. "¿Qué estamos haciendo *ahora mismo*?"
2.  **📚 LTM (Largo Plazo)**: Conocimientos consolidados. "¿Qué aprendimos la semana pasada?"
3.  **🕸️ Grafo**: Conexiones entre entidades. "¿Cómo se relaciona la `función A` con el `error B`?"

### 🏆 Recuperación de Clase Mundial
Hice pruebas comparativas (benchmark) de AMP contra el principal competidor (**Mem0**) en el complejo conjunto de datos **LoCoMo**. Los resultados no fueron ajustados.

| Sistema | Precisión de Recuperación del LLM | ¿Por qué? |
| :--- | :--- | :--- |
| **AMP** | **81.6%** 🚀 | **Contexto Primero**. Preserva la *narrativa*. |
| Mem0 | 21.7% | **Extracción Primero**. El resumen agresivo pierde detalles. |

---

## Configuración Rápida (30 segundos)

### 1. Instalar con `uv` (Recomendado)
```bash
# Instalar la herramienta
uv tool install amp-memory

# Iniciar el cerebro
amp serve
```

### 2. O instalar con `pip`
```bash
pip install amp-memory
amp serve
```

### 3. Abrir el Panel
Visita `http://localhost:8000`. 
La interfaz es **Modo Galaxia** por defecto. Cambia a **Modo Fuerza** para ver conexiones basadas en física.

---

## 4. Conectar a IDEs y Herramientas

AMP funciona de forma nativa con **Antigravity**, **Cursor**, **VS Code Copilot** y **Claude Desktop**.
Agrega esto a tu archivo de configuración de MCP (generalmente `mcp_config.json` o `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "amp-memory": {
      "command": "uv",
      "args": ["tool", "run", "amp-memory", "serve"],
      "env": {
        "PYTHONPATH": "."
      }
    }
  }
}
```

Ahora puedes decir:
> "*@amp recuerda que estoy refactorizando el controlador de inicio de sesión.*" 
> "*@amp ¿cuál fue el último error que arreglamos?*"

**Lo sabe.**

---

## Hoja de Ruta 🗺️

*   [x] **Vista Galaxia**: Espacio Semántico Visual.
*   [x] **API de Grafos**: Visualización impulsada por D3.js.
*   [x] **Búsqueda Semántica**: Ordenamiento de relevancia basado en vectores.
*   [ ] **Sincronización en la Nube**: Sincronizar recuerdos entre dispositivos.
*   [ ] **Enjambre Multiagente**: Memoria compartida para equipos de agentes.

## Historial de Estrellas

[![Star History Chart](https://api.star-history.com/svg?repos=akshayaggarwal99/amp&type=Date)](https://star-history.com/#akshayaggarwal99/amp&Date)

---

<p align="center">
  Hecho con ❤️ por Akshay.
</p>
