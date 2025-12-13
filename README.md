# AMP: The Agent Memory Protocol 🧠

**The Open Standard for Agentic Memory.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Native-purple)](https://modelcontextprotocol.io/introduction)
[![PyPI](https://img.shields.io/pypi/v/amp-memory.svg)](https://pypi.org/project/amp-memory/)

---

## The (Short) Story

I was tired of building AI agents that **forgot everything** the moment I closed the terminal. 

RAG (Retrieval Augmented Generation) is great for documents, but terrible for *experience*. It chunks text blindly, losing the narrative. When I asked my agents "Why did we decide this yesterday?", they gave me hallucinated nonsense.

So I built **AMP**. It's not just a database; it's a **Hippocampus** for your agents. It mimics the human brain's distinction between **Working Memory** (Short-term context) and **episodic Long-Term Memory**, giving your agents a continuous, evolving sense of self.

## Why developers are switching to AMP?

### 🌌 Galaxy View (Visualization)
Don't just *guess* what your agent knows. **See it.**
AMP comes with a stunning, 60fps local dashboard. Watch memories form constellations in real-time. Nodes cluster by semantic meaning—if two ideas are related, they physically move together.

![Galaxy View](assets/Galaxy.png)

---

### 🕸️ Force Mode (Physics)
Toggle to **Force Mode** to see the topological connections between your memories. It uses a physics simulation (D3.js) to show you how different memory clusters are "pulled" together by shared context.

![Force Mode](assets/Force.png)

---

### 🔍 Semantic Query
Stop guessing keywords. Query your agent's memory using natural language. I built a dedicated interface that not only finds relevant memories but shows you the **Relevance Score** (0-100%) so you know exactly why a memory was retrieved.

![Semantic Query](assets/sementic_query.png)

---

### 🔌 MCP Native (Plug & Play)
Built from day one for the **Model Context Protocol**.
*   **Claude Desktop**: Add AMP to your config, and Claude remembers you forever.
*   **Cursor**: Give your coding assistant persistent context of your project history.

### 🧠 The "3-Layer" Brain
I don't just dump text into a vector store. I structure it:
1.  **⚡ STM (Short Term)**: High-fidelity buffer. "What are we doing *right now*?"
2.  **📚 LTM (Long Term)**: Consolidated insights. "What did we learn last week?"
3.  **🕸️ Graph**: Connections between entities. "How is `function A` related to `bug B`?"

### 🏆 Best-in-Class Recall
I benchmarked AMP against the leading competitor (**Mem0**) on the complex **LoCoMo** dataset. The results weren't close.

| System | LLM Recall Accuracy | Why? |
| :--- | :--- | :--- |
| **AMP** | **81.6%** 🚀 | **Context-First**. Preserves the *narrative*. |
| Mem0 | 21.7% | **Extraction-First**. Aggressive summarization loses detail. |

---

## Quick Setup (30 seconds)

### 1. Install via `uv` (Recommended)
```bash
# Install the tool
uv tool install amp-memory

# Start the brain
amp serve
```

### 2. Or, Install via `pip`
```bash
pip install amp-memory
amp serve
```

### 3. Open the Dashboard
Visit `http://localhost:8000`. 
The interface is **Galaxy Mode** by default. Switch to **Force Mode** to see physics-based connections.

---

## 4. Connect to IDEs & Tools

AMP works native with **Antigravity**, **Cursor**, **VS Code Copilot**, and **Claude Desktop**.
Add this to your MCP configuration file (usually `mcp_config.json` or `claude_desktop_config.json`):

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

Now you can say:
> *"@amp remember that I am refactoring the login controller."* 
> *"@amp what was the last bug we fixed?"*

**It knows.**

---

## Roadmap 🗺️

*   [x] **Galaxy View**: Visual Semantic Space.
*   [x] **Graph API**: D3.js powered visualization.
*   [x] **Semantic Search**: Vector-based relevance sorting.
*   [ ] **Cloud Sync**: Sync memories across devices.
*   [ ] **Multi-Agent Swarm**: Shared memory for agent teams.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=akshayaggarwal99/amp&type=Date)](https://star-history.com/#akshayaggarwal99/amp&Date)

---

<p align="center">
  Made with ❤️ by Akshay.
</p>
