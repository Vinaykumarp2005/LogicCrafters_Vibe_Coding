# SEFS — Semantic Entropy File System

A self-organising file manager that uses semantic analysis to automatically cluster and organise files into meaningful folders. Drop files into a monitored directory and watch SEFS analyse their content, compute semantic similarity, and reorganise them into intelligently named folders — all visualised in real time through an interactive 2D force-directed graph.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![D3.js](https://img.shields.io/badge/D3.js-v7-orange?logo=d3.js)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Semantic Clustering** — TF-IDF or Sentence-Transformer embeddings + K-Means with automatic cluster-count selection via silhouette score.
- **OS-Level Folder Organisation** — Files are physically moved into semantically named directories derived from top keywords in each cluster.
- **Real-Time File Watching** — Watchdog-based monitor with debouncing detects new, modified, moved, and deleted files instantly.
- **Live 2D Visualisation** — Interactive D3.js force-directed graph with cluster hulls, drag/zoom, tooltips, and colour-coded nodes.
- **WebSocket Updates** — Flask-SocketIO pushes changes to the browser the moment the file system is reorganised.
- **PDF & Text Extraction** — Reads `.pdf` (via PyPDF2), `.txt`, `.md`, `.rst`, `.csv`, `.json`, and `.log` files with multi-encoding fallback.
- **REST API** — Endpoints for status, graph data, folder structure, configuration, and manual actions (rescan, reset).
- **Configurable** — CLI flags, JSON config file, or in-code defaults for every tunable parameter.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│              CLI parsing · Component init · Server start     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌───────────┐  ┌─────────────┐  ┌───────────┐
   │  Watcher  │  │ SyncManager │  │ WebServer  │
   │ (watchdog)│─▶│ (orchestra) │◀─│(Flask+SIO) │
   └───────────┘  └──────┬──────┘  └───────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌──────────────┐
        │Extractor │ │Semantic│ │FolderManager │
        │(PDF/text)│ │ Engine │ │  (OS moves)  │
        └──────────┘ └────────┘ └──────────────┘
```

**Pipeline:** File change → Content extraction → Embedding + Clustering → Folder reorganisation → WebSocket notification → UI update.

---

## Project Structure

```
Vibe-Coding/
├── main.py                      # Entry point & CLI
├── requirements.txt             # Python dependencies
├── monitored_root/              # Default directory to monitor
│   ├── <semantic-folder-1>/     # Auto-created by SEFS
│   ├── <semantic-folder-2>/
│   └── ...
└── src/
    └── sefs/
        ├── __init__.py
        ├── config.py            # Centralised configuration
        ├── extractor.py         # PDF & text content extraction
        ├── semantic_engine.py   # TF-IDF / Transformer embeddings + K-Means
        ├── folder_manager.py    # OS-level folder creation & file moves
        ├── sync_manager.py      # Central pipeline coordinator
        ├── watcher.py           # File system watcher (watchdog)
        ├── web_server.py        # Flask + SocketIO server & REST API
        ├── static/
        │   ├── css/
        │   │   └── style.css    # Dark-theme UI styles
        │   └── js/
        │       └── app.js       # D3.js graph + Socket.IO client
        └── templates/
            └── index.html       # Main UI template
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/Vibe-Coding.git
cd Vibe-Coding

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run

```bash
python main.py
```

Open **http://127.0.0.1:5000** in your browser.

Drop `.pdf`, `.txt`, `.md`, or other supported files into the `monitored_root/` folder and watch them get organised automatically.

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--root PATH` | `./monitored_root` | Directory to monitor |
| `--method {tfidf,transformer}` | `tfidf` | Embedding method |
| `--clusters N` | `0` (auto) | Fixed cluster count (0 = auto-detect) |
| `--threshold F` | `0.15` | Similarity threshold (0.0–1.0) |
| `--port N` | `5000` | Web UI port |
| `--host ADDR` | `127.0.0.1` | Web UI host |
| `--debug` | off | Enable debug logging |

**Examples:**

```bash
# Monitor a custom folder with transformer embeddings
python main.py --root "C:\Users\Me\Documents" --method transformer

# Fixed 5 clusters on port 8080
python main.py --clusters 5 --port 8080

# Debug mode
python main.py --debug
```

---

## REST API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/api/status` | GET | System status (file count, cluster count, method, etc.) |
| `/api/graph` | GET | Graph data — nodes & edges for D3.js visualisation |
| `/api/folders` | GET | Managed folder structure with file lists |
| `/api/config` | GET | Current configuration |
| `/api/rescan` | POST | Trigger a full rescan & re-cluster |
| `/api/reset` | POST | Move all files back to root & remove semantic folders |
| `/api/open_folder` | POST | Open a folder in the OS file explorer |

---

## WebSocket Events

| Event | Direction | Payload |
|---|---|---|
| `connect` | Client → Server | Server responds with initial `graph_update` + `folder_update` |
| `graph_update` | Server → Client | `{ nodes: [...], edges: [...] }` |
| `folder_update` | Server → Client | `{ folders: [...] }` |
| `status_update` | Server → Client | `{ file_count, cluster_count, ... }` |
| `request_rescan` | Client → Server | Triggers full pipeline |
| `request_reset` | Client → Server | Resets folder structure |

---

## Embedding Methods

### TF-IDF (default)
- **Fast** — no GPU required, works instantly on hundreds of files.
- Vocabulary size up to 5 000 features with English stop-word removal.
- Best for documents with distinct keyword profiles.

### Sentence Transformers
- Uses `all-MiniLM-L6-v2` (384-dim embeddings).
- **More accurate** for semantically similar documents that use different wording.
- Requires `sentence-transformers` and `torch` (installed via requirements).
- Enable with `--method transformer`.

---

## How Clustering Works

1. **Content Extraction** — Text is pulled from each file (PyPDF2 for PDFs, multi-encoding fallback for text files).
2. **Embedding** — Each document is converted to a vector via TF-IDF or Sentence Transformers.
3. **Optimal K Selection** — Silhouette scores are computed for K = 2…10; the K with the highest score is chosen.
4. **K-Means Clustering** — Files are assigned to clusters.
5. **Folder Naming** — The top-2 TF-IDF keywords from each cluster's combined text become the folder name (e.g., *"Learning & Machine"*, *"Energy & Solar"*).
6. **File Movement** — Files are physically moved into their cluster's folder; stale empty folders are cleaned up.

---

## Configuration File

Create `sefs_config.json` in the project root to persist settings:

```json
{
  "root_folder": "./monitored_root",
  "embedding_method": "tfidf",
  "num_clusters": 0,
  "similarity_threshold": 0.15,
  "watch_extensions": [".pdf", ".txt", ".md", ".rst", ".csv", ".json", ".log"],
  "host": "127.0.0.1",
  "port": 5000
}
```

---

## Supported File Types

| Extension | Extraction Method |
|---|---|
| `.pdf` | PyPDF2 page-by-page text extraction |
| `.txt` `.md` `.rst` `.log` | Raw text read with encoding fallback (UTF-8 → Latin-1 → CP1252 → ASCII) |
| `.csv` `.json` | Read as plain text |

---

## Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `flask-socketio` | Real-time WebSocket communication |
| `watchdog` | File system monitoring |
| `PyPDF2` | PDF text extraction |
| `scikit-learn` | TF-IDF vectorisation & K-Means clustering |
| `numpy` | Numerical operations |
| `sentence-transformers` | (Optional) Transformer-based embeddings |
| `torch` | (Optional) PyTorch backend for transformers |

---

## License

MIT
