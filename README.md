# GhostMind

**The Cognition Core for Mini Von**

A modular, asynchronous AI agent framework focused on robust local reasoning, memory management, and structured cognition.
**IMPORTANT: This README.md is not complete, as I am still adding functions.

## ✨ Features

- **Modular Architecture**: Everything inherits from `GhostModule` with clean lifecycle management
- **Local-First LLM**: Designed for [llama.cpp](https://github.com/ggerganov/llama.cpp) server
- **Structured Reasoning Pipeline**: Intent → Decomposition → Planning → Reflection
- **Memory Systems**: Working memory + bridge to long-term memory (DreamCloud ready)
- **Event-Driven**: Powerful async event bus for loose coupling
- **Production-Ready**: Heartbeat monitoring, graceful shutdown, structured logging
- **Beautiful GUI**: Built with CustomTkinter (dark mode)

## 🏗️ Architecture

GhostMind/ ├── core/ # Runtime, config, modules, state, logging ├── cognition/ # Reasoning engines (Intent, Planning, Reflection…) ├── memory/ # Working memory + persistence bridge ├── llm/ # Local model client & server launcher ├── orchestration/ # Event bus ├── configs/ # YAML configuration ├── gui_app.py # Desktop interface (Mini Von) └── requirements.txt
### Core Components

- **`GhostModule`** — Base class for all subsystems
- **`Runtime`** — Central orchestrator and lifecycle manager
- **`CognitionPipeline`** — Main reasoning engine
- **`ModelClient`** — Async interface to llama.cpp
- **`EventBus`** — Pub/sub messaging
- **`DecisionLedger`** — Tracks decisions and reasoning history

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Timoune/GhostMind.git
cd GhostMind
2. Install dependencies
pip install -r requirements.txt
3. Set up your local LLM
	•	Download a GGUF model (e.g. Llama-3.2-3B-Instruct.Q5_K_M.gguf)
	•	Update configs/models.yaml with correct paths
4. Configure
Edit configuration files in the configs/ directory:
	•	runtime.yaml
	•	models.yaml
	•	cognition.yaml
	•	safety.yaml
5. Run Mini Von
python gui_app.py
📖 Documentation
	•	Architecture (coming soon)
	•	Configuration Guide (coming soon)
	•	Module Development (coming soon)
🛠️ Tech Stack
	•	Python 3.10+
	•	CustomTkinter — Modern GUI
	•	asyncio — Full async architecture
	•	structlog — Structured logging
	•	llama.cpp — Local inference server
	•	YAML — Configuration
🔧 Configuration
All settings are defined in the configs/ folder. Key areas:
	•	Models: LLM endpoints, context windows, temperatures
	•	Cognition: Prompt templates, reasoning parameters
	•	Memory: Context limits and retrieval settings
	•	Safety: Guardrails and content policies
	•	Runtime: Logging, heartbeat intervals
🎯 Current Status
Active Development — This is the cognitive backend for Mini Von, a personal AI desktop companion.
📝 License
All Rights Reserved
Copyright © 2026 Timoune. All rights reserved.
This software and its source code are proprietary. No part of this project may be copied, modified, distributed, or used commercially without explicit written permission from the copyright holder.
You may not:
	•	Use this software for commercial purposes
	•	Redistribute this software
	•	Create derivative works
	•	Remove or modify this license notice
For licensing inquiries or commercial use requests, please contact the repository owner.

Made with ❤️ for Mini Von
GhostMind — Because even ghosts need a mind.

Copyright © 2026 Timoune. All rights reserved.

All rights to this software and its source code are reserved by the copyright holder.
No permission is granted to copy, modify, distribute, sublicense, or use this software
for any purpose without explicit prior written consent.
