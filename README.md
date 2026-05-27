# GhostMind

**The Cognition Core for Mini Von**

A modular, asynchronous AI agent framework focused on robust local reasoning, memory management, and structured cognition.

![Mini Von](https://via.placeholder.com/800x200/1a1a2e/00ff9f?text=Mini+Von+-+GhostMind) <!-- Replace with actual screenshot later -->

## ✨ Features

- **Modular Architecture**: Everything inherits from `GhostModule` with clean lifecycle management
- **Local-First LLM**: Designed for [llama.cpp](https://github.com/ggerganov/llama.cpp) server
- **Structured Reasoning Pipeline**: Intent → Decomposition → Planning → Reflection
- **Memory Systems**: Working memory + bridge to long-term memory (DreamCloud ready)
- **Event-Driven**: Powerful async event bus for loose coupling
- **Production-Ready**: Heartbeat monitoring, graceful shutdown, structured logging
- **Beautiful GUI**: Built with CustomTkinter (dark mode)

## 🏗️ Architecture