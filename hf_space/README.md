---
title: GridGuard AI
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# GridGuard AI Hugging Face Space

This Space runs the GridGuard AI API and dashboard in one Docker container for demonstration.

Required secrets:

- `MONGODB_URI` optional but recommended
- `MONGODB_DB` optional, default `gridguard`

Before deployment, make sure trained artifacts exist in the `artifacts/` folder or modify the Dockerfile startup to download artifacts from a Hugging Face model repository.
