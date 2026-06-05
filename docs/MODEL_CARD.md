# Model Card — GridGuard AI

## 1. Model overview

**Project name:** GridGuard AI  
**Model family:** Electricity-load forecasting and anomaly screening  
**Primary task:** Next-hour electricity-load forecasting  
**Secondary task:** Abnormal-load-pattern screening  
**Current version:** Registry-driven champion–specialist release  
**Lifecycle stage:** Enterprise-oriented pre-production prototype

GridGuard AI uses a champion–specialist architecture:

- A **global champion model** provides broad fallback coverage.
- Approved **regional specialist models** are routed automatically when they demonstrate measurable improvement over the global model.
- Current approved specialists:
  - `market_AEMO`
  - `market_NYISO`

The system returns:

- Point forecast for next-hour electricity load.
- P10/P90 prediction interval.
- Operational lower and upper bounds.
- Anomaly flag and anomaly score.
- Routing metadata explaining which model served the request.

---

## 2. Intended use

GridGuard AI is intended for:

- Smart-grid operational monitoring.
- Electricity-load forecasting.
- Energy-demand planning.
- Regional grid analytics.
- Smart-building and campus energy intelligence.
- Data-center and industrial load-monitoring demonstrations.
- MLOps portfolio and interview demonstration.

The system is designed to help operators understand near-future demand behavior and model-routing decisions.

---

## 3. Out-of-scope use

GridGuard AI should **not** be used as a certified production control system without additional validation.

It should not be used for:

- Automatic grid-dispatch control.
- Safety-critical switching decisions.
- Regulatory settlement.
- Billing decisions.
- SLA enforcement.
- Emergency operations without human review.
- Long-horizon demand planning beyond its validated forecast horizon.

---

## 4. Model architecture

The current system includes:

| Component | Purpose |
|---|---|
| Global LightGBM model | Default next-hour load forecast |
| AEMO LightGBM specialist | Regional specialist for AEMO |
| NYISO LightGBM specialist | Regional specialist for NYISO |
| P10 LightGBM model | Lower quantile estimate |
| P90 LightGBM model | Upper quantile estimate |
| IsolationForest | Unsupervised anomaly screening |
| Registry-driven router | Selects approved specialist or global fallback |

The router is not hardcoded for specific markets. It reads the active model registry from:

```text
artifacts/model_registry.json

<!-- GRIDGUARD_MODEL_CARD_PRODUCTION_GOVERNANCE_START -->
## Production safeguards, maintenance, and deployment boundaries

### Recommended safeguards before real deployment

GridGuard AI should not be used as a mission-critical grid-control product without additional safeguards.

Recommended safeguards before real deployment:

- Human-in-the-loop review.
- Drift monitoring.
- Periodic retraining.
- Backtesting on new regions.
- Model registry approval workflow.
- Production-grade logging and alerting.
- Authentication and access control.
- Operational SLAs.

### Recommended production maintenance

Recommended production maintenance:

- Run CI tests on every commit.
- Rebuild the model registry after every specialist training run.
- Retrain models on a scheduled basis.
- Monitor WMAPE, MAE, RMSE, latency, and routing distribution.
- Investigate markets with high worst-series WMAPE.
- Promote specialists only when they beat the global model by the configured threshold.
- Review MongoDB prediction logs and Prometheus metrics regularly.
- Validate all required artifacts before model promotion.

### Current limitations

Current limitations:

- Anomaly detection is unsupervised.
- The system is validated for next-hour forecasting only.
- Real production deployment requires authentication, access control, drift monitoring, and operational SLAs.
- Model artifacts may need Git LFS or external model storage for large deployments.
- Weather and holiday context are not included in the current release.
- Prediction intervals should be calibrated further before safety-critical use.

For more detail, see `docs/PRODUCTION_READINESS.md`.
<!-- GRIDGUARD_MODEL_CARD_PRODUCTION_GOVERNANCE_END -->
