# Dataset Card — GridGuard AI

## 1. Dataset overview

**Dataset used:** GloElecLoad / Worldwide Electricity Load Dataset  
**Task:** Electricity-load forecasting  
**Target:** `target_load_mw_next_hour`  
**Frequency after preprocessing:** Hourly  
**Primary unit:** MW  
**Use in project:** Training global and regional electricity-load forecasting models

The project originally explored household-level electricity data, but the final system uses a multi-market electricity-load dataset suitable for regional grid forecasting.

---

## 2. Dataset role in GridGuard AI

The dataset is used to train:

- Global champion forecasting model.
- AEMO specialist forecasting model.
- NYISO specialist forecasting model.
- P10/P90 quantile forecasting models.
- Unsupervised anomaly-screening model.

The processed dataset is saved locally as:

```text
data/processed/gloelecload_features.parquet