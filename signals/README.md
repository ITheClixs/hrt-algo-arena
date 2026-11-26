# signals/

Team signal exchange: one parquet file per idea, `signals/<author>/<name>.parquet`, holding out-of-fold
predictions for the in-sample sessions and predictions for the out-of-sample sessions.
Write them with `arena.signals.export_signal`; read and blend them with `load_signals`, `blend_cv` and `blend_predict`.
