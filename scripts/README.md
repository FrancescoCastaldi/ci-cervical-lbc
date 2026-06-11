# scripts/ — Script di Esecuzione

Pipeline completa per il preprocessing, esecuzione di tutti i metodi, e generazione dei risultati.

## Utilità per l'esame

- **Riproducibilità**: eseguendo gli script in ordine si rigenerano tutti i risultati
- **Automazione**: ogni metodo ha il suo script dedicato
- **Orchestrazione**: lo script `plot_results.py` raccoglie i risultati di tutti i metodi

## Pipeline

```bash
# 1. Preprocessing (una volta sola)
python scripts/preprocess.py

# 2. Esecuzione metodi (indipendenti, ordine arbitrario)
python scripts/run_tv.py          # Total Variation
python scripts/run_unet.py        # UNet
python scripts/run_diffpir.py     # DiffPIR

# 3. Confronto finale
python scripts/plot_results.py
```

## File

| Script | Descrizione |
|---|---|
| `preprocess.py` | Genera split files e immagini degradate dal dataset raw |
| `run_tv.py` | Esegue TV su test set, salva metriche e qualitative |
| `run_unet.py` | Training + evaluation UNet, salva modello e risultati |
| `run_diffpir.py` | Esegue DiffPIR su test set, salva metriche e qualitative |
| `plot_results.py` | Carica metriche di tutti i metodi, genera grafico comparativo |
| `gen_tv_qual.py` | Script helper — genera solo qualitative TV |
| `eval_unet.py` | Script helper — solo evaluation UNet (modello già addestrato) |
