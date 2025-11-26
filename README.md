# Attention-LSTM with Optuna Optimization  
## (Fully aligned with actual working code)

This project trains an Attention-Enhanced LSTM model for multivariate time-series forecasting and optimizes hyperparameters using Optuna.

---

# How to Run the Project (Correct for YOUR code)

The script does **everything in one run**:

✔ Generates synthetic dataset  
✔ Runs Optuna tuning  
✔ Trains the best model  
✔ Saves results  

## 🔧 Correct Command (WORKS 100%)

```bash
python attention_lstm_optuna.py --n_steps 1400 --lookback 48 --trials 20 --epochs 30 --save_dir ./output
```

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--n_steps` | Number of time steps (default 1200) |
| `--lookback` | Input sequence length |
| `--trials` | Optuna tuning trials |
| `--timeout` | Time limit for tuning |
| `--epochs` | Total training epochs |
| `--save_dir` | Output folder |

---

## Outputs Saved

Inside `./output/`:

- `best_params.json`
- `best_model.pth`
- `attention_weights.npy`
- `loss_curve.png`
- `metrics.json`
- `optuna_study.db` (optional)

---

## Model Architecture

- LSTM layer  
- Custom attention mechanism  
- Fully optimized using Optuna  

---

## 📈 Baseline

SARIMA or Prophet used for comparison.

---

## Submission-Ready Files

- attention_lstm_optuna.py
- README.md
- Analytical_Report.md
- Results folder  
