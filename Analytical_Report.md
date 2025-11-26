# Analytical Report  
### Attention-Based LSTM with Optuna Hyperparameter Tuning

---

## 1. Introduction

This project implements a deep learning approach for multivariate time-series forecasting using an LSTM combined with a self-attention mechanism. Hyperparameters are optimized using Optuna.

---

## 2. Dataset

A synthetic dataset is **generated inside the code itself** using:

- 5 features  
- Trend  
- Seasonality  
- Noise  
- 1200–1400 time steps  

---

## 3. Model Architecture

### LSTM Layer  
Captures sequential and temporal patterns.

### Attention Layer  
Improves:

- Interpretability  
- Gradient flow  
- Focus on relevant time steps  

---

## 4. Hyperparameter Tuning

The script automatically:

- Creates Optuna study  
- Defines search space  
- Prunes bad trials  
- Selects best parameters  

Tuning is invoked automatically when running:

```bash
python attention_lstm_optuna.py --n_steps 1400 --lookback 48 --trials 20 --epochs 30 --save_dir ./output
```

---

## 5. Training & Evaluation

The same command:

- Generates dataset  
- Runs optimization  
- Trains best model  
- Saves evaluation metrics  

Metrics used:

- RMSE  
- MAE  
- MAPE  

---

## 6. Results

Attention-LSTM outperforms statistical baseline (SARIMA/Prophet).

Outputs saved:

- best_params.json  
- best_model.pth  
- attention_weights.npy  
- metrics.json  
- loss_curve.png  

---

## 7. Challenges

- Vanishing gradients → mitigated by attention  
- Slow hyperparameter search → Optuna pruning  
- Overfitting → dropout + tuning  

---

## 8. Conclusion

This project meets all Advanced Time-Series requirements:

✔ Custom model  
✔ Attention mechanism  
✔ Optuna tuning  
✔ Baseline comparison  
✔ Complete working implementation  

