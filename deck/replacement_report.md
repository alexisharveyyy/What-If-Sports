# Deck metrics replacement report

Seed: **42**  
Dataset: **140,374** player-weeks across **10,000** unique players (train 6,997 / val 1,497 / test 1,506, stratified by NIL tier).

## Baselines (test split, last-week-per-player)

| Model | Accuracy | Macro F1 | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Logistic regression | 0.7895 | 0.7075 | — | — | — |
| Random forest | 0.7835 | 0.6781 | $51,939 | $82,374 | 0.9221 |
| XGBoost | 0.7829 | 0.7103 | $52,544 | $84,166 | 0.9187 |

## Sequence models (test split, full-season inputs)

| Model | Accuracy | Macro F1 | MAE | RMSE | R² | ECE before | ECE after |
|---|---|---|---|---|---|---|---|
| BiLSTM + Attention | 0.7703 | 0.6379 | $63,657 | $107,129 | 0.8682 | 0.020 | 0.021 |
| Transformer (reg-only) | — | — | $61,955 | $103,490 | 0.8770 | — | — |
| Transformer (cls-only) | 0.7696 | 0.6036 | — | — | — | 0.027 | — |
| Transformer (multi-task) | 0.7716 | 0.6079 | $64,129 | $104,948 | 0.8735 | 0.029 | 0.021 |

## Volatile cohort (top-CV quartile)

- CV threshold: 0.2122  
- Volatile players in test split: **377**  
- XGBoost MAE: **$60,237**  
- BiLSTM + Attention MAE: **$76,365**  
- Δ vs. XGBoost: **+26.77%**

## Temperature scaling (15 bins)

| Model | T | ECE before | ECE after | Reduction |
|---|---|---|---|---|
| Transformer (multi-task) | 0.878 | 0.029 | 0.021 | 28.82% |
| BiLSTM + Attention | 0.930 | 0.020 | 0.021 | -8.92% |

## Feature importance (XGBoost regressor, top-6 by gain)

| Rank | Feature | Gain | Normalized |
|---|---|---|---|
| 1 | `program_tier` | 6832990453760.00 | 100.0/100 |
| 2 | `currently_injured` | 132385603584.00 | 1.94/100 |
| 3 | `performance_score` | 97077862400.00 | 1.42/100 |
| 4 | `engagement_rate` | 49052024832.00 | 0.72/100 |
| 5 | `market_size_score` | 35418460160.00 | 0.52/100 |
| 6 | `social_media_followers` | 26343512064.00 | 0.39/100 |

## Sanity check

- Multi-task transformer MAE ($64,129) is **higher** than XGBoost ($52,544). The transformer should usually win — investigate before presenting.
- BiLSTM ECE rose after temperature scaling (0.020 -> 0.021).
- BiLSTM **lost** to XGBoost on volatile players (+26.77% vs. expectation of negative).
- Tiers with <7% of player-weeks may inflate macro F1 noise: elite
