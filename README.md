# What-If Sports NIL Simulator

A full-stack NIL (Name, Image, and Likeness) probability simulator for collegiate athletes. Create hypothetical player profiles with custom performance inputs and receive real-time forecasted NIL valuations and probability timelines.

## Architecture

```
Data Layer → Feature Engineering → ML Models → Simulator Engine → FastAPI → Dashboard
(scraper)    (pipeline/)           (models/)   (simulator/)       (app/)    (frontend/)
```

**Models**: Bidirectional LSTM and Transformer encoder with multi-task heads (NIL tier classification + valuation regression).

**Evaluation**: Accuracy, F1, MAE, RMSE, R², Expected Calibration Error (ECE), and volatility robustness testing.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
cd what-if-sports
python data/sample/generate_sample.py

# 3. Run preprocessing + feature engineering
python pipeline/preprocess.py
python pipeline/features.py

# 4. Train baselines
python models/baseline.py

# 5. Train LSTM model
python train/train.py --model lstm --epochs 50

# 6. Start the API server
uvicorn app.main:app --reload --port 8000

# 7. Open the dashboard
# Visit http://localhost:8000/frontend/index.html
```

## Features

- **Player Builder**: Form-based input for sport, school, conference, stats, injury status
- **NIL Forecast**: Tier probability distribution + dollar valuation estimate
- **Probability Timeline**: Multi-week forecast with confidence intervals
- **Cohort Comparison**: Find similar historical players, percentile ranking
- **Scenario Case Studies**: Pre-built breakout freshman, injury recovery, transfer portal scenarios
- **Multi-task Learning**: Joint NIL tier classification + valuation regression
- **Calibration-aware**: ECE metrics and reliability analysis for probabilistic forecasting

## Testing

```bash
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/simulate` | Run what-if simulation |
| GET | `/simulate/sports` | List supported sports |
| GET | `/simulate/conferences` | List conferences |
| GET | `/players` | List players |
| GET | `/players/{id}/history` | Player snapshot history |
| GET | `/health` | Health check |

## Key Design Decisions

- **No data leakage**: Train/val/test split by `player_id`, not by row
- **Synthetic data fallback**: Full pipeline runs on generated sample data without scraping
- **Config-driven**: All hyperparameters in `train/config.yaml`
- **Model versioning**: Checkpoints saved with timestamps and validation loss

## Synthetic NIL Evaluation Dataset

The full training corpus is produced by `pipeline/generate_nil_dataset.py`. Each row is one player's per-game averages for a specific week of the 2024-2025 NCAA Men's Basketball season. Players are drawn from six conferences that match the Player Builder UI (ACC, SEC, Big Ten, Big 12, Pac-12, and Independent) and the rosters reflect the post-realignment landscape so that Washington State, Oregon State, and Gonzaga remain on the Pac-12 side while UCLA, USC, Oregon, and Washington appear under the Big Ten.

Identity columns include `player_id` (UUID), `player_name`, `school`, `conference`, `program_tier`, `position`, and `class_year`. Per-game stats include `ppg`, `apg`, `rpg`, `spg`, `bpg`, `mpg`, `fg_pct`, `three_pt_pct`, and `ft_pct`. Context columns include `week_number`, `game_date`, `games_played`, and `currently_injured`. NIL outputs are `nil_valuation_usd`, `social_media_followers`, `engagement_rate`, `market_size_score`, `performance_score`, `nil_probability_score`, and `nil_tier`.

Stat distributions are position-aware. Guards skew toward higher assist totals and three-point percentages, forwards balance scoring and rebounding, and centers dominate rebounds and blocks while shooting fewer threes. Stats are sampled from truncated normal and beta distributions and then evolve week to week with realistic noise plus occasional hot streaks and cold spells.

NIL valuation is computed as a weighted function of a composite performance score (ppg 30 percent, apg 15 percent, rpg 15 percent, fg_pct 10 percent, three_pt_pct 10 percent, mpg 20 percent), the program tier multiplier (3.0x for tier 1 down to 0.4x for tier 5), the school market size, and a logarithmic contribution from social followers scaled by engagement rate. Gaussian and lognormal noise make the relationship learnable rather than deterministic. NIL tier is then a percentile-based bucketing within each conference: top 5 percent elite, next 15 percent high, middle 40 percent mid, next 30 percent low, bottom 10 percent developmental. The probability score is a softmax over log-valuation within each conference and position cohort.

Generate the data with:

```bash
python pipeline/generate_nil_dataset.py                       # 10,000 players to data/raw/
python pipeline/generate_nil_dataset.py --target sample       # 500 players to data/sample/
python pipeline/generate_nil_dataset.py --players 5000 --seed 7
```

## Multi-Task Transformer

The model in `models/transformer_model.py` and `models/multitask_head.py` implements a shared Transformer encoder with two heads. The encoder consumes a 20-week sequence of player snapshots, prepends a learnable `[CLS]` token, applies sinusoidal positional encoding, runs four pre-norm encoder layers with eight attention heads and a feedforward dimension of 512, and returns the `[CLS]` hidden state as the player-level representation. Padded weeks are excluded from attention via a key padding mask, so short or partial seasons are handled gracefully.

`NILTierClassificationHead` is a 128 to 64 to 5 MLP that emits logits over the five NIL tiers. `NILValuationRegressionHead` is a 128 to 64 to 1 MLP with a softplus on the output so the predicted dollar valuation is always non-negative. `MultiTaskNILModel` composes the encoder with both heads and returns a dictionary of `tier_logits`, `valuation_pred`, and the shared representation.

`MultiTaskLoss` combines cross-entropy on the classifier with a Huber (smooth L1) loss on the regression. Huber is more robust to the long-tailed dollar distribution than mean squared error. The default mode uses fixed alpha and beta weights, but a `use_uncertainty_weighting` flag enables Kendall and Gal style learnable log-variance parameters that let the model balance the two objectives automatically.

Joint training of the two tasks regularizes the shared encoder. Each task's gradients inform the representation that the other reads, which reduces the overfitting that two independent models would each invite. The classification head also gives the simulator UI an interpretable tier label alongside the dollar prediction, which is more actionable for end users than a single scalar.

## Training and Evaluation

Run training with:

```bash
python pipeline/generate_nil_dataset.py
python pipeline/preprocess.py
python train/train_multitask_transformer.py --epochs 50 --batch-size 64
```

The training script applies a `log1p` transform to `nil_valuation_usd` before optimizing, splits players 70/15/15 stratified by NIL tier so no `player_id` leaks across splits, optimizes with AdamW (learning rate 1e-4, weight decay 1e-5) under a cosine annealing schedule, clips gradients at 1.0, and runs mixed precision on CUDA when available. Early stopping triggers after seven epochs without validation loss improvement.

Per epoch the script tracks classification accuracy, macro F1, and per-tier precision and recall, alongside regression MAE in dollars, RMSE in dollars, and R-squared. A composite metric (the average of macro F1 and clipped R-squared) is reported for combined performance. After training the best checkpoint at `models/saved/multitask_transformer_best.pt` is reloaded against the held-out test set and the script writes a confusion matrix, a predicted-versus-actual scatter, residuals broken out by program tier and conference, and an attention-weight visualization to `models/saved/`. A JSON metrics report is saved to `models/saved/training_report.json`.

## Simulator Integration

`simulator/engine.py` loads `models/saved/multitask_transformer_best.pt` along with the fitted encoders and scaler from `pipeline/`. The `WhatIfSimulator` accepts player-builder fields, encodes categoricals, scales numerics, packs the resulting frame into a 20-week padded sequence with attention mask, and runs the model to return tier probabilities and a dollar valuation. `simulator/comparator.py` exposes `PlayerComparator` for side-by-side comparisons of two simulated players' tiers and valuations, plus the existing `CohortComparator` for finding similar historical players.

## Tests

```bash
pytest tests/test_pipeline.py tests/test_multitask.py -v
```

