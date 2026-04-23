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
