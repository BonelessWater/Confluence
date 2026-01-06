# Quick Start Guide - Refactored Trading System

## 🚀 **Get Started in 5 Minutes**

### **1. Verify Installation**

Make sure you have all dependencies:
```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib
```

### **2. Run Tests (Verify Everything Works)**

```bash
cd tests
python run_all_tests.py
```

**Expected Output:**
```
================================================================================
TEST SUMMARY
================================================================================
  FEATURES: ✓ PASSED
  BACKTEST: ✓ PASSED
  MODELS: ✓ PASSED
================================================================================

🎉 ALL TESTS PASSED!
```

### **3. Run a Simple Backtest**

Create `scripts/quick_backtest.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.linear_model import LinearModel
from src.backtesting.backtest_engine import BacktestEngine
from config.settings import TICKERS

# Create a simple model
model = LinearModel(name="LinearBaseline", n_features=50, alpha=1.0)

# Create backtest engine
engine = BacktestEngine(
    model=model,
    tickers=['SPY'],
    apply_transaction_costs=True  # Realistic 8 bps costs
)

# Run backtest
print("Running backtest on SPY...")
results = engine.run_single_ticker('SPY')

print(f"\nResults saved to: output/{model.name}/SPY/")
```

Run it:
```bash
cd scripts
python quick_backtest.py
```

### **4. Compare Multiple Models**

Create `scripts/compare_models.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.linear_model import LinearModel
from src.models.xgboost_model import XGBoostModel
from src.backtesting.backtest_engine import BacktestEngine

# Create models
models = [
    LinearModel(name="Linear", n_features=50),
    XGBoostModel(name="XGBoost", max_depth=5, n_estimators=100)
]

# Run backtests
for model in models:
    print(f"\n{'='*80}")
    print(f"Testing {model.name}")
    print(f"{'='*80}")

    engine = BacktestEngine(model=model, tickers=['SPY'])
    results = engine.run_single_ticker('SPY')

print("\nDone! Check output/ directory for results")
print("  output/Linear/SPY/")
print("  output/XGBoost/SPY/")
```

Run it:
```bash
python compare_models.py
```

---

## 📊 **Check Your Results**

After running backtests, check the `output/` directory:

```
output/
├── Linear/
│   └── SPY/
│       ├── SPY_trades.csv          # All trades
│       ├── SPY_equity_curve.csv    # Capital over time
│       ├── SPY_metrics.txt         # Performance summary
│       └── SPY_backtest_results.png # Visualization
└── XGBoost/
    └── SPY/
        └── ...
```

**Key Metrics to Look For:**
- **Total Return**: Should be 5-15% (realistic)
- **Sharpe Ratio**: Should be 1.5-2.5 (good but achievable)
- **Max Drawdown**: Expect -20% to -35%
- **Win Rate**: Around 50-55%

---

## 🔧 **Common Tasks**

### **Adjust Transaction Costs**

Edit `config/settings.py`:
```python
COMMISSION_BPS = 5.0  # Change to 3.0 for lower costs
SLIPPAGE_BPS = 3.0    # Change to 2.0 for lower costs
```

### **Change Tickers**

Edit `config/settings.py`:
```python
TICKERS = ['SPY', 'QQQ', 'IWM']  # Fewer tickers
```

### **Tune Model Hyperparameters**

For Linear model:
```python
model = LinearModel(
    n_features=30,  # Try different values: 20, 50, 100
    alpha=0.5       # Try: 0.1, 1.0, 10.0
)
```

For XGBoost:
```python
model = XGBoostModel(
    max_depth=3,        # Try: 3, 5, 7
    n_estimators=50,    # Try: 50, 100, 200
    learning_rate=0.05  # Try: 0.01, 0.1, 0.2
)
```

For Attention model:
```python
model = AttentionModel(
    hidden_dim=256,     # Try: 256, 512, 1024
    num_heads=4,        # Try: 4, 8, 16
    num_layers=2,       # Try: 2, 3, 4
    dropout=0.3         # Try: 0.1, 0.2, 0.3
)
```

---

## ⚠️ **Important: What Changed**

### **Performance Will Be Lower**
- **Old System**: 34% returns (unrealistic)
- **New System**: 5-15% returns (realistic)

**Why?**
1. Fixed look-ahead bias (was seeing the future ❌)
2. Fixed circular logic (was using training labels ❌)
3. Added transaction costs (was assuming zero costs ❌)

### **This is GOOD!**
- Old results were inflated and not achievable
- New results are realistic and tradeable
- You can trust the new numbers

---

## 🧪 **Debugging**

### **Tests Failing?**

Run individual test files to see details:
```bash
cd tests
python test_features.py   # Check for look-ahead bias
python test_backtest.py   # Check return calculation
python test_models.py     # Check model interfaces
```

### **Import Errors?**

Make sure you're running from the correct directory:
```bash
# Should be in confluence/ directory
pwd  # or cd on Windows

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### **Missing Dependencies?**

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install numpy pandas scikit-learn xgboost torch matplotlib scipy
```

---

## 📚 **Learn More**

- **Full Documentation**: See `IMPLEMENTATION_SUMMARY.md`
- **Test Details**: Check `tests/` directory
- **Configuration**: See `config/settings.py`
- **Models**: Browse `src/models/`

---

## 🎯 **Next Steps**

1. ✅ Run tests (`python tests/run_all_tests.py`)
2. ✅ Run simple backtest (Linear model on SPY)
3. ✅ Compare models (Linear vs XGBoost)
4. 🔲 Try ensemble model
5. 🔲 Run walk-forward validation
6. 🔲 Optimize hyperparameters
7. 🔲 Test on multiple tickers

---

## 💡 **Pro Tips**

### **Start Simple**
- Begin with LinearModel (fastest)
- Move to XGBoost (good baseline)
- Try Attention only if baselines work
- Use Ensemble to combine best models

### **Watch for Overfitting**
- Train correlation >> Validation correlation ⚠️
- Use walk-forward validation
- Check performance stability over time

### **Realistic Expectations**
- 10% annual return is excellent
- Sharpe > 1.5 is very good
- Win rate ~50-55% is normal
- Drawdowns -20% to -30% are expected

---

**Ready to start? Run the tests and then your first backtest!**

```bash
# Step 1: Test
cd tests && python run_all_tests.py

# Step 2: Backtest
cd ../scripts && python quick_backtest.py
```

Good luck! 🚀
