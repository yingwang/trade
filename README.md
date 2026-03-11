# Quant Trading System / 量化交易系统

A multi-factor quantitative trading system for medium-term US equities. Combines six alpha factors with mean-variance portfolio optimization and automated execution via Alpaca.

多因子量化交易系统，专注于美股中期投资。结合六大 Alpha 因子、均值-方差组合优化，以及通过 Alpaca 自动执行交易。

## Strategy Overview / 策略概述

The system scores stocks using a weighted composite of six factors:

系统使用六个因子的加权复合评分对股票进行排名：

| Factor / 因子 | Weight / 权重 | Description / 说明 |
|--------|--------|-------------|
| Momentum / 动量 | 30% | 1/3/6/12 个月截面收益率（跳过最近一个月的短期反转） |
| Trend / 趋势 | 20% | SMA 50/200 比率 — 偏好处于上升趋势的股票 |
| Value / 价值 | 15% | P/E 和 P/B 倒数 — 偏好估值较低的股票 |
| Mean Reversion / 均值回归 | 15% | 布林带 z-score — 偏好超卖的股票 |
| Volatility / 波动率 | 10% | 63 日实际波动率 — 偏好低波动率股票 |
| Quality / 质量 | 10% | ROE、利润率、盈利增长 |

The portfolio holds up to 20 positions, rebalances monthly, and targets 15% annualized volatility.

组合最多持有 20 只股票，每月再平衡一次，目标年化波动率 15%。

## Quick Start / 快速开始

### Install / 安装

```bash
pip install -r requirements.txt
```

Dependencies / 依赖: `numpy`, `pandas`, `yfinance`, `scipy`, `matplotlib`, `seaborn`, `ta`

### Run a Backtest / 运行回测

```bash
# 默认回测（2022-01-01 至今，$1M 初始资金）
python run.py backtest

# 自定义日期并生成图表
python run.py backtest --start 2023-01-01 --end 2024-12-31 --plot

# 详细日志
python run.py -v backtest --plot --plot-output my_backtest.png
```

### View Current Signals / 查看当前信号

```bash
python run.py signal
```

显示股票池中每只股票的复合 Alpha 评分，从强到弱排列。

---

## Manual Usage (Google Colab) / 手动使用（Google Colab）

Notebook `quant_trading_colab.ipynb` 提供可视化交互工作流，无需本地环境。

### Setup in Colab / Colab 设置

1. 在 Google Colab 中打开 notebook
2. 运行 **Cell 1**（Setup）— 克隆仓库并安装依赖
3. 运行 **Cell 2**（Imports）— 加载配置并初始化策略

### What Each Cell Does / 各单元格功能

| Cell / 单元格 | Purpose / 功能 |
|------|---------|
| 1 - Setup / 设置 | 克隆仓库、安装依赖包 |
| 2 - Imports / 导入 | 加载配置、创建策略对象 |
| 3 - Backtest / 回测 | 运行完整历史回测，输出摘要指标（Sharpe、CAGR、最大回撤等） |
| 4 - Equity Curve / 净值曲线 | 绘制净值曲线、回撤、滚动 Sharpe |
| 5 - Monthly Returns / 月度收益 | 按年/月的收益率热力图 |
| 6 - Risk Report / 风险报告 | 扩展统计：偏度、峰度、VaR、CVaR、胜率 |
| 7 - Current Signals / 当前信号 | 获取今天所有股票的 Alpha 评分 |
| 8 - Target Portfolio / 目标组合 | **核心单元格** — 显示今天该买什么 |

### Using the Target Portfolio (Cell 8) / 使用目标组合

将 `MY_CAPITAL` 修改为你的账户资金：

```python
MY_CAPITAL = 50_000  # <-- 你的组合资金（美元）
```

运行后输出示例：

```
=================================================================
  TARGET PORTFOLIO  |  Capital: $50,000
=================================================================
  Stock      Weight    Dollars   Shares      Price     Score
  ------------------------------------------------------------
  NVDA        9.8%    $4,900      36   $136.25    0.842
  META        8.5%    $4,250       7   $607.14    0.731
  ...
  ------------------------------------------------------------
  TOTAL      95.2%   $47,600
  Cash reserve: $2,400
=================================================================
```

然后在你的券商账户中手动下单。

### Rebalancing Manually / 手动再平衡

每月运行一次 Cell 8（约每 21 个交易日）。将新目标与当前持仓对比后调整：
- **卖出** 已跌出前 20 名的持仓
- **买入** 新进入的持仓
- **调整** 现有持仓至目标权重

---

## Automated Trading with Alpaca / 使用 Alpaca 自动交易

### 1. Create an Alpaca Account / 创建 Alpaca 账户

在 [alpaca.markets](https://alpaca.markets) 注册。建议先使用 **模拟交易（Paper Trading）** 账户测试，无需真金白银。

### 2. Set Environment Variables / 设置环境变量

```bash
export ALPACA_API_KEY="your-api-key"        # 你的 API Key
export ALPACA_SECRET_KEY="your-secret-key"  # 你的 Secret Key
```

默认为模拟交易。如需实盘交易（风险自负），另设：

```bash
export ALPACA_PAPER="false"
```

### 3. Install the Alpaca SDK / 安装 Alpaca SDK

```bash
pip install alpaca-trade-api
```

### 4. Paper Trading Commands / 模拟交易命令

```bash
# 预览交易计划（不执行）
python paper_trade.py --dry-run

# 查看当前持仓状态
python paper_trade.py --status

# 执行再平衡（仅在距上次 21+ 个交易日后运行）
python paper_trade.py

# 强制立即再平衡
python paper_trade.py --force
```

### 5. Automate with Cron / 使用 Cron 自动化

将以下内容添加到 crontab，在工作日美东时间下午 3:55 自动再平衡：

```bash
crontab -e
```

```
55 15 * * 1-5 cd /path/to/trade && python paper_trade.py >> logs/trade.log 2>&1
```

脚本在 `logs/paper_trade_state.json` 中记录状态，仅在达到配置间隔（21 个交易日）后才执行再平衡。使用 `--force` 可强制执行。

### 6. Monitoring / 监控

- **交易日志**: `logs/paper_trade_YYYYMMDD.log` — 每笔订单含时间戳和成交价
- **状态文件**: `logs/paper_trade_state.json` — 上次再平衡日期及完整交易历史
- **快速检查**: `python paper_trade.py --status`

---

## Configuration / 配置说明

所有参数在 `config.yaml` 中：

```yaml
universe:
  symbols: [AAPL, MSFT, GOOGL, ...]   # 交易股票池
  benchmark: SPY                        # 基准指数

portfolio:
  max_positions: 20          # 最大持仓数
  max_position_weight: 0.10  # 单只股票最大权重 10%
  min_position_weight: 0.02  # 单只股票最小权重 2%
  target_volatility: 0.15    # 目标年化波动率 15%
  rebalance_frequency_days: 21  # 再平衡周期（交易日）
  transaction_cost_bps: 10   # 交易成本 10 个基点

risk:
  max_drawdown_limit: 0.20   # 最大回撤 20% 时停止交易
  max_sector_weight: 0.30    # 单一行业最大权重 30%
  stop_loss_pct: 0.08        # 单只股票止损线 8%
```

### Customizing the Universe / 自定义股票池

编辑 `config.yaml` 中的 `symbols` 列表。系统支持 Yahoo Finance 上所有可用的美股。默认包含 50 只股票，涵盖大型科技股（NVDA、AMD、CRM、PLTR 等）和蓝筹防御股（JNJ、PG、KO 等）。

### Tuning Factor Weights / 调整因子权重

因子权重决定组合风格。编辑 `config.yaml` 中的 `signals.factor_weights`：

```yaml
signals:
  factor_weights:
    momentum: 0.30       # 动量 — 近期价格赢家
    mean_reversion: 0.10 # 均值回归 — 超卖反弹候选
    trend: 0.25          # 趋势 — 站上关键均线的股票
    volatility: 0.05     # 波动率 — 偏好低波动（越高越防御）
    value: 0.15          # 价值 — P/E 和 P/B 较低
    quality: 0.15        # 质量 — 高 ROE、利润率、盈利增长
```

**Presets / 预设方案：**

| Style / 风格 | momentum / 动量 | trend / 趋势 | value / 价值 | volatility / 波动率 | mean_rev / 均值回归 | quality / 质量 |
|-------|----------|-------|-------|------------|----------|---------|
| Growth/Tech / 成长科技 | 0.35 | 0.30 | 0.05 | 0.00 | 0.10 | 0.20 |
| Balanced / 均衡（默认） | 0.30 | 0.25 | 0.15 | 0.05 | 0.10 | 0.15 |
| Defensive / 防御收益 | 0.15 | 0.10 | 0.25 | 0.25 | 0.10 | 0.15 |

权重总和必须为 1.0。

### Adjusting Risk / 调整风险

- **更保守**: 降低 `target_volatility`（如 0.10）、降低 `max_position_weight`、提高 `min_position_weight`
- **更激进**: 提高 `target_volatility`（如 0.20）、允许更大的单只持仓

---

## Project Structure / 项目结构

```
trade/
├── config.yaml                 # 策略配置
├── run.py                      # 命令行：回测与信号
├── paper_trade.py              # Alpaca 模拟/实盘交易
├── quant_trading_colab.ipynb   # 交互式 Colab Notebook
├── requirements.txt
├── quant/
│   ├── strategy.py             # 主策略调度器
│   ├── data/
│   │   └── market_data.py      # Yahoo Finance 数据获取
│   ├── signals/
│   │   └── factors.py          # Alpha 因子计算
│   ├── portfolio/
│   │   └── optimizer.py        # 均值-方差优化
│   ├── backtest/
│   │   ├── engine.py           # 回测引擎
│   │   └── report.py           # 分析与报告
│   ├── execution/
│   │   ├── broker.py           # 券商接口 + 模拟券商
│   │   └── alpaca_broker.py    # Alpaca API 集成
│   └── utils/
│       └── config.py           # 配置加载器
└── tests/                      # 单元测试
```

## Disclaimer / 免责声明

本软件仅用于教育和研究目的。过去的业绩不代表未来表现。使用风险自负。请务必先使用模拟交易测试，再投入真实资金。
