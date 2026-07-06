# Quant Trading System / 量化交易系统

A multi-factor quantitative trading system for medium-term US equities. Uses momentum-driven alpha signals with mean-variance portfolio optimization, dynamic leverage, and automated execution via Alpaca.

多因子量化交易系统，专注于美股中期投资。基于动量驱动的 Alpha 信号，结合均值-方差组合优化、动态杠杆管理，以及通过 Alpaca 自动执行交易。

**[Live Performance Dashboard / 实时表现看板 →](https://yingwang.github.io/trade/)**

---

## Backtest Performance / 回测表现

> **Configuration**: 5 price-based alpha factors (momentum, 52-week high proximity, short-term reversal, trend persistence, volatility contraction), 12 concentrated positions, 22% target volatility, 3-week rebalance with a 40% cap on total turnover (exit legs and leverage changes included), dynamic leverage with fast regime detection, Almgren-Chriss market impact cost model. Sharpe/Sortino computed against a 4% risk-free rate. Fundamental factors disabled due to yfinance look-ahead limitations.
>
> **配置**: 5个价格因子（动量50% + 52周新高20% + 短期反转10% + 波动率收缩10% + 趋势持续10%），12只集中持仓，22%目标波动率，3周再平衡+40%总换手上限（含退出腿与杠杆变化），快速regime检测动态杠杆，Almgren-Chriss市场冲击成本模型。Sharpe/Sortino 按 4% 无风险利率计算。基本面因子因yfinance前视偏差已禁用。

### 5-Year Backtest (2021-07 → 2026-07)

| Metric / 指标 | Strategy / 策略 | SPY | Difference / 差异 |
|---------------|:-----------:|:---:|:---------:|
| **Total Return / 总收益** | **+193.6%** | +84.1% | **+109.6pp** |
| **CAGR / 年化收益** | **24.2%** | — | — |
| **Sharpe Ratio** | **0.87** | — | — |
| **Sortino Ratio** | **1.20** | — | — |
| **Max Drawdown / 最大回撤** | -28.9% | — | — |
| **Information Ratio** | **+0.86** | — | — |

> Note: Tables recomputed 2026-07-06 by the `readme-backtest` workflow (Actions run 28829367568) after the July 2026 review fixes: Sharpe/Sortino now use a 4% risk-free rate, the 40% turnover cap is enforced on total turnover (exit legs and leverage changes included), and the LightGBM validation split is purged/embargoed. Market impact coefficient 2.5 (Almgren-Chriss), stop-loss (15%) enforced daily.
>
> 注：表格由 readme-backtest workflow 于 2026-07-06 重算（Actions run 28829367568），包含 7 月 review 修复：Sharpe/Sortino 按 4% 无风险利率计算，40% 换手上限按含退出腿与杠杆变化的总换手执行，LightGBM 验证窗做了 purge/embargo。市场冲击系数 2.5（Almgren-Chriss），止损（15%）每日执行。

### 3-Year Backtest (2023-07 → 2026-07)

| Metric / 指标 | Strategy / 策略 | SPY | Difference / 差异 |
|---------------|:-----------:|:---:|:---------:|
| **Total Return / 总收益** | **+158.8%** | +75.8% | **+82.9pp** |
| **CAGR / 年化收益** | **37.6%** | — | — |
| **Sharpe Ratio** | **1.33** | — | — |
| **Sortino Ratio** | **1.97** | — | — |
| **Max Drawdown / 最大回撤** | -27.6% | — | — |
| **Information Ratio** | **+1.14** | — | — |

### 1-Year Backtest (2025-07 → 2026-07)

| Metric / 指标 | Strategy / 策略 | SPY | Difference / 差异 |
|---------------|:-----------:|:---:|:---------:|
| **Total Return / 总收益** | **+48.7%** | +21.3% | **+27.4pp** |
| **CAGR / 年化收益** | **49.4%** | — | — |
| **Sharpe Ratio** | **1.82** | — | — |
| **Sortino Ratio** | **3.35** | — | — |
| **Max Drawdown / 最大回撤** | -11.6% | — | — |
| **Information Ratio** | **+1.64** | — | — |

### Performance Chart / 净值曲线 (5-Year)

![5-Year Backtest](backtest_5yr.png)

> **Note / 注意**: 回测结果仍存在幸存者偏差（静态100股票池排除了历史退市股）。真实样本外表现预计会略低。详见 `docs/audit/CONFIDENCE_ASSESSMENT.md`。

---

## Strategy Overview / 策略概述

### Current Configuration (Growth Profile)

| Parameter / 参数 | Value / 值 | Description / 说明 |
|---------|-------|-------------|
| **Alpha Signals** | 5 factors | 动量50% + 52周新高20% + 短期反转10% + 波动率收缩10% + 趋势持续10% |
| **Positions** | 12 | 集中持仓，高信念选股 |
| **Position Bounds** | 3% - 12% | 每只股票的权重范围 |
| **Target Volatility** | 22% | 接近满仓投资，最小化现金拖累 |
| **Rebalance** | Every 21 trading days | 3周再平衡，降低换手成本 |
| **Max Turnover** | 40% per rebalance | 总换手上限（含退出腿与杠杆变化）；超限时向上期组合渐进过渡 |
| **Max Sector** | 50% | 允许科技股集中但有上限 |
| **Max Drawdown** | 25% | 回测层面的风险监控指标；实盘的对应保护是日亏损熔断与每日止损 |
| **Stop Loss** | 15% | 单只股票止损线，回测与实盘均每日检查 |
| **Leverage** | Up to 1.8x (calm) / 0.8x (stress) | 动态杠杆，基于SPY波动率的市场环境检测 |
| **Cost Model** | Almgren-Chriss | 动态市场冲击成本 = 固定15bps + 冲击系数(2.5) × √(参与率) |

### Signal Pipeline / 信号管道

```
Price Data (yfinance, live path behind a hard data-quality gate)
    │
    ├── Momentum (42d/126d/252d, skip 1m) ─── 50%
    ├── 52-Week High Proximity ──────────── 20%
    ├── Short-Term Reversal (5d) ────────── 10%
    ├── Volatility Contraction (10d/63d) ── 10%
    ├── Trend Persistence (21d autocorr) ── 10%
    │   └── All: Industry-neutral z-score → Winsorize ±3
    │
    ├── Trend Filter: price < 200d SMA → score × 0.5
    └── Blowoff Filter: z-score > 4.0 → score × 0.5
        │
        ▼
    Portfolio Optimizer (Ledoit-Wolf covariance + 5x turnover penalty)
        │
        ├── Sector constraints (max 50% per sector)
        ├── Vol-targeting (22% annual, regime-adjusted)
        ├── Dynamic leverage (21d SPY vol regime detection)
        └── Total turnover cap: 40% per rebalance, on final weights,
            including exit legs and leverage changes
            │
            ▼
    Execution (Alpaca API with safety checks; T+1 close in backtest)
```

### Disabled Factors / 已禁用因子

Quality (质量) 和 Value (价值) 因子已禁用（权重=0），因为 `yfinance.Ticker.info` 只提供当前快照数据，在回测中会造成严重的前视偏差。详见 `docs/audit/ALPHA_AUDIT.md`。

接入时点基本面数据源（如 Sharadar ~$30/月）后可重新启用。

---

## Quick Start / 快速开始

### Install / 安装

```bash
pip install -r requirements.txt
```

Dependencies / 依赖: `numpy`, `pandas`, `yfinance`, `scipy`, `scikit-learn`, `matplotlib`

### Run a Backtest / 运行回测

```bash
# 5年回测（默认配置）
python run.py backtest --start 2021-03-16 --plot

# 自定义日期范围
python run.py backtest --start 2023-01-01 --end 2026-03-16 --plot

# 输出到指定文件
python run.py backtest --start 2021-03-16 --plot --plot-output my_backtest.png
```

### View Current Signals / 查看当前信号

```bash
python run.py signal
```

---

## Automated Trading with Alpaca / 使用 Alpaca 自动交易

### Setup / 设置

```bash
pip install alpaca-trade-api
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
```

### Commands / 命令

```bash
python paper_trade.py --dry-run     # 预览交易（不执行）
python paper_trade.py --status      # 查看当前持仓
python paper_trade.py               # 执行再平衡（需达到21个交易日间隔）
python paper_trade.py --force       # 强制立即再平衡（使用新策略参数）
python paper_trade.py --reconcile   # 对账：策略目标 vs 实际持仓
```

### Safety Features / 安全特性

- **Pre-trade checks**: 单笔订单上限 $50k，日内交易总额上限 $500k，日亏损上限 $25k
- **TWAP splitting**: 大单自动拆分为时间加权分批执行
- **Position reconciliation**: 每次再平衡后自动对账
- **Lock file**: 防止并发执行
- **Paper mode gate**: 防止意外连接实盘账户

### Automate with Cron / 使用 Cron 自动化

```bash
# 工作日美东时间下午 3:55 自动执行（每21个交易日实际再平衡）
55 15 * * 1-5 cd /path/to/trade && python paper_trade.py >> logs/trade.log 2>&1
```

---

## Project Structure / 项目结构

```
trade/
├── config.yaml                 # 策略配置（进攻型）
├── config_etf.yaml             # honest-backtest 的 ETF 对照池配置
├── run.py                      # CLI：回测与信号（含 backtest-lgbm / ensemble）
├── paper_trade.py              # Alpaca 自动交易入口（多因子账户）
├── paper_trade_lgbm.py         # Alpaca 自动交易入口（LightGBM 账户）
├── paper_trade_common.py       # 两个交易入口的共享实现
├── generate_site.py            # 看板数据生成（多因子；仅在 Actions 上跑）
├── generate_site_lgbm.py       # 看板数据生成（LightGBM；仅在 Actions 上跑）
├── site_common.py              # 看板脚本共享逻辑（拆股修正、交易历史）
├── refresh_backtest_tables.py  # README 表格重算（经 readme-backtest workflow）
├── quant/
│   ├── strategy.py             # 多因子策略调度器
│   ├── strategy_ensemble.py    # 多因子 + LightGBM 集成（研究用）
│   ├── data/
│   │   ├── market_data.py      # Yahoo Finance 数据获取
│   │   └── quality.py          # 数据质量检查 + 实盘质量闸门 + 时点数据管理
│   ├── signals/
│   │   ├── factors.py          # Alpha 因子计算
│   │   ├── factor_analysis.py  # IC/ICIR、因子衰减分析工具
│   │   ├── ml_features.py      # LightGBM 特征工程（46 个价格派生特征）
│   │   ├── lgbm_model.py       # LightGBM 排序模型 + purged split
│   │   └── lgbm_strategy.py    # LightGBM 策略调度器
│   ├── portfolio/
│   │   └── optimizer.py        # MVO优化 + Ledoit-Wolf + 换手上限 + 风控
│   ├── backtest/
│   │   ├── engine.py           # 事件驱动回测引擎（T+1、每日止损、保证金利息）
│   │   └── report.py           # 报告与分析
│   ├── execution/
│   │   ├── broker.py           # 券商接口 + 模拟券商
│   │   ├── alpaca_broker.py    # Alpaca API 集成
│   │   └── safety.py           # 交易安全模块（限额、TWAP、对账）
│   └── utils/
│       └── config.py           # 配置加载器
├── .github/workflows/          # rebalance ×2、update-site、tests、honest-backtest、readme-backtest
├── tests/                      # 单元测试（离线合成数据）
└── docs/audit/                 # 6-Agent 审计报告（12份文档，历史存档）
```

## System Audit / 系统审计

本系统经过 6-Agent 团队的全面审计和优化（详见 `docs/audit/`）：

| Agent / 角色 | Key Deliverable / 主要成果 |
|-------|-------------------|
| Data Engineer / 数据工程师 | 修复 ffill bug，添加数据质量检查，标记前视偏差 |
| Alpha Research / 因子研究员 | 修复3个因子bug，添加因子分析工具 |
| Execution Engineer / 执行工程师 | 添加完整安全模块（限额、TWAP、对账） |
| Portfolio & Risk / 组合风控 | Ledoit-Wolf协方差、换手率惩罚、行业约束执行 |
| Backtest & QA / 回测质控 | 89测试全过，回测可信度评估 3/10 → 待接入时点数据 |
| Lead Orchestrator / 总指挥 | 跨模块集成、最终交付报告 |

---

## Configuration / 配置说明

所有参数在 `config.yaml` 中。当前为**进攻型配置**：

```yaml
signals:
  momentum_windows: [42, 126, 252]
  factor_weights:
    momentum: 0.50         # 核心动量信号
    high_proximity: 0.20   # 52周新高距离
    short_term_reversal: 0.10  # 短期反转
    vol_contraction: 0.10  # 波动率收缩
    trend_persistence: 0.10  # 趋势持续（收益自相关；旧名 volume_momentum，实际从未用过成交量）
    quality: 0.00          # 已禁用（前视偏差）
    value: 0.00            # 已禁用（前视偏差）

portfolio:
  max_positions: 12
  target_volatility: 0.22
  rebalance_frequency_days: 21    # 3周再平衡
  max_turnover_per_rebalance: 0.40  # 40%总换手上限（含退出腿与杠杆变化）

risk:
  max_drawdown_limit: 0.25
  max_sector_weight: 0.50
  stop_loss_pct: 0.15

backtest:
  market_impact_coeff: 2.5   # Almgren-Chriss 市场冲击系数（小账户适配）
  risk_free_rate: 0.04       # Sharpe/Sortino 按超额收益计算
```

---

## Known Limitations / 已知局限

1. **Survivorship bias / 幸存者偏差**: 静态100股票池排除历史退市股，回测收益偏高。定量对照见 honest-backtest workflow（同一策略跑在无幸存者偏差的 ETF 池上）
2. **No point-in-time fundamentals / 无时点基本面**: yfinance只提供当前快照，quality/value因子已禁用
3. **T+1 close execution / T+1收盘价执行**: 回测中信号在 T 日收盘计算，交易在 T+1 日收盘价成交；真实盘中成交价仍会有偏差
4. **Paper-trading fills / 模拟盘成交**: Alpaca paper 账户的成交不含真实点差与市场冲击，实盘成本会更高

（旧版此处写着"止损未激活"与"同日执行"，均已过时：回测引擎与实盘脚本都做每日止损检查，回测为 T+1 执行。）

详见 `docs/audit/CONFIDENCE_ASSESSMENT.md` 和 `docs/audit/FINAL_DELIVERY.md`。

---

## Disclaimer / 免责声明

本软件仅用于教育和研究目的。过去的回测业绩不代表未来表现。使用风险自负。请务必先使用模拟交易（Paper Trading）充分测试，再考虑投入真实资金。
