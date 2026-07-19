# Quant Trading System / 量化交易系统

A multi-factor quantitative trading system for medium-term US equities. Uses momentum-driven alpha signals with mean-variance portfolio optimization, dynamic leverage, and automated execution via Alpaca.

多因子量化交易系统，专注于美股中期投资。基于动量驱动的 Alpha 信号，结合均值-方差组合优化、动态杠杆管理，以及通过 Alpaca 自动执行交易。

**[Live Performance Dashboard / 实时表现看板 →](https://yingwang.github.io/trade/)**

---

## Backtest Performance / 回测表现

> **Configuration**: 5 price-based alpha factors (momentum, 52-week high proximity, short-term reversal, trend persistence, volatility contraction), 12 concentrated positions, 22% target volatility, 3-week rebalance with a 40% cap on total turnover (exit legs and leverage changes included), dynamic leverage with fast regime detection, Almgren-Chriss market impact cost model. Sharpe/Sortino computed against a 4% risk-free rate. Fundamental factors disabled due to yfinance look-ahead limitations.
>
> **配置**: 5个价格因子（动量50% + 52周新高20% + 短期反转10% + 波动率收缩10% + 趋势持续10%），12只集中持仓，22%目标波动率，3周再平衡+40%总换手上限（含退出腿与杠杆变化），快速regime检测动态杠杆，Almgren-Chriss市场冲击成本模型。Sharpe/Sortino 按 4% 无风险利率计算。基本面因子因yfinance前视偏差已禁用。

### Results pending rerun / 结果待重跑

The previous tables and chart were generated before the next-session-open
execution model, corrected Sortino definition, fixed rebalance calendar,
continuous-window reporting, and final exposure limits were introduced. They
have been removed rather than presented as current evidence. Run the manual
`README Backtest Refresh` workflow to publish a new traceable set of results.

旧表格和净值图生成于本轮回测时序、Sortino、固定再平衡日历、连续窗口及最终风险上限修复之前，因此不再作为当前结果展示。请运行手动 `README Backtest Refresh` workflow 后再发布可追溯的新结果。

Even after rerunning, the default static stock universe must be labelled as
survivorship-biased. A bias-reduced run requires both the optional
point-in-time universe file and delisting-return file described in
`quant/data/point_in_time.py`.

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
    Execution (alpaca-py with safety checks; next-session open in backtest)
```

### Disabled Factors / 已禁用因子

Quality (质量) 和 Value (价值) 因子已禁用（权重=0），因为 `yfinance.Ticker.info` 只提供当前快照数据，在回测中会造成严重的前视偏差。详见 `docs/audit/ALPHA_AUDIT.md`。

接入时点基本面数据源（如 Sharadar ~$30/月）后可重新启用。

---

## Quick Start / 快速开始

### Install / 安装

```bash
python3.12 -m pip install --require-hashes -r requirements.lock
```

`requirements.lock` pins the complete dependency graph and hashes. Direct
dependencies are declared in `requirements.in`; regenerate the lock only as an
intentional dependency update.

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
- **Idempotent orders**: 稳定的 `client_order_id` 防止网络重试重复下单
- **Position reconciliation**: 每次再平衡后自动对账
- **Cross-run concurrency**: GitHub Actions 并发组串行化两个账户的工作流
- **Partial-fill recovery**: 部分成交或拒单不会把再平衡错误标记为完成
- **Corporate-action fail closed**: 若 Alpaca 模拟账户仍以拆股前单位记录 BKNG，交易会暂停；这是防止数量被放大 25 倍，不影响代码提交或回测
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
│   │   ├── ml_features.py      # 价格派生特征 + 显式缺失标记
│   │   ├── lgbm_model.py       # 分组 LambdaRank + purged split + IC/ICIR
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
| Backtest & QA / 回测质控 | 离线回归测试 + 时序/缺失行情/退市事件边界检查 |
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
  rebalance_anchor_date: "2000-01-03"  # 各报告窗口共用固定再平衡相位
  market_impact_coeff: 2.5   # Almgren-Chriss 市场冲击系数（小账户适配）
  risk_free_rate: 0.04       # Sharpe/Sortino 按超额收益计算
```

可选的时点研究数据接口：

```yaml
data:
  point_in_time_universe_file: data/pit_universe.csv
  delisting_returns_file: data/delisting_returns.csv
```

二者必须同时提供并通过模式校验；仅提供历史成分股却遗漏退市收益仍会高估结果，因此系统会拒绝这种配置。

---

## Known Limitations / 已知局限

1. **Survivorship bias / 幸存者偏差**: 静态100股票池排除历史退市股，回测收益偏高。ETF control workflow 只用于跨资产稳健性检查，不能量化这个偏差；偏差收敛需要时点成分股与退市收益文件
2. **No point-in-time fundamentals / 无时点基本面**: yfinance只提供当前快照，quality/value因子已禁用
3. **Next-open execution / 次日开盘执行**: 回测中信号在 T 日收盘计算，优先按下一可交易日开盘价成交；若开盘缺失才使用该日收盘，开盘和收盘都缺失则等待，不会沿用旧价格假成交
4. **Paper-trading fills / 模拟盘成交**: Alpaca paper 账户的成交不含真实点差与市场冲击，实盘成本会更高

4. **Paper corporate actions / 模拟账户公司行动**: BKNG 25 拆 1 于 2026-04-02 生效、2026-04-06 起按拆股价交易。若模拟账户仍显示拆股前数量/成本，自动交易会安全停止，需先在 Alpaca 重置或修复该模拟持仓。已手工补入的拆股现金必须记录在 `dashboard.split_cash_compensations`，避免仪表盘再次自动补账

详见 `docs/audit/CONFIDENCE_ASSESSMENT.md` 和 `docs/audit/FINAL_DELIVERY.md`。

---

## Disclaimer / 免责声明

本软件仅用于教育和研究目的。过去的回测业绩不代表未来表现。使用风险自负。请务必先使用模拟交易（Paper Trading）充分测试，再考虑投入真实资金。
