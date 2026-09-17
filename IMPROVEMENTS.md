# Trade 项目 Review(2026-07-07)

> **状态更新(2026-07-07)**:下列问题 1-16 已全部修复并推送 main。要点:LGBM 训练改用 purged/embargoed split(quant/signals/lgbm_model.py 的 purged_train_val_split);config 加 risk_free_rate: 0.04;optimizer 新增 enforce_turnover_cap 对最终权重(含退出腿与杠杆变化)执行 40% 总换手上限;实盘路径接 enforce_live_data_quality 硬闸门;LGBM 实盘 fallback 改硬中止;休市日整个 run 直接退出;止损检查提前到每个交易日(原先被 should_rebalance 挡住,只在调仓日跑);两个 workflow cron 错峰至 15:00/15:10 UTC 且 push 前 rebase;拆股修正加建仓价量级防护并收拢进 site_common.py;training_history 属性名修正;两份 paper_trade 收拢进 paper_trade_common.py 且删除死参数 --capital;prev_scores 与账户真实权重接入实盘;进场价只在新建仓时记录;DailyTracker 跨 run 持久化;README/CLAUDE.md 文档全面修正;main 上过时的 site/data 快照与 v2-v5 旧图删除并 gitignore;earnings_revision/low_proximity 死因子删除;volume_momentum 更名 trend_persistence(含 ml_features 各代理特征更名);SignalGenerator 隐藏默认权重置零。测试从 89 个扩到 141 个,全部通过。README 回测数字经 readme-backtest workflow(Actions)重算。

本次 review 通读了仓库全部 Python 源码(约 9600 行)、五个 GitHub Actions workflow、两份配置与测试套件,并用 gh CLI 核对了远端 main 分支的实际运行记录与实盘状态。结论先行:这套系统的工程质量高于绝大多数个人量化项目,安全层、诚实性文档(docs/audit)与 ETF 对照组的思路都做得扎实;但仍存在几处会直接影响回测可信度与实盘正确性的问题,按优先级列在下面。

## 〇、当前实盘状态(来自远端 main,非本地陈旧副本)

本地 clone 停在 6 月中,远端实际一直在跑。两个账户都在 2026-06-29 完成了最近一次 rebalance(v3 于 17:18:55 UTC,LGBM 于 17:17:59 UTC),按 21 个交易日的节奏,下一次大约落在 7 月底。每日 workflow 除 6-29 的 v3 那次(见问题 7)以外全部绿色。本地仓库建议先 `git pull`。

---

## 一、高优先级:影响回测可信度与实盘正确性

### 1. LGBM 验证集存在前视泄漏(缺 purge/embargo)

位置:`quant/signals/lgbm_strategy.py:156-171`(`_train_model`),同样的结构复制在 `quant/strategy_ensemble.py:286-297`。

训练目标 y 在第 t 行是 t+1 到 t+21 的前向收益排名。验证窗取 `X[val_start:date_idx]`,一直贴到当前 rebalance 日 date_idx 为止。于是验证窗最后 21 行的目标,用到了 date_idx 之后才发生的价格;early stopping 以这个验证集选轮数,等于用未来信息挑模型。此外训练窗末尾若干行的目标窗口与验证窗前段重叠,也就是标准的 purged cross-validation 问题(de Prado),会让验证分数系统性偏乐观。

实盘路径不受影响(未来目标是 NaN,`_prepare_panel_data` 会把它们过滤掉),受影响的是回测数字的诚实性。修法:验证窗改为 `X[val_start : date_idx - pred_horizon]`,并在训练窗与验证窗之间再空出 pred_horizon 行。

### 2. Sharpe/Sortino 按无风险利率 0 计算

位置:`quant/backtest/engine.py:47` 读取 `risk_free_rate`,而 `config.yaml` 里根本没有这个键,于是回退为 0。

2026 年现金利率在 4% 上下,README 里 5 年期 Sharpe 0.83 若按真实 rf 计算大约会掉到 0.6 左右。这是一行配置的事:在 `config.yaml` 的 backtest 段加 `risk_free_rate: 0.04`(或接一个动态源),并同步刷新 README 表格。

### 3. 换手率约束与惩罚都漏掉了"退出腿",杠杆缩放又发生在约束之后

位置:`quant/portfolio/optimizer.py:180-205`。

`w_prev = prev_weights.reindex(selected).fillna(0)` 把上一期持有、本期未入选的股票直接从 L1 项里丢掉,于是清仓这部分既不进 40% 换手约束,也不进换手惩罚。另外 `apply_vol_scaling` 在优化完成之后把整组权重再乘一个 regime 系数(0.8x 到 1.8x),这一步产生的换手同样不受约束。README 里"40% 换手上限"与"平均换手 79%"并排出现,根源就在这两处。修法:把 selected 与 prev_weights 的并集作为优化变量空间(未入选者上界 0),让退出腿进入 L1;或者至少在文档里把 40% 的语义改写为"入选集合内部的换手上限"。

### 4. 实盘路径没有任何数据质量闸门

位置:`quant/strategy.py:174-245` 与 `quant/signals/lgbm_strategy.py:458-550`(两处 `get_current_portfolio`)。

`DataQualityChecker` 只接在 `run_backtest` 里,而且即便 FAILED 也只是记日志继续跑。实盘每天从 yfinance 拉数据,一旦被限流、返回半截数据或者大面积 NaN,信号会在垃圾数据上照常生成、照常下单,没有任何一步会拦住它。修法:在两个 `get_current_portfolio` 的取数之后接同一个 checker,`passed == False` 或有效标的数低于阈值(比如少于 80 只)就抛异常终止本次 run,宁可当天不调仓。

### 5. ML 后端失效时静默退化成"按字母序买前 12 只"

位置:`quant/signals/lgbm_strategy.py:224-227`(`_fallback_scores`)与 `quant/portfolio/optimizer.py:127-130`。

fallback 分数是全体等值常数,`sort_values` 稳定排序后 `head(12)` 拿到的就是列顺序(近似字母序)的前 12 只。也就是说,lightgbm 安装失败或训练抛异常的那天,系统会一声不响地把组合换成 AAPL、ABBV、ABT 这一串,并真的下单。对回测这是可接受的兜底,对实盘应当硬失败:在 `paper_trade_lgbm.py` 里检测到 fallback(或 `model.model is None`)就直接 abort,当天不交易。

### 6. 休市日照常提单,存在悬挂订单与重复执行的组合风险

位置:`paper_trade.py:224-226`(市场关闭只 warning 不退出)、`quant/execution/alpaca_broker.py:213-250`(超时撤单)。

调仓日撞上美股假日时,脚本仍会把整批市价单提交给 Alpaca,然后每单轮询 30 秒超时、逐一尝试撤单。撤单成功则当天白跑一场(能自愈);但只要有一单撤单失败,它会在次日开盘不受控成交,而脚本因为当天"无成交"没有更新 last_rebalance,次日会重新计算并再下一整套单,叠加昨天的悬挂单就是双份买入。修法很简单:`is_market_open()` 为 False 时直接 return,别提交。顺带一提,cron 注释里自己也写了 14:30 UTC 在冬令时等于 09:30 ET,恰好是开盘价差最宽的时刻,建议冬季观察一下成交质量或干脆挪到 15:00 UTC。

---

## 二、中优先级:运维与一致性

### 7. 两个 rebalance workflow 同时 push main 的竞态,6-29 已经真实咬过一次

`rebalance.yml` 与 `rebalance-lgbm.yml` 用同一条 cron(30 14 * * 1-5),commit 步骤都是 add、commit、push,没有先 pull。2026-06-29 两个 run 几乎同时进入交易(都花了 2 分多钟),LGBM 先推,v3 的 push 被拒,run 标红;交易本身已经完成,state 靠 Actions cache 在次日 6-30 的 run 里补提交回 main(那条"chore: update trade state"提交比交易晚了一天,就是这么来的)。系统事实上自愈了,但每次撞上都会留下一个红色 run 和一天的状态滞后。修法任选:push 前加 `git pull --rebase origin main`;或把两个 job 合并进一个 workflow 串行跑;或把两条 cron 错开十分钟。

### 8. 拆股修正表没有日期与批次防护,BKNG 一旦重新入选就会污染面板

位置:`generate_site.py:137` 与 `generate_site_lgbm.py:147`(`STOCK_SPLITS`,BKNG 1:25 于 2026-04-06)。

`_adjust_for_splits` 对任何当前持有的 BKNG 都做 qty×25、entry÷25,不检查这笔持仓是不是拆股之前建立的;`sold_credit` 对拆股日之后的每一笔 BKNG 卖单永久追加影子现金。眼下两个账户都不持有 BKNG,所以没事;哪天模型重新选中它,买入的新仓会被再乘 25,面板权益直接失真。修法:比较 avg_entry_price 与拆股前后价格量级来判断该仓位是否需要修正,或给修正项加"建仓日期早于拆股日"的条件;长期看应该把这张手工表换成从 yfinance 拉 corporate actions 自动比对。

### 9. 训练历史面板永远为空:属性名写错

位置:`generate_site_lgbm.py:471` 检查 `hasattr(model, 'training_history_')`,而模型里存的是 `_train_history`(`quant/signals/lgbm_model.py:104`)。site/lgbm/data/training_history.json 里 `training_runs` 恒为空数组,证实了这一点。一行改名即可修好。

### 10. 两份 paper_trade 脚本 95% 重复,且 `--capital` 是死参数

`paper_trade.py` 与 `paper_trade_lgbm.py` 除了策略类、state 文件名和环境变量前缀,其余逐行相同;6-09 那次"enforce dead safety guardrails"类的修复必须记得改两遍,漏一遍就是行为分叉。建议抽一个 `paper_trade_common.py`,两个入口只留十几行参数。另外两份脚本都定义了 `--capital` 却从未使用(资金永远取 `broker.get_portfolio_value()`),要么实现要么删掉。

### 11. 实盘 LGBM 每次 run 从零重训,turnover_penalty 在实盘形同虚设

位置:`quant/signals/lgbm_strategy.py:416-456`。

回测里 `_prev_scores` 在循环内持续存在,score 层面的换手惩罚真实生效;实盘每次是新进程,`_prev_scores` 永远是 None,惩罚从未起过作用,所以实盘换手天然高于回测假设。修法:把上一期 scores 序列化进 state json,run 开始时读回来传入。顺带,模型每次重训也意味着实盘并没有"每 3 次 rebalance 重训一次"这回事,文档与实现不一致。

### 12. 止损的进场价语义在实盘与回测里不一致

回测引擎只在 0 到正仓位的时刻记录 entry price(`quant/backtest/engine.py:121-122`),加仓不重置;实盘脚本对每一笔 buy 成交都覆盖 entry price(`paper_trade.py:429-431`),加仓即重置,止损基准随之抬高或降低。两边应统一(建议向回测语义看齐,或改用 Alpaca 的 avg_entry_price)。

### 13. 日累计安全限额只活在单个进程里

`PreTradeCheck.DailyTracker` 是内存对象,Actions 每次 run 新建,"单日交易总额 500k、单日亏损 25k"的累计语义跨 run 不成立。当前一天只跑一次影响不大,但手动 dispatch 补跑时限额会重新计数。低成本修法是把当日累计值也写进 state json。

---

## 三、低优先级:文档与仓库卫生

### 14. 文档与代码脱节的三处

README"Known Limitations"仍写着"Stop-loss not active / 止损未激活"与"Same-day execution",而引擎早已实现每日止损(engine.py:143-169)和 T+1 执行(engine.py:92-138),README 上文与 CLAUDE.md 也各自写着"已每日执行",前后矛盾;README 风险表声称"组合回撤超限时停止交易",但实盘路径根本没有接 `RiskMonitor.check_drawdown`,只有日亏损开关,应把说法改准确或把功能补上;`strategy_ensemble.py:352-362` 的"drawdown protection"只打日志不降仓位,是装饰性代码,要么实现要么删除(该文件只被 run.py 的研究命令引用,不在生产链路上)。

### 15. main 分支上的 site/data 是 3 月底的旧快照

gh-pages 每晚由 update-site.yml 重新生成,main 上那份 site/data/*.json(2026-03-31 / 04-01)不再被任何流程消费,只会误导读仓库的人。建议从 main 删除并 gitignore,或在 README 里注明它们只是样例。

### 16. 零碎项

`refresh_backtest_tables.py` 尚未纳入版本控制,建议 commit(它是 README 数字的再生工具,本身设计为本地运行,与"generate 脚本只在 Actions 跑"的约定不冲突)。`volume_momentum` 及 ml_features 里的 OBV、ATR、volume ratio 全部是无成交量数据的收益率代理,命名会让 dashboard 上的 feature importance 引人误读,建议在面板或文档里注明。`earnings_revision` 与 `low_proximity` 两个因子每次全量计算却权重为零,白花算力,可以移出默认管线。`SignalGenerator` 的默认权重字典里 mean_reversion/trend/volatility 是非零值,一旦有人从 config 里删掉对应行,它们会静默复活,建议默认值全部置 0,让 config 成为唯一事实来源。update-site.yml 里 `generate_factor_data` 会把 `get_current_portfolio` 再跑一遍,整个站点生成流程重复取数两到三次,合并一次取数能把 5 分多钟压掉近半。

---

## 四、建议的动手顺序

第一批(半天):问题 2(加 risk_free_rate 并刷新 README)、问题 9(改属性名)、问题 6(休市即退出)、问题 10 的死参数、问题 14 的文档修正。全部是小改动,立刻提高数字与文档的诚实度。

第二批(一到两天):问题 4(实盘质量闸门)与问题 5(ML 失败硬中止),这两个决定实盘会不会在坏数据上交易;问题 7(push 前 rebase);问题 8(拆股防护)。

第三批(研究性质):问题 1(purge/embargo 后重跑 LGBM 回测,看数字掉多少)、问题 3(换手语义修正后重跑),再用 honest-backtest 的 ETF 对照结果一起更新 README 的预期管理。第 10、11、12 条的重构可以顺路做。

改完推 main 即可,回测表格用 workflow_dispatch 触发 honest-backtest 与 update-site 验证,不需要本地跑生成脚本。
