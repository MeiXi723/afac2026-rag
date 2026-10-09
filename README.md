# AFAC2026 金融长文本 RAG 检索问答系统

天池 AFAC2026 金融数据智能竞赛的 RAG（检索增强生成）核心管道，面向金融长文本文档的检索问答与多选题自动作答。

## 架构

五阶段解耦管道，各阶段可独立调试：

```
文档解析 → 分块 → 索引 → 混合检索 → 生成 + 校验
(parser)  (chunker) (indexer) (retriever) (generator + validator)
```

- **解析/分块**：金融研报、公告、年报等长文档的结构化解析与语义分块
- **索引**：BM25（稀疏）+ FAISS 向量索引（稠密）双路，按文档类型构建 profile
- **检索**：领域路由 + 混合检索 + RRF 融合 + CrossEncoder 重排序，多级 fallback
- **生成**：Prompt 模板驱动作答，多选题支持，LLM 输出 JSON 鲁棒提取（多级 fallback）
- **校验**：答案归一化与交叉验证，不通过自动回退

## 工程特性

- 多选题 ThreadPoolExecutor 并发检索作答，提效约 3 倍
- 全链路 JSONL 结构化日志，推理决策可追溯
- 30+ 配置项集中管理，带启动校验
- pytest 测试覆盖各阶段核心逻辑（含端到端用例）

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 LLM API Key
pytest tests/          # 运行测试
```

> 注：本仓库为竞赛核心管道代码，不含竞赛数据与答案文件。

## 环境变量

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` |

## 赛后复盘工具链（scripts/）

比赛结束后基于盘上产物构建的离线复盘管线（零 LLM API、规则化可复算）：

- **retrieve_dump.py**：按题型差异化的证据包导出（mcq/multi/tf/计算题各自独立的上下文预算与查询策略），内置规则系保底（选项×文档感知的四级规则）与 `RB_SKIP_*` 单变量对照开关
- **answer_judge.py**：判分单一真源——字母题全等 + 数值题容差（0.5% 相对误差、约剥离、token 序列校验），`eval_v2` / `eval_faith` / `eval_report` 三处共用，杜绝多副本漂移
- **eval_rag / eval_faith / eval_report**：三层评测（检索 Recall@bundle + MRR、生成 Accuracy + 多选 F1、忠实度 2×2 归因矩阵），全部规则化、可复算
- **rebuild_indexes / verify_indexes**：BM25 + FAISS 三件套串行重建与完整性校验
- **chunker 实验**：v2（域 profile/token 预算）至 v5（bge-small 语义断点检测）及双粒度混合的完整探索实现，含负结果归档

## 运行间噪声与实验纪律

检索管线存在 ±1-2 题（百分榜样本）的运行间噪声（疑源：CrossEncoder CPU 推理的浮点归约顺序）。单变量实验结论必须超出噪声带方可采信；脚本化变更失败必须阻塞依赖它的长任务。
