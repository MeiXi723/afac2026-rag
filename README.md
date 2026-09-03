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
