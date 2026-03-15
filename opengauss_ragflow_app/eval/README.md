# RAG Eval

这个目录提供一套最小可用的 RAG 评测脚手架，适合当前项目直接使用。

## 数据格式

使用 `jsonl`，每行一个样本。字段约定：

```json
{
  "id": "q1",
  "vector_db_id": "pgvector-demo",
  "question_type": "procedural",
  "question": "How does hybrid retrieval work in this demo?",
  "gold_document_ids": ["doc_9c4e676fd6"],
  "gold_answer": "Hybrid retrieval combines vector recall and keyword recall with reranking.",
  "query_rewrite": false,
  "mode": "hybrid",
  "max_chunks": 5,
  "ranker_type": "rrf",
  "alpha": 0.6,
  "impact_factor": 60
}
```

最重要的字段：

- `question`
- `vector_db_id`
- `gold_document_ids`

可选字段：

- `question_type`
- `gold_answer`
- `mode`
- `query_rewrite`
- `max_chunks`
- `ranker_type`
- `alpha`
- `impact_factor`

## 运行方式

只评测检索：

```bash
python opengauss_ragflow_app/eval/run_eval.py \
  --dataset opengauss_ragflow_app/eval/sample_eval.jsonl \
  --app-base-url http://localhost:8787 \
  --task retrieval
```

同时评测检索和对话：

```bash
python opengauss_ragflow_app/eval/run_eval.py \
  --dataset opengauss_ragflow_app/eval/sample_eval.jsonl \
  --app-base-url http://localhost:8787 \
  --task both \
  --details-out opengauss_ragflow_app/eval/output.json
```

## 输出指标

- `retrieval_recall_at_k`
- `retrieval_mrr`
- `chat_exact_match`
- `chat_citation_hit_rate`
- `avg_retrieval_latency_ms`
- `avg_chat_latency_ms`
- `by_question_type`

## 指标原理

- `Recall@K`
  - gold 文档是否出现在 top-k 检索结果里
  - 衡量“召回到了没有”
- `MRR`
  - 第一个正确结果排得越靠前，分数越高
  - 衡量“排得好不好”
- `Citation Hit Rate`
  - 回答里引用的文档是否命中 gold supporting document
  - 衡量“引用靠不靠谱”
- `Exact Match`
  - 回答是否与标准答案完全一致
  - 适合短答案或定义型问题
- `Latency`
  - 检索和对话响应耗时
  - 衡量系统实用性

## 适合写在简历上的实验方向

建议至少比较这几组：

1. `vector`
2. `keyword`
3. `hybrid`
4. `rewrite + hybrid`

然后按问题类型分桶看：

- 事实型问题
- 定义型问题
- 流程型问题
- 多跳问题

最值得写的不是“做了评测”，而是：

- 哪种检索策略在什么问题类型上 Recall 更高
- rewrite 是否提升了多跳问题的 supporting chunk 命中率
- citation 命中率是否提升了回答可追溯性
