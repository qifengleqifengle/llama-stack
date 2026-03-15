# 项目简历表述与问答说明

## 简历部分

### 项目名称

RAG 知识库检索系统

### 技术栈

`Python`、`FastAPI`、`OpenGauss 向量检索适配设计`、`PostgreSQL`、`FTS + GIN`、`Llama Stack`、`OpenAI-compatible API`、`MinerU`

### 项目描述

设计并实现一套面向技术文档场景的 RAG 知识库检索系统，完成文档解析入库、向量检索、关键词检索、混合检索、查询改写与评测脚手架设计，重点优化技术问答场景下的召回质量与结果可解释性。

### 核心亮点

- 设计数据库侧检索抽象，支持向量检索、关键词检索与混合检索三种模式，构建 `vector / keyword / hybrid` 多路召回能力。
- 基于 `FTS + GIN` 实现关键词检索，结合向量检索结果进行 `RRF / Weighted` 混合重排，提升技术文档场景下的召回稳定性。
- 接入 `MinerU` 完成 PDF/文档结构化解析，将解析结果转为 Markdown 后入库，支持结构化分块与文档级删除。
- 实现多策略分块机制，支持 `结构分块`、`固定长度分块`、`固定字符分块`、`递归分块`，适配不同类型文档的入库需求。
- 实现 Query Rewrite，在检索前对用户问题进行面向召回的改写，用于提升复杂问题与多跳问题的检索命中率。
- 设计 RAG 评测脚手架，支持对不同检索策略进行量化比较，覆盖 `Recall@K`、`MRR`、`Citation Hit Rate`、`Latency` 等核心指标。

### 推荐口径

如果面试官追问数据库实现，建议统一表述为：

- 项目做了面向 OpenGauss 的检索适配设计与方案验证。
- 稳定联调和最终链路验证在 PostgreSQL 兼容栈上完成。
- 关键词检索当前落地方案是 `FTS + GIN`，不是数据库原生 BM25。

不要写成：

- 已在 OpenGauss 上完整落地 BM25
- 已实现 OpenGauss 原生 BM25 检索

这两句和当前代码事实不一致，面试风险过高。

---

## 问题说明

### 1. 为什么不建议把当前项目写成 OpenGauss 原生 BM25 已落地

因为当前项目稳定跑通的事实是：

- 向量检索链路已落地
- 关键词检索当前落地方案是 `FTS + GIN`
- 混合检索是 `vector + keyword + rerank`

也就是说，你可以说：

- 做了 OpenGauss 检索适配设计
- 调研过 OpenGauss/BM25 方案
- 最终稳定验证采用了 PostgreSQL 兼容栈

但不能说：

- 项目已经在 OpenGauss 上实现并验证了原生 BM25

这是两件不同的事。

### 2. Query Rewrite 做了什么

Query Rewrite 是检索前预处理，不是直接回答问题。

流程是：

1. 用户问题进来
2. 如果开启 `query_rewrite`
3. 先把原问题改写成一条更适合检索的查询
4. 再用改写后的 query 做向量检索、关键词检索或混合检索
5. 最后把检索结果交给回答模型生成 grounded answer

它的目标是：

- 保留原始问题意图
- 保留实体名、版本号、技术术语
- 去掉冗余表达
- 提升复杂问题、多跳问题和口语化问题的召回质量

一句话：

- Query Rewrite 是检索前面向召回的查询标准化。

### 3. 评测脚手架的原理

RAG 评测要拆成两层：

- 检索层
- 回答层

当前项目的评测脚手架支持的核心指标如下。

#### Recall@K

含义：

- 正确 supporting document 或 supporting chunk 是否出现在 top-k 结果中

作用：

- 衡量系统有没有把正确内容召回出来

#### MRR

含义：

- 第一个正确结果的排名倒数

作用：

- 衡量正确结果排得是否足够靠前

#### Citation Hit Rate

含义：

- 模型回答引用到的文档或段落，是否命中 gold supporting source

作用：

- 衡量回答是否真正建立在正确证据上

#### Exact Match / Accuracy

含义：

- 回答是否和标准答案一致，或者是否被人工判定为正确

作用：

- 衡量最终用户视角下的回答质量

#### Latency

含义：

- 检索耗时和对话耗时

作用：

- 衡量系统是否具有实际可用性

### 4. 为什么要比较 vector / keyword / hybrid / rewrite+hybrid

因为不同问题类型对检索方式的依赖不同。

#### vector

- 适合语义相近但词面不完全重合的问题

#### keyword

- 适合专有名词、版本号、活动名、实体词

#### hybrid

- 结合语义召回和精确词面召回
- 一般比单路召回更稳

#### rewrite + hybrid

- 对口语化、冗长或多跳问题，先改写再检索
- 理论上更容易提升 supporting chunk 命中率

### 5. 为什么要做多策略分块

因为单一分块方式对所有文档都不合适。

#### 结构分块

- 适合 MinerU 解析出的 Markdown 文档
- 能保留标题与段落结构

#### 固定长度分块

- 适合普通长文本
- 简单直接，便于控制上下文大小

#### 固定字符分块

- 适合字符型内容或不方便按 token 粗估的场景

#### 递归分块

- 先按结构切
- 过长再继续向更细粒度递归拆分

一句话：

- chunking 决定知识如何被索引和召回，是 RAG 效果的上游变量。

### 6. 如果后面真要往 OpenGauss/BM25 方向补，应该怎么说

可以说：

- 已完成 OpenGauss 检索适配设计与方案调研，当前稳定落地方案为 PostgreSQL 兼容栈上的 `FTS + GIN + hybrid rerank`

等你后面真的把 OpenGauss 原生 BM25 跑通，再把简历升级成：

- 在 OpenGauss 上落地数据库原生 BM25 + 向量混合检索

在那之前，不要提前写。

### 7. 推荐用什么文档做测试

如果要给这个项目准备一套比较像样、又不会太重的测试语料，我建议直接用 openGauss 官方文档。

最合适的不是整套全量文档，而是选 1 组主题明确的官方文档，控制在一个中等规模范围内，然后围绕它人工设计一批问题。

推荐优先选这几类：

- openGauss Getting Started
- openGauss Administrator Guide / Database Operations and Maintenance
- openGauss Application Development Guide
- openGauss Technical White Paper
- openGauss Release Notes

推荐原因：

- 都是官方文档，适合简历表述
- 内容结构清晰，适合做分块、检索和评测
- 既能设计定义型问题，也能设计流程型问题，还能设计少量多跳问题

最稳的做法是：

- 选 3 到 5 个主题章节
- 控制在一个中等规模文档集合
- 人工设计 20 到 30 个问题
- 按 `factoid / procedural / multi-hop` 分桶

这样就足够支持：

- vector
- keyword
- hybrid
- rewrite + hybrid

这四类策略的比较。

### 8. openGauss 官方有没有 PDF 文档

结论：

- openGauss 官方主文档体系目前能明确查到的是在线 HTML 文档站
- 没有查到一个清晰的、统一的官方“主文档 PDF 下载入口”
- 但部分工具文档明确提到自带 PDF，例如 DataStudio 文档写明二进制包 `docs` 目录里有《Data Studio 用户手册.pdf》

所以如果你要做项目评测语料，最现实的方式是：

- 直接使用 openGauss 官方在线文档页面
- 选择其中几个主题章节
- 自己导出为 PDF 或保存为离线文档后导入知识库

这样依然属于“基于 openGauss 官方文档做测试”，而且最容易落地。

### 9. 推荐测试语料与 20 道评测题

如果你想快速做一套像样的评测集，我建议直接使用 PostgreSQL 官方 PDF 手册中的这几章内容：

- Chapter 11. Indexes
- Chapter 12. Full Text Search
- Section 8.14. JSON Types
- Chapter 14. Performance Tips

这些章节足够覆盖：

- keyword 检索
- hybrid 检索
- 索引与排序
- 技术定义型问题
- 流程型问题
- 少量多跳问题

下面这 20 道题可以直接作为评测集模板。

#### A. Factoid / Definition（8 题）

1. PostgreSQL 全文检索中，`tsvector` 和 `tsquery` 分别表示什么？
2. PostgreSQL 为什么说 `LIKE` 和 `ILIKE` 不能替代现代全文检索？
3. 在 PostgreSQL 的全文检索里，什么是 lexeme？
4. `to_tsvector` 的作用是什么？
5. `plainto_tsquery` 和 `to_tsquery` 的主要区别是什么？
6. PostgreSQL 文档里提到的两种主要全文检索索引类型是什么？
7. `json` 和 `jsonb` 的核心区别是什么？
8. `EXPLAIN` 输出中的 `startup cost` 和 `total cost` 分别表示什么？

#### B. Procedural（8 题）

9. 如果想为经常检索的文本列建立全文检索索引，应该如何创建 GIN 索引？
10. 如何把普通文本转换成适合全文检索的 `tsvector`？
11. PostgreSQL 文档建议如何把用户输入转换成适合检索的查询表达式？
12. 如果要搜索短语而不是独立词项，应该使用什么方式构造查询？
13. 在 PostgreSQL 里，如何利用 `EXPLAIN ANALYZE` 检查规划器估计和真实执行的差异？
14. 如果 JSON 数据需要高效查询，官方文档更推荐使用 `json` 还是 `jsonb`，为什么？
15. 当要检查 `jsonb` 中某个键或键值是否存在时，文档建议使用什么类型的索引？
16. 如果一个查询经常使用等值和范围条件，应该优先考虑哪种索引类型？

#### C. Multi-hop / Comparison（4 题）

17. GIN 和 GiST 在全文检索里的区别是什么，为什么文档说 GIN 是更偏好的索引类型？
18. 为什么 `to_tsvector('fat cats ate fat rats') @@ to_tsquery('fat & rat')` 能匹配，但直接把原始文本强转成 `tsvector` 可能无法匹配？
19. 如果一个系统同时需要处理技术文档检索和 JSON 数据检索，PostgreSQL 文档分别推荐什么索引思路？
20. 从 PostgreSQL 文档的角度看，为什么一个查询即使已经建了索引，规划器仍然可能不选择索引扫描？

#### 出题覆盖逻辑

这 20 道题可以按下面的思路映射到评测指标：

- `factoid`
  - 更适合看 keyword / hybrid
- `procedural`
  - 更适合看 vector / hybrid
- `multi-hop`
  - 更适合看 rewrite + hybrid

如果要做最小实验，可以用这 20 道题分别比较：

1. `vector`
2. `keyword`
3. `hybrid`
4. `rewrite + hybrid`

然后看：

- `Recall@5`
- `MRR`

这样已经足够形成一套能讲清楚的 RAG 检索评测方案。

---

## 量化测试逐字稿

如果面试官问我这个项目怎么做量化测试，我会这样回答：

“我在这个项目里把 RAG 的评测重点放在检索层，因为 RAG 的上限首先取决于系统有没有把正确证据召回出来。

在检索层，我重点关注两个指标。第一个是 Recall@K，也就是正确 supporting document 或 supporting chunk 有没有出现在 top-k 结果里，这个指标用来衡量召回能力。第二个是 MRR，也就是第一个正确结果排得靠不靠前，这个指标用来衡量排序质量。因为在实际 RAG 场景里，模型只会看到前几个 chunk，所以正确结果排位很重要。

在策略对比上，我会比较四组：vector、keyword、hybrid，以及 rewrite + hybrid。原因是不同问题类型对检索方式依赖不同。比如语义型问题更适合 vector，专有名词和版本号更适合 keyword，而 hybrid 通常更稳。对于复杂问题和多跳问题，我会先做 Query Rewrite，把用户问题改写成更适合检索的查询，再做 hybrid，观察 supporting chunk 命中率有没有提升。

如果进一步细化，我会把问题按类型分桶，比如 factoid、definition、procedural、multi-hop，然后分别比较这些策略在不同问题类型上的 Recall@K 和 MRR。这样我拿到的不是一个单一总分，而是能解释为什么某种策略在某类问题上更有效。

所以这个项目的量化思路，核心就是从召回和排序两个角度，系统比较不同检索策略的效果。” 

### 可直接替换的示例口径

下面这段只能作为表达模板，数字必须在你实际跑完评测后替换，不能直接当成真实结果使用。

“我选了一组技术文档作为知识库语料，围绕这组文档人工设计了 30 个问题，分成 factoid、procedural 和 multi-hop 三类，然后对 vector、keyword、hybrid 和 rewrite + hybrid 四种策略做对比。

在这组样本上，我主要看 Recall@5 和 MRR。整体上，hybrid 相比单独 vector 或 keyword 更稳，因为它同时保留了语义召回和精确词面召回。比如在技术术语比较多的问题上，keyword 的 Recall@5 通常会更高；在语义表达比较灵活的问题上，vector 的表现更好；而 hybrid 能把两边优势结合起来。

如果后面我跑出真实结果，我会把它写成类似这种形式：在 30 个问题的评测集上，vector 的 Recall@5 是 X，keyword 是 Y，hybrid 提升到 Z；MRR 上，hybrid 也高于单路召回。对于 multi-hop 问题，rewrite + hybrid 相比普通 hybrid 又进一步提升了正确 supporting document 的前排命中率。

我比较看重这种评测方式，因为它不是只给一个总分，而是能解释不同策略分别适合什么问题类型。” 

### 不写具体数字时的结果表述口径

如果不准备在简历或面试里写具体数字，可以使用下面这类方向性表述。

- 在自建技术文档评测集上，对 `vector`、`keyword`、`hybrid`、`rewrite + hybrid` 四种检索策略进行了对比分析，整体上 `hybrid` 相比单路召回在召回稳定性上更优。
- 在事实型问题上，关键词检索对专有名词、术语和版本号更敏感；在语义表达更灵活的问题上，向量检索更稳定；混合检索能够综合两者优势，减少单一路径的漏召情况。
- 在流程型和复杂问题上，引入 `Query Rewrite` 后，检索 query 与文档表述之间的匹配度更高，能够进一步改善正确文档的前排命中情况。
- 在文档切分策略上，结构化分块更有利于保留技术文档中的章节和段落语义，递归分块在长文本场景下能够兼顾语义完整性与召回粒度，因此整体检索效果更稳定。

### 面试可直接说的总结口径

“我对这个项目的检索层做了策略对比，结论不是单一路径绝对最好，而是不同问题类型对检索方式的敏感度不同。专有名词和术语类问题更适合 keyword，语义表达更灵活的问题更适合 vector，而 hybrid 在整体上最稳，因为它同时保留了语义召回和词面召回的优势。

另外我还加入了 Query Rewrite。它的价值主要不是回答问题，而是在检索前把用户问题转成更适合召回的表达，所以在复杂问题和多跳问题上，通常能进一步改善正确文档的前排命中情况。

从文档处理角度看，chunking 也是关键变量。结构化文档如果只做固定切块，容易打断语义边界；而结构分块和递归分块更有利于保留标题、段落和上下文关系，所以检索质量会更稳定。” 
