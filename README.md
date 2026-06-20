# ehs-kb-v12 — EHS 知识库问答机器人（飞书 / 企业微信）

v11 的演进版：同一套 RAG 问答（**混合检索 ChromaDB 向量 + BM25 → cross-encoder 精排
→ LLM 作答**），在 v11 基础上补全了 deepseek 分支、企业微信入口等。

## 架构
```
文档(reviewed_md) ──切块──▶ ChromaDB(bge-m3 向量) + BM25 索引
                                       │
用户提问 ─▶ 查询扩展 ─▶ 混合检索 ─▶ 精排(共享 rerank 服务) ─▶ LLM(deepseek 等) ─▶ 飞书/企微卡片
```

## 关键组件
- `index_reviewed_md.py` — 把 `reviewed_md/` 切块写入 `chroma_db/`（`--force` 全量重建）
- `sync_kb_v11.py` — PDF / docx → 结构化切块入库
- `feishu_ws_server_v12.py` — 飞书长连接服务 + 检索 / 精排 / 作答主流程
- `wecom_server.py` — 企业微信入口
- 精排走**独立的共享 GPU rerank 服务** → [allnewv11](https://github.com/Eddie57310/allnewv11)
  （4GB 小显卡上 v11/v12 共用一份 reranker；本仓库通过 `http://127.0.0.1:8765` 调用，
  服务不可达时自动降级为向量 / BM25 融合排序）

## 运行
1. 配置 `.env`（`FEISHU_APP_ID/SECRET`、各 LLM key 等，**不入库**）
2. 起共享 rerank 服务：见 allnewv11 的 `start_rerank.sh`
3. 起机器人：`bash start_feishu.sh`（飞书）/ `bash start_wecom.sh`（企微）

> 详细设计 / 入库流程 / 运维见 [`SYSTEM_DOC.md`](SYSTEM_DOC.md)。

## 说明
机密知识库（`reviewed_md/`、`Local_KB/`、`chroma_db/`、`qa_logs/`）与 `.env`
均已 `.gitignore`，不进仓库。
