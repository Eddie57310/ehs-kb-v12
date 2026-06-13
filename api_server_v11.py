import os
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(title="V11 局域网工程 API 服务 (ChromaDB 向量检索)")

EMBEDDING_DEVICE = "cpu"
_DB_CACHE = {}

def _get_db(version: str = "v11"):
    if version in _DB_CACHE:
        return _DB_CACHE[version]
    db_dir = os.path.expanduser(f"~/doc_parser_{version}/chroma_db")
    if not os.path.isdir(db_dir):
        return None
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': EMBEDDING_DEVICE})
    db = Chroma(persist_directory=db_dir, embedding_function=emb)
    _DB_CACHE[version] = db
    return db

print("🔍 正在连接默认向量数据库 (v11)...")
_DB_CACHE["v11"] = _get_db("v11")
if _DB_CACHE["v11"]:
    print("✅ 知识库（ChromaDB v11）挂载完毕")

class QueryRequest(BaseModel):
    question: str
    model: str = "qwen2.5:32b"
    db: str = "v11"

@app.post("/api/ask")
def ask_local(req: QueryRequest):
    print(f"\n[网络请求] db={req.db} 正在思考: {req.question}")

    target_db = _get_db(req.db.lower())
    if target_db is None:
        return {"status": "error", "message": f"知识库版本 {req.db} 不存在或路径不可访问。可用版本: v11"}

    # ── ChromaDB 向量检索 ──
    docs_and_scores = target_db.similarity_search_with_score(req.question, k=20)
    valid_docs = [doc for doc, score in docs_and_scores if score < 1.5][:10]

    # 兜底：得分过滤后为空，取最相似的前5条
    if not valid_docs and docs_and_scores:
        valid_docs = [doc for doc, score in docs_and_scores[:5]]
        print(f"  ⚠️  得分过滤后为空，启用兜底取前5条（最低得分: {docs_and_scores[0][1]:.4f}）")

    if not valid_docs:
        print("  ⚠️ 未找到相关资料")
        return {"status": "success", "answer": "知识库中未找到足够相关的内容。"}

    print(f"  🧠 成功提纯 {len(valid_docs)} 段向量资料")

    context_str = "\n---\n".join([
        f"【来源: {d.metadata.get('source', '未知文件')} | 生效时间: {d.metadata.get('date_str', '未知时间')}】\n{d.page_content}"
        for d in valid_docs
    ])

    prompt = f"参考以下带有来源的工程资料：\n{context_str}\n\n请回答：{req.question}\n要求：绝不编造！务必明确写出引用的【来源】和【生效时间】。"

    try:
        res = requests.post("http://127.0.0.1:11434/api/generate",
                            json={"model": req.model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}},
                            timeout=300)
        return {"status": "success", "answer": res.json().get('response', '')}
    except Exception as e:
        print(f"  ❌ 模型调用失败: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
