import os, re, json, shutil, time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_DIR      = os.path.join(BASE_DIR, "chroma_db")
CROPS_DIR   = os.path.join(BASE_DIR, "user_crops")
CROPS_INDEX = os.path.join(CROPS_DIR, "index.json")

os.makedirs(CROPS_DIR, exist_ok=True)

CROP_TAG_RE = re.compile(r'\[📷[^\]]*\]')

SOURCES = {
    "设计部/案例/20241001_惊喜库v1.pptx": "20241001_惊喜库v1",
    "设计部/案例/20241101_惊喜库v2.pdf":  "20241101_惊喜库v2",
    "设计部/案例/20241201_惊喜库v3.pptx": "20241201_惊喜库v3",
}

# ── 1. 加载嵌入模型 ────────────────────────────────────────────
print("加载嵌入模型...")
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
print("模型加载完成")

# ── 2. 读取 index.json ──────────────────────────────────────────
def load_index():
    if not os.path.exists(CROPS_INDEX):
        return {}
    with open(CROPS_INDEX, encoding="utf-8") as f:
        return json.load(f)

def save_index(idx):
    with open(CROPS_INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

# ── 3. 主循环 ───────────────────────────────────────────────────
for source, stem in SOURCES.items():
    print(f"\n处理: {source}")
    results = db._collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
    )
    stats = {"reuse": 0, "new": 0, "fix_only": 0, "skip": 0, "warn": 0}

    for chunk_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        if not meta:
            print(f"  ⚠️  chunk {chunk_id} 无 metadata，跳过")
            stats["warn"] += 1
            continue

        page = meta.get("page")
        if page is None:
            stats["skip"] += 1
            continue

        slide_src = os.path.join(BASE_DIR, f"slide_images/{stem}/p{page:04d}.png")
        if not os.path.exists(slide_src):
            print(f"  ⚠️  幻灯片图不存在: slide_images/{stem}/p{page:04d}.png")
            stats["warn"] += 1
            continue

        index_key = f"{source}::p{page}"
        idx = load_index()

        # 判断 index.json 是否已有有效条目
        existing = idx.get(index_key, [])
        valid_existing = [
            c for c in existing
            if os.path.exists(os.path.join(BASE_DIR, c["path"]))
        ]

        if valid_existing:
            # 已有有效条目：复用，只检查 chunk 文本是否需要修正
            correct_tag = valid_existing[0]["tag"]
            stats["reuse"] += 1
        else:
            # 需要新建：复制幻灯片图到 user_crops
            ts    = int(time.time() * 1000) % 1_000_000
            fname = f"{stem}_p{page}_{ts}.png"
            dest  = os.path.join(CROPS_DIR, fname)
            shutil.copy(slide_src, dest)

            path  = f"user_crops/{fname}"
            correct_tag = f"[📷 {path}]"
            idx[index_key] = [{"tag": correct_tag, "fname": fname, "path": path}]
            save_index(idx)
            stats["new"] += 1
            time.sleep(0.002)  # 确保下一个 chunk 的 ts 不重复

        # 检查 chunk 文本是否已经包含正确 tag
        if correct_tag in doc:
            continue  # 已经正确，不动 ChromaDB

        # 修正 chunk 文本：删旧标记，追加正确标记
        new_doc = CROP_TAG_RE.sub("", doc).rstrip()
        new_doc = new_doc + f"\n\n{correct_tag}"
        stats["fix_only"] += 1

        # 更新 ChromaDB（delete + add，让向量重新嵌入）
        updated_meta = {**meta, "modified_at": time.strftime("%Y-%m-%d %H:%M")}
        db._collection.delete(ids=[chunk_id])
        db.add_texts(texts=[new_doc], metadatas=[updated_meta])

    print(f"  复用已有: {stats['reuse']}  新建: {stats['new']}  仅修文本: {stats['fix_only']}  跳过: {stats['skip']}  警告: {stats['warn']}")

# ── 4. 重建 BM25 ────────────────────────────────────────────────
print("\n重建 BM25 索引...")
from chunk_reviewer import _trigger_bm25_rebuild
_trigger_bm25_rebuild()
time.sleep(3)
# 再确认写入完毕
if os.path.exists(os.path.join(BASE_DIR, "bm25_index.pkl")):
    print("✅ BM25 索引已就绪")
else:
    print("⚠️  BM25 索引文件未出现，稍后需手动触发")
print("完成")
