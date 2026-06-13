"""快照工具：把 ChromaDB 当前所有块原文带 id 存档，作为人工改动前的 before 基线。
用法: python snapshot_kb.py
排除已定稿的排雷库 V1/V2/V3 案例。每次运行生成带日期的快照目录。
"""
import chromadb, json, os, time

EXCLUDE = {
    "客关部/排雷库/排雷库案例/25050501_杭州公司武器库之排雷库V1.0.pdf",
    "客关部/排雷库/排雷库案例/20250926_杭州公司武器库之排雷库V2.0.pdf",
    "客关部/排雷库/排雷库案例/20260401_浙江公司武器库之排雷库V3.0.pdf",
}

def main():
    c = chromadb.PersistentClient(path="chroma_db")
    col = c.get_collection(c.list_collections()[0].name)
    res = col.get(include=["documents", "metadatas"])

    sources = {}
    excluded_n = 0
    for _id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        meta = meta or {}
        src = meta.get("source", "?")
        if src in EXCLUDE:
            excluded_n += 1
            continue
        sources.setdefault(src, []).append(
            {"id": _id, "chunk_seq": meta.get("chunk_seq"),
             "document": doc, "metadata": meta})
    # 每个 source 内按 chunk_seq 排序（None 视为大值排末尾）
    for v in sources.values():
        v.sort(key=lambda x: (x["chunk_seq"] is None, x["chunk_seq"] or 0))

    date = time.strftime("%Y%m%d")
    outdir = f"原文存档_{date}"
    os.makedirs(outdir, exist_ok=True)
    total = sum(len(v) for v in sources.values())
    snap = {
        "snapshot_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "人工改动前 before 基线；按 block id 关联，可对比/恢复",
        "total_sources": len(sources),
        "total_blocks": total,
        "excluded_sources": sorted(EXCLUDE),
        "excluded_blocks": excluded_n,
        "sources": sources,
    }
    path = os.path.join(outdir, f"snapshot_{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
    print(f"✅ 快照已存: {path}")
    print(f"   存档 {len(sources)} 个文件 / {total} 块（排除排雷库V1/V2/V3 共 {excluded_n} 块）")
    print(f"\n=== 按 domain 分布 ===")
    from collections import Counter
    dom = Counter((v[0]["metadata"] or {}).get("domain", "?") for v in sources.values())
    for d, n in dom.most_common():
        print(f"   {n:3d} 文件  domain={d}")


if __name__ == "__main__":
    main()
