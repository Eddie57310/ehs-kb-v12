"""为排雷库案例块自动裁剪右侧蓝框配图（V1.0/V2.0 等仍标 [📷 待补] 的块）。

每页版式固定：左侧粉色字段表 + 右侧一个蓝/靛色边框的框，框内是配图+图注。
本脚本检测该蓝框，裁剪框内内容存到 user_crops/，把块正文里的
[📷 待补] 替换成真实图标签 [📷 user_crops/xxx.png]，并更新 index.json。

用法:
  python add_case_screenshots.py <source子串>            # dry-run：只检测+存预览，不写库
  python add_case_screenshots.py <source子串> --write    # 裁剪+写库(重嵌入)+更新index+重建BM25
可选: --preview-dir <dir>  dry-run 时把裁剪预览存到该目录（默认 _crop_preview）
"""
import sys, os, re, time, json, pickle
import numpy as np
from PIL import Image
import chromadb

BASE        = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR   = os.path.join(BASE, "user_crops")
CROPS_INDEX = os.path.join(CROPS_DIR, "index.json")
BM25_PKL    = os.path.join(BASE, "bm25_index.pkl")
PAD         = 6          # 裁剪时向内缩进，避开蓝边框线本身
PLACEHOLDER = "[📷 待补]"


def frame_mask(im: np.ndarray) -> np.ndarray:
    """蓝/靛色边框像素掩码（B 明显高于 R、G，且非高亮）。"""
    R, G, B = im[:, :, 0].astype(int), im[:, :, 1].astype(int), im[:, :, 2].astype(int)
    return (B - R > 25) & (B - G > 15) & (B > 80) & (R < 170)


def detect_box(pil: Image.Image):
    """检测右侧蓝框，返回 (left, top, right, bottom) 像素框；检测不到返回 None。

    依据：边框的左右竖线是贯穿整个框高的连续蓝色线（投影计数接近框高），
    远强于图片内部的零散蓝色（天空/窗户等）。
    """
    im = np.array(pil.convert("RGB"))
    h, w, _ = im.shape
    m = frame_mask(im)

    # 竖线：在右侧 40%~100% 列里，找蓝色像素占列高 >55% 的列
    colc = m.sum(axis=0)
    cand_cols = [x for x in range(int(w * 0.40), w) if colc[x] > 0.55 * h]
    if not cand_cols:
        return None
    left, right = min(cand_cols), max(cand_cols)
    if right - left < 0.20 * w:           # 太窄，不是真正的框
        return None

    # 横线：在 [left,right] 区间内，找蓝色像素占框宽 >55% 的行
    sub = m[:, left:right + 1]
    rowc = sub.sum(axis=1)
    span = right - left
    cand_rows = [y for y in range(h) if rowc[y] > 0.55 * span]
    if not cand_rows:
        return None
    top, bot = min(cand_rows), max(cand_rows)
    if bot - top < 0.20 * h:
        return None
    return (left, top, right, bot)


def crop_interior(pil: Image.Image, box):
    l, t, r, b = box
    return pil.crop((l + PAD, t + PAD, r - PAD, b - PAD))


def load_index() -> dict:
    if os.path.exists(CROPS_INDEX):
        try:
            with open(CROPS_INDEX, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_index(idx: dict):
    with open(CROPS_INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def rebuild_bm25(col):
    """与 chunk_reviewer._rebuild_bm25_async 一致：pickle 出 list(zip(docs, metas))。"""
    r = col.get(include=["documents", "metadatas"])
    corpus = list(zip(r["documents"], r["metadatas"]))
    with open(BM25_PKL + ".tmp", "wb") as f:
        pickle.dump(corpus, f)
    os.replace(BM25_PKL + ".tmp", BM25_PKL)
    print(f"✅ BM25 重建：{len(corpus)} 条 -> {os.path.basename(BM25_PKL)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src_sub = sys.argv[1]
    write = "--write" in sys.argv
    preview_dir = os.path.join(BASE, "_crop_preview")
    if "--preview-dir" in sys.argv:
        preview_dir = sys.argv[sys.argv.index("--preview-dir") + 1]

    c = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db"))
    col = c.get_collection(c.list_collections()[0].name)

    # 找到匹配的 source（可用子串）
    allr = col.get(include=["metadatas"])
    sources = sorted({m.get("source", "") for m in allr["metadatas"]})
    matched = [s for s in sources if src_sub in s]
    if not matched:
        print(f"❌ 没有 source 含 '{src_sub}'。可选：")
        for s in sources:
            print("  ", s)
        sys.exit(1)

    if not write:
        os.makedirs(preview_dir, exist_ok=True)

    grand_ok = grand_fail = 0
    todo_for_write = []   # (id, new_doc, source, page, rel_path)

    for src in matched:
        r = col.get(where={"source": src},
                    include=["documents", "metadatas"])
        stem = os.path.splitext(os.path.basename(src))[0]
        pend = [(i, d, m) for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])
                if PLACEHOLDER in d]
        print(f"\n===== {src}")
        print(f"      共 {len(r['ids'])} 块，待补 {len(pend)} 块")
        ok = fail = 0
        for _id, doc, meta in pend:
            page = meta.get("page")
            img_rel = meta.get("image_path", "")
            slide = os.path.join(BASE, img_rel) if img_rel else ""
            if not slide or not os.path.exists(slide):
                print(f"   ⚠️ p{page}: 找不到幻灯片图 {img_rel}")
                fail += 1
                continue
            pil = Image.open(slide)
            box = detect_box(pil)
            if box is None:
                print(f"   ⚠️ p{page}: 未检测到蓝框，跳过（需人工）")
                fail += 1
                continue
            w, h = pil.size
            l, t, rr, b = box
            frac = [round(l / w, 3), round(t / h, 3), round(rr / w, 3), round(b / h, 3)]
            crop = crop_interior(pil, box)
            if write:
                ts = int(time.time() * 1000) % 1_000_000
                fname = f"{stem}_p{page}_{ts}.png"
                crop.save(os.path.join(CROPS_DIR, fname), "PNG")
                rel = f"user_crops/{fname}"
                new_doc = doc.replace(PLACEHOLDER, f"[📷 {rel}]")
                todo_for_write.append((_id, new_doc, src, page, rel))
                print(f"   ✅ p{page}: {frac} -> {fname} {crop.size}")
            else:
                crop.save(os.path.join(preview_dir, f"{stem}_p{page}.png"), "PNG")
                print(f"   ✅ p{page}: {frac} {crop.size}  (预览)")
            ok += 1
            time.sleep(0.002)   # 保证 ts 不撞
        print(f"   小计：成功 {ok}  失败 {fail}")
        grand_ok += ok
        grand_fail += fail

    print(f"\n总计：成功 {grand_ok}  失败 {grand_fail}")

    if not write:
        print(f"（dry-run）预览图在 {preview_dir}/，确认无误后加 --write 执行")
        return

    if not todo_for_write:
        print("无可写入项。")
        return

    # 1) 重嵌入并写回 ChromaDB
    print("\n加载 bge-m3 重新嵌入更新块…")
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                                model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})
    docs = [d for _, d, _, _, _ in todo_for_write]
    vecs = emb.embed_documents(docs)
    col.update(ids=[i for i, _, _, _, _ in todo_for_write],
               documents=docs, embeddings=vecs)
    print(f"✅ 写回 ChromaDB：{len(todo_for_write)} 块（已替换图标签 + 重嵌入）")

    # 2) 更新 user_crops/index.json（key = source::p页码，供 8085 复核面板显示）
    idx = load_index()
    for _id, _doc, src, page, rel in todo_for_write:
        key = f"{src}::p{page}"
        fname = os.path.basename(rel)
        idx[key] = [{"tag": f"[📷 {rel}]", "fname": fname, "path": rel}]
    save_index(idx)
    print(f"✅ 更新 index.json：{len(todo_for_write)} 个 key")

    # 3) 重建 BM25
    rebuild_bm25(col)
    print("\n全部完成。8085 重新选择该文件即可看到新配图；服务读新 BM25。")


if __name__ == "__main__":
    main()
