"""自动复刻用户的整排截图：对含 [📷 待补] 的页，按图纵向聚类(被文字断开则分组),
每组取 bbox 并集裁剪整排图(下扩含图注)。复刻自 EHS工具箱 p3-p6 的人工截法。
用法: python ehs_autocrop.py <source>          # dry,图存 /tmp/ehs_crops,不动库
      python ehs_autocrop.py <source> --write   # 图存 user_crops + 更新块[📷待补]→真实路径
"""
import fitz, sys, os, random, chromadb

GAP = 40        # 图纵向间隔 > GAP(点) 视为被文字断开 → 分组
ZOOM = 2
MIN = 50        # 过滤 logo/页脚条：宽高任一 < MIN 丢弃


def clusters_of(boxes):
    boxes = sorted(boxes, key=lambda b: b[1])
    cl, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if b[1] - max(x[3] for x in cur) > GAP:
            cl.append(cur); cur = [b]
        else:
            cur.append(b)
    cl.append(cur)
    return cl


def main():
    src = sys.argv[1]
    write = '--write' in sys.argv
    skip = set()
    if '--skip' in sys.argv:
        skip = {int(x) for x in sys.argv[sys.argv.index('--skip') + 1].split(',')}
    pad_bottom = 34
    if '--pad' in sys.argv:
        pad_bottom = int(sys.argv[sys.argv.index('--pad') + 1])
    pdf = os.path.join('Local_KB', src)
    stem = os.path.basename(src).replace('.pdf', '')
    outdir = 'user_crops' if write else '/tmp/ehs_crops'
    os.makedirs(outdir, exist_ok=True)
    doc = fitz.open(pdf)

    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)
    r = col.get(where={'source': src}, include=['documents', 'metadatas'])
    todo = {m.get('page'): (i, d) for i, d, m in zip(r['ids'], r['documents'], r['metadatas'])
            if '[📷 待补]' in d and m.get('page')}

    upd_ids, upd_docs, regs = [], [], []
    for pno in sorted(todo):
        if pno in skip:
            print(f'page {pno}: ⏭️ 跳过(留[📷待补],人工处理)')
            continue
        page = doc[pno - 1]
        H, W = page.rect.height, page.rect.width
        boxes = [im['bbox'] for im in page.get_image_info(xrefs=True)
                 if (im['bbox'][2] - im['bbox'][0]) > MIN and (im['bbox'][3] - im['bbox'][1]) > MIN]
        if not boxes:
            print(f'page {pno}: ⚠️ 无大图,跳过(留[📷待补])')
            continue
        # 剔除面积 < 该页最大图 50% 的孤立小图(避免把旁边正文一起圈进裁剪框)
        maxa = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= 0.5 * maxa]
        groups = clusters_of(boxes)
        tags = []
        for g in groups:
            x0 = min(b[0] for b in g); y0 = min(b[1] for b in g)
            x1 = max(b[2] for b in g); y1 = max(b[3] for b in g)
            clip = fitz.Rect(max(0, x0 - 12), max(0, y0 - 4), min(W, x1 + 12), min(H, y1 + pad_bottom))
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=clip)
            fn = f'{stem}_p{pno}_{random.randint(100000, 999999)}.png'
            pix.save(os.path.join(outdir, fn))
            tags.append(f'[📷 user_crops/{fn}]')
            if write:
                regs.append((pno, fn))
        yr = [f"{int(min(b[1] for b in g))}-{int(max(b[3] for b in g))}" for g in groups]
        print(f'page {pno}: {len(groups)} 组 → {len(tags)} 张  y范围={yr}')
        _id, d = todo[pno]
        upd_ids.append(_id); upd_docs.append(d.replace('[📷 待补]', '\n'.join(tags)))

    print(f'\n图片输出目录: {outdir}')
    if write and upd_ids:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})
        col.update(ids=upd_ids, documents=upd_docs, embeddings=emb.embed_documents(upd_docs))
        print(f'✅ 更新 {len(upd_ids)} 块([📷待补]→真实路径)+重编码')
        # 登记 chunk_reviewer 的 index.json(key=source::p页),否则 8085 不显示预览
        import json as _json
        IDX = os.path.join('user_crops', 'index.json')
        idx = _json.load(open(IDX, encoding='utf-8')) if os.path.exists(IDX) else {}
        for pno, fn in regs:
            rel = f'user_crops/{fn}'
            idx.setdefault(f'{src}::p{pno}', []).append({'tag': f'[📷 {rel}]', 'fname': fn, 'path': rel})
        _json.dump(idx, open(IDX, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print(f'✅ 登记 {len(regs)} 张到 index.json(8085 可预览)')


if __name__ == '__main__':
    main()
