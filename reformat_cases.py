"""把 V1.0/V2.0 排雷库案例块重排成 V3.0 字段格式（草稿）。
用法:
  python reformat_cases.py <source>            # 仅生成草稿 md，不写库
  python reformat_cases.py <source> --write    # 重排后写回 ChromaDB(逐块 update+重编码)
"""
import sys, re, chromadb

FIELDS = ['编号', '案例名称', '排雷节点', '问题分类', '风险描述',
          '风险原因', '预防建议', '涉及专业', '涉及项目', '产生后果']
# 产生后果之后出现这些词即视为图注，从该行起截断丢弃
IMG_HINT = ['照片', '效果图', '实景', '沙盘', '现场', '标注', '文本', '泛光',
            '规范要求', '测量', '缺陷', '图册', '总图', '大门', '满溢',
            '示意', '平面', '户型', '区位', '截图', '展示']
CODE_RE = re.compile(r'编号[:：]?\s*\n?\s*([A-Z]{2,}-[A-Z]+\d+)')


def is_header(s):
    return (s.startswith('核心商密') or ('华润置地' in s and '排雷库' in s)
            or s.isdigit() or not s)


def reformat(doc):
    """返回 (code, 重排文本) 或 None(非案例块)。"""
    if not CODE_RE.search(doc):
        return None
    fields = {}
    cur = None
    for raw in doc.splitlines():
        s = raw.strip()
        if is_header(s):
            continue
        matched = None
        for f in FIELDS:
            if s == f or s.startswith(f + '：') or s.startswith(f + ':'):
                matched = f
                rest = s[len(f):].lstrip('：:').strip()
                break
        if matched:
            cur = matched
            fields[cur] = [rest] if rest else []
        elif cur:
            fields[cur].append(s)
    # 产生后果：遇图注词截断
    if '产生后果' in fields:
        val = []
        for l in fields['产生后果']:
            if any(h in l for h in IMG_HINT):
                break
            val.append(l)
        fields['产生后果'] = val
    code = (fields.get('编号') or ['?'])[0]
    out = []
    for f in FIELDS:
        v = ''.join(fields.get(f, [])).strip()
        out.append(f'{f}：{v}')
    out.append('[📷 待补]')
    return code, '\n'.join(out)


def classify(doc):
    """clean=已审核(保留) / case=乱格式案例(重排) / noncase=非案例或碎块(删除)"""
    d = doc.lstrip()
    if '[📷' in doc or d.startswith('编号：') or d.startswith('编号:'):
        return 'clean'   # 已带真实图 或 已是重排格式
    if d.startswith('核心商密') and CODE_RE.search(doc):
        return 'case'    # 乱格式案例 → 重排
    return 'noncase'     # 清单分隔页 / 垃圾碎块 → 删除


def main():
    src = sys.argv[1]
    write = '--write' in sys.argv
    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)
    r = col.get(where={'source': src}, include=['documents', 'metadatas'])

    cases = []       # (code, id, new_text) 乱格式案例 → 重排
    noncase_ids = [] # 乱格式非案例 → 删除
    noncase_head = []
    clean = 0
    for _id, doc in zip(r['ids'], r['documents']):
        k = classify(doc)
        if k == 'clean':
            clean += 1
        elif k == 'case':
            code, txt = reformat(doc)
            cases.append((code, _id, txt))
        else:
            noncase_ids.append(_id)
            noncase_head.append(' / '.join([l for l in doc.splitlines() if l.strip()][:2])[:50])
    cases.sort(key=lambda x: x[0])

    tag = src.split('/')[-1].replace('.pdf', '')
    if cases:
        with open(f'_draft_{tag}.md', 'w', encoding='utf-8') as f:
            f.write(f'# {tag} 案例重排草稿（共 {len(cases)} 块）\n\n')
            for code, _id, txt in cases:
                f.write(txt + '\n\n---\n\n')
    print(f'分类: 保留(已审){clean}  重排{len(cases)}  删除(非案例){len(noncase_ids)}')
    for h in noncase_head:
        print(f'   待删: {h}')

    if write:
        if cases:
            from langchain_huggingface import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})
            vecs = emb.embed_documents([t for _, _, t in cases])
            col.update(ids=[i for _, i, _ in cases],
                       documents=[t for _, _, t in cases], embeddings=vecs)
            print(f'✅ 重排写回 {len(cases)} 块')
        if noncase_ids:
            col.delete(ids=noncase_ids)
            print(f'✅ 删除非案例块 {len(noncase_ids)} 个')


if __name__ == '__main__':
    main()
