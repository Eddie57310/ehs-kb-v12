"""正面·字段卡型 重排脚本。
用法:
  python rework_positive.py <source>                    # dry-run: 输出 before→after 对比
  python rework_positive.py <source> --write            # 重排后写回 ChromaDB(合并块,重编码)
  python rework_positive.py --checklist <source> [src2] # 只输出 checklist 一致性报告
"""
import sys, re, chromadb

# ── 字段解析 ──────────────────────────────────────────────
FIELD_KEYS = [
    '案例名称', '案例来源', '涉及相关方', '主要内容',
    '描述', '价值成效', '复制推广',
    '适用项目业态', '适用相关方', '适用开发阶段',
]
# 价值成效 的子字段
SUB_FIELDS = {
    '价值成效': ['解决问题', '管理提升'],
}

CHAPTER_RE = re.compile(r'^([一二三四五六七八九十]+)、(.+)$')
SUBCH_RE  = re.compile(r'^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+-(.+)$')
CASE_RE    = re.compile(r'➢\s*-\d+-\s*案例名称[:：]')


def is_field_start(line, key):
    """判断一行是否是字段名的起始行."""
    line = line.strip()
    return line == key or line.startswith(key + '：') or line.startswith(key + ':')


def parse_cases(doc_text):
    """从合并后的案例文本解析字段字典.
    Returns: dict of field→value (multi-line values joined by '')
    """
    fields = {}
    cur_key = None
    cur_lines = []
    chapter = ''
    subchapter = ''

    for raw in doc_text.splitlines():
        s = raw.rstrip('\r')
        stripped = s.strip()
        if not stripped:
            continue  # 跳过空行

        # 章节标题
        m = CHAPTER_RE.match(stripped)
        if m:
            chapter = f'{m.group(1)}、{m.group(2)}'
            continue
        m = SUBCH_RE.match(stripped)
        if m:
            subchapter = m.group(1)
            continue

        # 提取案例名称值 (➢ -N-案例名称：XXX)
        m = CASE_RE.match(stripped)
        if m:
            name_val = stripped[m.end():].strip()
            if not cur_key:  # 第一个字段, 直接设为案例名称
                fields['案例名称'] = [name_val]
                cur_key = '案例名称'
                cur_lines = [name_val]
            continue

        # 字段边界检测
        matched = None
        for fk in FIELD_KEYS:
            if is_field_start(stripped, fk):
                matched = fk
                break
        if matched:
            # 保存上一个字段
            if cur_key:
                fields[cur_key] = cur_lines
            cur_key = matched
            # 提取字段名后面的值（同行取值）
            rest = stripped
            for fk in FIELD_KEYS:
                if rest == fk:
                    rest = ''
                    break
                if rest.startswith(fk + '：'):
                    rest = rest[len(fk)+1:].strip()
                    break
                if rest.startswith(fk + ':'):
                    rest = rest[len(fk)+1:].strip()
                    break
            cur_lines = [rest] if rest else []
        elif cur_key:
            cur_lines.append(stripped)

    if cur_key:
        fields[cur_key] = cur_lines

    return fields, chapter, subchapter


def reformat(fields, chapter, subchapter):
    """按规则重排为精简文本."""
    out = []

    # 案例名称
    name_lines = fields.get('案例名称', [])
    out.append(f'案例名称：{" ".join(name_lines)}')

    # 所属
    parts = [p for p in [chapter, subchapter] if p]
    out.append(f'所属：{" > ".join(parts) if parts else "（未知）"}')

    # 案例来源
    src_lines = fields.get('案例来源', [])
    out.append(f'案例来源：{(" ".join(src_lines)).strip()}')

    # 涉及相关方
    party_lines = fields.get('涉及相关方', [])
    out.append(f'涉及相关方：{(" ".join(party_lines)).strip()}')

    # 描述 (核心字段)
    desc_lines = fields.get('描述', [])
    desc = ' '.join(desc_lines).strip()
    out.append(f'描述：{desc}')

    # 价值成效 → 子字段
    val_lines = fields.get('价值成效', [])
    val_text = '\n'.join(val_lines)
    # 解析子字段
    sub_vals = {}
    cur_sub = None
    sub_buf = []
    for raw in val_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        matched = None
        for sf in SUB_FIELDS.get('价值成效', []):
            if is_field_start(s, sf):
                matched = sf
                break
        if matched:
            if cur_sub:
                sub_vals[cur_sub] = sub_buf
            cur_sub = matched
            rest = s
            if rest == matched:
                rest = ''
            elif rest.startswith(matched + '：'):
                rest = rest[len(matched)+1:].strip()
            elif rest.startswith(matched + ':'):
                rest = rest[len(matched)+1:].strip()
            sub_buf = [rest] if rest else []
        elif cur_sub:
            sub_buf.append(s)
    if cur_sub:
        sub_vals[cur_sub] = sub_buf

    # 构造价值成效行
    vp_parts = []
    for sf in SUB_FIELDS.get('价值成效', []):
        val = ' '.join(sub_vals.get(sf, [])).strip()
        if val:
            vp_parts.append(f'{sf}：{val}')
    if vp_parts:
        out.append(f'价值成效：{"; ".join(vp_parts)}')
    else:
        out.append('价值成效：')

    # 图标记
    out.append('[📷 待补]')

    return '\n'.join(out)


def group_blocks_into_cases(ids, documents):
    """将连续块按案例边界分组.
    策略: 遇到含「案例名称」的块 → 新案例开始;
          不含「案例名称」的块 → 续接到上一个案例.
    """
    cases = []  # [(case_blocks_ids, case_blocks_texts)]
    cur_ids = []
    cur_docs = []

    for bid, bdoc in zip(ids, documents):
        # 检测该块是否是新的案例起始
        has_case_name = '案例名称' in bdoc
        if has_case_name:
            # 保存上一个 case
            if cur_docs:
                cases.append((list(cur_ids), list(cur_docs)))
            cur_ids = [bid]
            cur_docs = [bdoc]
        else:
            cur_docs.append(bdoc)
            cur_ids.append(bid)

    if cur_docs:
        cases.append((list(cur_ids), list(cur_docs)))

    return cases


def main():
    args = sys.argv[1:]

    # --checklist 模式
    if '--checklist' in args:
        args = [a for a in args if a != '--checklist']
        checklist_mode(args)
        return

    write = '--write' in args
    src = args[0] if args else None
    if not src:
        print('用法: python rework_positive.py <source> [--write]')
        return

    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)
    r = col.get(where={'source': src}, include=['documents', 'metadatas'])

    ids = list(r['ids'])
    docs = list(r['documents'])

    cases = group_blocks_into_cases(ids, docs)

    tag = src.split('/')[-1].replace('.pdf', '')
    print(f'# 文件: {tag}   原始块数: {len(docs)}   案例数: {len(cases)}\n')

    updates = []  # [(new_id, new_text)]

    for ci, (case_ids, case_docs) in enumerate(cases, 1):
        merged = '\n'.join(case_docs)
        fields, chapter, subchapter = parse_cases(merged)
        new_text = reformat(fields, chapter, subchapter)

        before_one = '\n'.join(case_docs)
        print(f'{"="*60}')
        print(f'案例 {ci}/{len(cases)}  原块数: {len(case_ids)}')
        print(f'{"-"*30}[ BEFORE ]{"-"*30}')
        print(before_one)
        print(f'{"-"*30}[ AFTER  ]{"-"*30}')
        print(new_text)
        print()

        # 用第一个块的 id 作为合并后的 id
        updates.append((case_ids[0], new_text))

    if write:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})
        new_texts = [t for _, t in updates]
        new_ids = [i for i, _ in updates]
        vecs = emb.embed_documents(new_texts)
        col.update(ids=new_ids, documents=new_texts, embeddings=vecs)
        print(f'✅ 重排写回 {len(updates)} 块')

        # 删除合并后多余的原块 id
        kept = set(new_ids)
        remove_ids = [bid for bid in ids if bid not in kept]
        if remove_ids:
            col.delete(ids=remove_ids)
            print(f'✅ 删除多余块 {len(remove_ids)} 个')
    else:
        print('（dry-run, 未写库。加 --write 写回。）')


def checklist_mode(sources):
    """打印 checklist 三项的原始值，比较一致性."""
    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)

    for src in sources:
        r = col.get(where={'source': src}, include=['documents'])
        ids = list(r['ids'])
        docs = list(r['documents'])
        cases = group_blocks_into_cases(ids, docs)
        tag = src.split('/')[-1].replace('.pdf', '')

        print(f'\n==== {tag} ({len(cases)} 案例) ====')

        for ci, (case_ids, case_docs) in enumerate(cases, 1):
            merged = '\n'.join(case_docs)
            fields, ch, subch = parse_cases(merged)

            for k in ['适用项目业态', '适用相关方', '适用开发阶段']:
                raw = ' '.join(fields.get(k, [])).replace('\n', ' | ')
                raw_compact = ' '.join(raw.split())
                print(f'  [案例{ci}] {k}: {raw_compact}')


if __name__ == '__main__':
    main()
