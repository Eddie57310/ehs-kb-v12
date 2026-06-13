"""按用户在 EHS工具箱 page3-6 的改法，泛化整改同类块。
规律：删头部噪音(CRLand/华润 陈嘉 版权所有/核心商密) → 标题行后加空行 → 正文保留 → 末尾 [📷 待补]
跳过 page 2(保留) 和已改块(含[📷])。
用法: python rework_ehs.py <source>          # dry,生成预览
      python rework_ehs.py <source> --write   # 写库(update同id+重编码)
"""
import sys, chromadb

def rework(doc):
    kept = []
    for l in doc.splitlines():
        s = l.strip()
        if not s:
            continue
        if s.startswith('CRLand East China Region'):
            continue
        if s == '华润 陈嘉 版权所有':
            continue
        if s.startswith('核心商密'):
            continue
        kept.append(s)
    if not kept:
        return None
    title, body = kept[0], kept[1:]
    if '|' in title and body:          # 标题行后加空行
        txt = title + '\n\n' + '\n'.join(body)
    else:
        txt = '\n'.join(kept)
    return txt.rstrip() + '\n[📷 待补]'


def main():
    src = sys.argv[1]
    write = '--write' in sys.argv
    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)
    r = col.get(where={'source': src}, include=['documents', 'metadatas'])

    todo = []
    for i, d, m in zip(r['ids'], r['documents'], r['metadatas']):
        if m.get('page') == 2:           # 保留
            continue
        if '[📷' in d:                    # 已改
            continue
        if not d.lstrip().startswith('CRLand'):
            continue
        todo.append((m.get('page'), i, d, rework(d)))
    todo.sort(key=lambda x: (x[0] is None, x[0] or 0))

    with open('_preview_ehs.md', 'w', encoding='utf-8') as f:
        for pg, i, before, after in todo:
            f.write(f'======== page {pg} ========\n--- BEFORE ---\n{before}\n--- AFTER ---\n{after}\n\n')
    print(f'待改 {len(todo)} 块 → 预览 _preview_ehs.md')
    print('\n===== 试跑前 2 块 =====')
    for pg, i, before, after in todo[:2]:
        print(f'######## page {pg} ########')
        print('--- BEFORE ---'); print(before)
        print('--- AFTER ---'); print(after); print()

    if write:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})
        ids = [i for _, i, _, _ in todo]
        docs = [a for _, _, _, a in todo]
        vecs = emb.embed_documents(docs)
        col.update(ids=ids, documents=docs, embeddings=vecs)
        print(f'\n✅ 写回 {len(ids)} 块(update同id+重编码)')


if __name__ == '__main__':
    main()
