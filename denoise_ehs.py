"""通用噪音清理：删 EHS 块里的页眉/版权/页码/地址等确定垃圾行。安全普适,不动语义。
用法: python denoise_ehs.py          # dry,列出所有将删的行(去重)供自查
      python denoise_ehs.py --write   # 写库(update同id+重编码)
"""
import sys, re, chromadb
from collections import Counter


def is_noise(s):
    s = s.strip()
    if not s:
        return False
    if s.startswith('CRLand') or 'CRLand East China Region' in s:
        return True
    if s == '华润 陈嘉 版权所有':
        return True
    if s in ('华润置地', 'CR LAND', 'CRLAND', 'CRLand', 'CR  LAND'):
        return True
    if s.startswith('核心商密'):
        return True
    if 'China Resources Land' in s or 'China Resources Building' in s:
        return True
    if s.startswith('中国·') and ('路' in s or 'Road' in s):
        return True
    if s.startswith('电话Tel') or s.startswith('Http') or s.startswith('网址'):
        return True
    if re.match(r'^华润置地华东大区杭州公司\s*(China|$)', s):
        return True
    if s.isdigit() and len(s) <= 3:          # 孤立页码
        return True
    return False
    # 注：### / #### 是 Vision LLM 的 markdown 结构,混了版面噪音与正文标题,
    #     无法简单按前缀删(会误删 "#### 1.易燃可燃物" 这类正文),留待精细规则


def clean(doc):
    out = [l for l in doc.splitlines() if not is_noise(l)]
    # 压缩开头空行
    while out and not out[0].strip():
        out.pop(0)
    return '\n'.join(out)


def main():
    write = '--write' in sys.argv
    c = chromadb.PersistentClient(path='chroma_db')
    col = c.get_collection(c.list_collections()[0].name)
    res = col.get(where={'domain': 'EHS案例'}, include=['documents', 'metadatas'])

    removed = Counter()
    upd_ids, upd_docs = [], []
    for i, d in zip(res['ids'], res['documents']):
        nd = clean(d)
        if nd != d:
            for l in d.splitlines():
                if is_noise(l):
                    removed[l.strip()[:40]] += 1
            upd_ids.append(i); upd_docs.append(nd)

    print(f'将改动 {len(upd_ids)}/{len(res["ids"])} 块')
    print(f'\n=== 将删除的不同行(去重, 共{len(removed)}种) 自查有无正文误删 ===')
    for line, n in removed.most_common():
        print(f'  x{n:<3} {line}')

    if write and upd_ids:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name='BAAI/bge-m3', model_kwargs={'device': 'cpu'})
        col.update(ids=upd_ids, documents=upd_docs, embeddings=emb.embed_documents(upd_docs))
        print(f'\n✅ 已清理 {len(upd_ids)} 块 + 重编码')


if __name__ == '__main__':
    main()
