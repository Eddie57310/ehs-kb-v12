"""
index_reviewed_md.py
将 reviewed_md/ 下的 .md 文件切片后写入 chroma_db。

用法：
  python3 index_reviewed_md.py            # 增量：只处理未入库的文件
  python3 index_reviewed_md.py --force    # 全量重建（先删旧向量）
  python3 index_reviewed_md.py --file 公司内部/总部制度文件/foo.md  # 单文件
"""

import os, re, sys, logging, time
from pathlib import Path
from datetime import datetime

BASE_DIR  = os.path.expanduser("~/doc_parser_v12")
MD_DIR    = os.path.join(BASE_DIR, "reviewed_md")
KB_DIR    = os.path.join(BASE_DIR, "Local_KB")
DB_DIR    = os.path.join(BASE_DIR, "chroma_db")
LOG_DIR   = os.path.join(BASE_DIR, "logs")

SKIP_DOMAINS = {"EHS案例"}
# 任意以"案例"结尾的子目录都跳过（案例原件走 chunk 审核流程，不经此入库）
CHUNK_SIZE   = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE   = 32

os.makedirs(DB_DIR,  exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, f"index_md_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


# ── 向量库（cpu 嵌入，避免与飞书机器人争 GPU）────────────────
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger.info("初始化嵌入模型（cpu）...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
logger.info("chroma_db 已连接")


# ── 日期提取（复用 sync_kb 逻辑）────────────────────────────

def _extract_time(filename: str, rel_path: str):
    def _parse(s):
        m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', s)
        if m:
            y, mo, d = m.groups()
            ds = f"{y}-{mo}-{d}"
            try:
                return int(datetime.strptime(ds, "%Y-%m-%d").timestamp()), ds
            except ValueError:
                pass
        return None

    r = _parse(filename)
    if r:
        return r
    for part in reversed(Path(rel_path).parent.parts):
        r = _parse(part)
        if r:
            return r
    return 946684800, "2000-01-01"


# ── 列表截断兜底 ──────────────────────────────────────────
# 相邻块若是从编号列表中间被切开的（前块以列表项 N 结尾、后块以列表项 N+1 开头），
# 合并回去。无论切断原因（标题误判 / 超长 size 切分）一律还原，覆盖“列表游程保护”意图。

_LIST_ITEM_RE = re.compile(r'^(\d{1,2})\s+\S')   # 行首“裸数字+空格+内容”=列表项；带点(5.1.6)/表格(|)不算

def _merge_split_enumerations(chunks: list[str]) -> list[str]:
    def _last_item(text):
        for ln in reversed(text.splitlines()):
            if ln.strip():
                m = _LIST_ITEM_RE.match(ln.strip())
                return int(m.group(1)) if m else None
        return None

    def _first_item(text):
        lines = text.splitlines()
        i = 0
        while i < len(lines) and (not lines[i].strip() or
              (lines[i].strip().startswith('【') and lines[i].strip().endswith('】'))):
            i += 1
        if i < len(lines):
            m = _LIST_ITEM_RE.match(lines[i].strip())
            if m:
                return int(m.group(1)), '\n'.join(lines[i:])
        return None, text

    if len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for cur in chunks[1:]:
        ln = _last_item(out[-1])
        fn, body = _first_item(cur)
        if ln is not None and fn is not None and fn == ln + 1:
            out[-1] = out[-1].rstrip() + '\n' + body     # 去掉后块面包屑前缀，拼回前块
        else:
            out.append(cur)
    return out


# ── 章节感知切块（复用 sync_kb 逻辑）────────────────────────

def _structured_chunks(md_content: str, max_size: int = CHUNK_SIZE) -> list[str]:
    """按 Markdown 标题 / 数字编号标题切块，每块带完整章节路径前缀。"""
    _sub = RecursiveCharacterTextSplitter(
        chunk_size=max_size, chunk_overlap=80,
        separators=["\n\n", "\n", "。", "；", "，", " "],
    )
    md_re  = re.compile(r'^(#{1,6})\s+(.+)$')
    # 数字编号标题：裸数字章节号(1 总则)与多级(5.1.5)都认；但标题以句末标点(。；，、)结尾的
    # 一律当正文——那是列表项/句子(如“10 六级或六级以上强风。”)，否则会把清单从中间切断。
    num_re = re.compile(
        r'^(\d+(?:\.\d+)*)\s+([A-Za-z\u4e00-\u9fa5][\u4e00-\u9fa5\w\s（）【】、，。：:]{1,60})\s*$'
    )

    def _detect(line):
        m = md_re.match(line)
        if m:
            return len(m.group(1)), m.group(2).strip()
        m = num_re.match(line)
        if m:
            title = m.group(2).strip()
            if title and title[-1] in '。；，、':     # 句末标点结尾 → 列表项/句子，非标题
                return None
            num = m.group(1)
            return num.count('.') + 1, f"{num} {title}"
        return None

    stack: list[tuple[int, str]] = []
    body:  list[str] = []
    chunks: list[str] = []

    def _flush():
        content = '\n'.join(ln for ln in body if ln.strip())
        if not content.strip():
            return
        prefix = ('【' + ' > '.join(h[1] for h in stack) + '】\n') if stack else ''
        full = prefix + content
        if len(full) <= max_size:
            chunks.append(full)
        else:
            for sub in _sub.split_text(content):
                chunks.append(prefix + sub)

    for line in md_content.splitlines():
        h = _detect(line.rstrip())
        if h:
            _flush()
            body = []
            lv, title = h
            while stack and stack[-1][0] >= lv:
                stack.pop()
            stack.append((lv, title))
        else:
            body.append(line)
    _flush()

    chunks = _merge_split_enumerations(chunks)
    return chunks if chunks else _sub.split_text(md_content)


# ── 已入库文件集合 ────────────────────────────────────────

def _indexed_sources() -> set[str]:
    """从 chroma_db 拉取所有已入库的 source 值（直接查 sqlite3，避免 HNSW 不一致崩溃）。"""
    try:
        import sqlite3 as _sqlite3
        _db_path = os.path.join(DB_DIR, "chroma.sqlite3")
        _conn = _sqlite3.connect(_db_path)
        rows = _conn.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source'"
        ).fetchall()
        _conn.close()
        return {r[0] for r in rows}
    except Exception as e:
        logger.warning(f"拉取已入库列表失败: {e}")
        return set()


# ── 索引文件判定：所在目录(Local_KB)下有"*案例"子目录者，即库索引清单 ──

def _is_index_file(rel_path: str) -> bool:
    kb_dir = os.path.join(KB_DIR, os.path.dirname(rel_path))
    if not os.path.isdir(kb_dir):
        return False
    return any(
        d.endswith("案例") and os.path.isdir(os.path.join(kb_dir, d))
        for d in os.listdir(kb_dir)
    )


# ── 核心：处理单个 .md 文件 ───────────────────────────────

def process_md_file(rel_path: str) -> int:
    """读取 reviewed_md/{rel_path}，切片后写入 chroma_db。返回写入块数。"""
    full_path = os.path.join(MD_DIR, rel_path)
    md_content = Path(full_path).read_text(encoding="utf-8")
    if not md_content.strip():
        logger.warning(f"  空文件，跳过: {rel_path}")
        return 0

    domain    = Path(rel_path).parts[0]
    filename  = os.path.basename(rel_path)
    timestamp, date_str = _extract_time(filename, rel_path)
    rel_dir   = os.path.dirname(rel_path)
    category  = rel_dir if rel_dir else "根目录"

    metadata = {
        "source":    rel_path,
        "timestamp": timestamp,
        "date_str":  date_str,
        "category":  category,
        "domain":    domain,
    }
    if _is_index_file(rel_path):
        metadata["is_index"] = True
        logger.info(f"  📑 识别为库索引清单 (is_index): {rel_path}")

    chunks = _structured_chunks(md_content)
    docs = [
        Document(page_content=chunk, metadata={**metadata, "chunk_seq": i})
        for i, chunk in enumerate(chunks, start=1)
    ]

    for i in range(0, len(docs), BATCH_SIZE):
        db.add_documents(docs[i:i + BATCH_SIZE])

    logger.info(f"  入库 {len(docs)} 块: {rel_path}")
    return len(docs)


# ── 主流程 ────────────────────────────────────────────────

def _all_md_files() -> list[str]:
    result = []
    for root, _, files in os.walk(MD_DIR):
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(root, f), MD_DIR)
            parts = Path(rel).parts
            if parts[0] in SKIP_DOMAINS:
                continue
            if any(p.endswith("案例") for p in parts[1:-1]):
                continue
            result.append(rel)
    return sorted(result)


def main():
    force     = "--force" in sys.argv
    single    = None
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        single = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    all_files = [single] if single else _all_md_files()
    logger.info(f"MD 文件总数: {len(all_files)}  force={force}")

    if force and not single:
        logger.info("--force 模式：清空 chroma_db 中已有向量...")
        indexed = _indexed_sources()
        for src in indexed:
            records = db.get(where={"source": src})
            if records and records["ids"]:
                db.delete(ids=records["ids"])
        logger.info(f"  已清除 {len(indexed)} 个旧 source")

    indexed_sources = set() if force else _indexed_sources()
    logger.info(f"chroma_db 已有 {len(indexed_sources)} 个 source")

    ok = skip = fail = 0
    total_chunks = 0

    for rel in all_files:
        if rel in indexed_sources:
            logger.info(f"skip (已入库): {rel}")
            skip += 1
            continue
        logger.info(f"processing: {rel}")
        try:
            n = process_md_file(rel)
            total_chunks += n
            ok += 1
        except Exception as e:
            logger.error(f"  失败: {e}")
            fail += 1

    logger.info(
        f"done: {ok} 文件入库（{total_chunks} 块），{skip} 跳过，{fail} 失败"
    )


if __name__ == "__main__":
    main()
