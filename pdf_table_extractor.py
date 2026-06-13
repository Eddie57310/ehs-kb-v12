"""
pdf_table_extractor.py  v1.4
三引擎 PDF 提取模块，集成到 sync_kb_v5.py

引擎优先级：
  1. pdfplumber  —— 结构化表格提取（处理 MinerU 乱码场景）
  2. pymupdf     —— 快速文本提取（MinerU 超时/崩溃兜底）

乱码检测五规则：
  规则1: 竖线列串行符          —— 宽表列合并残留
  规则2: HTML空单元格比例过高  —— 跨列合并表格展开失败
  规则3: 替换字符堆积(U+FFFD)  —— 扫描件OCR识别失败
  规则4: 图片行占比高且文字少  —— 纯图片扫描件
  规则5: 重复页眉行占比过高    —— PDF页眉未过滤
"""

import re
from langchain_core.documents import Document

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

_EXACT_HEADERS = {
    '序号', '履职内容', '法律依据', '履职要点', '责任部门', '管理输出',
    '管理要素', '标准要求', '编号', '内容', '要求', '说明',
    '措施', '责任人', '完成时间', '备注', '名称'
}


# ═══════════════════════════════════════════════════════════
#  1. 乱码检测（五规则）
# ═══════════════════════════════════════════════════════════

def is_table_garbled(md_content: str) -> bool:
    """
    五条规则检测 MinerU 输出是否存在乱码，任一触发即返回 True。
    """
    # ── 规则1：竖线列串行符 —— 宽表列合并残留 ──
    if '丨' in md_content:
        return True

    # ── 规则2：HTML空单元格比例过高 —— 跨列合并表格展开失败 ──
    # 仅在文字内容也很稀少时才判乱码（技术规范有大量合并单元格但文字正常）
    total_td = len(re.findall(r'<td', md_content))
    empty_td = len(re.findall(r'<td>\s*</td>', md_content))
    text_chars_r2 = len(re.sub(r'<[^>]+>', '', md_content).strip())
    if total_td > 10 and empty_td / total_td > 0.4 and text_chars_r2 < 3000:
        return True

    # ── 规则3：替换字符堆积(U+FFFD) —— 扫描件OCR识别失败 ──
    if md_content.count('\ufffd') > 10:
        return True

    lines = md_content.strip().split('\n')
    non_empty = [l for l in lines if l.strip()]
    if non_empty:
        # ── 规则4：图片行占比过高且文字极少 —— 纯图片扫描件 ──
        img_lines = [l for l in non_empty if re.match(r'^\s*!\[.*?\]\(.*?\)\s*$', l)]
        text_chars = len(re.sub(r'!\[.*?\]\(.*?\)', '', md_content).strip())
        if len(img_lines) / len(non_empty) > 0.6 and text_chars < 200:
            return True

        # ── 规则5：重复页眉行占比过高 —— PDF页眉未过滤 ──
        counts = {}
        for l in non_empty:
            s = l.strip()
            if len(s) > 10:
                counts[s] = counts.get(s, 0) + 1
        repeat_total = sum(v for v in counts.values() if v >= 5)
        if repeat_total / len(non_empty) > 0.3:
            return True

    return False


# ═══════════════════════════════════════════════════════════
#  2. pdfplumber 路径（表格结构化提取）
# ═══════════════════════════════════════════════════════════

def _clean(text) -> str:
    if not text:
        return ''
    s = str(text).strip()
    s = re.sub(r'\n{3,}', '\n\n', s)
    return '' if s in ('\\', '\\\\', '/') else s

def _is_section_row(row) -> bool:
    non_null = [str(c).strip() for c in row if c and str(c).strip()]
    if not non_null:
        return True
    if len(non_null) == 1:
        t = non_null[0]
        if re.match(r'^[一二三四五六七八九十]+[、．.]', t): return True
        if re.match(r'^\d+$', t): return True
    return False

def _is_header_row(row) -> bool:
    non_null = [str(c).strip() for c in row if c and str(c).strip()]
    if not non_null:
        return False
    exact = sum(1 for c in non_null if c in _EXACT_HEADERS)
    return exact / len(non_null) > 0.5

def _row_to_text(cells, names) -> str:
    parts = []
    for name, cell in zip(names, cells):
        text = _clean(cell)
        if not text or name == '序号' or re.match(r'^\d+$', text):
            continue
        parts.append(f'【{name}】{text}')
    return '\n'.join(parts)

def _extract_tables_from_pdf(pdf_path: str) -> list:
    text_chunks = []
    compressed_names = []
    valid_idx = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                plain = page.extract_text()
                if plain and plain.strip():
                    text_chunks.append(plain.strip())
                continue

            for table in tables:
                if not table:
                    continue
                table = [r for r in table if any(c and str(c).strip() for c in r)]
                if not table:
                    continue

                ncols = len(table[0])

                if _is_header_row(table[0]):
                    raw = [str(c).strip() if c and str(c).strip() else None for c in table[0]]
                    compressed_names = [n for n in raw if n]
                    valid_idx = [i for i, n in enumerate(raw) if n]
                    data_rows = table[1:]
                else:
                    data_rows = table
                    if compressed_names and ncols == len(compressed_names):
                        valid_idx = list(range(ncols))
                    elif compressed_names and len(compressed_names) <= ncols:
                        valid_idx = list(range(len(compressed_names)))
                    else:
                        valid_idx = list(range(ncols))
                        compressed_names = [f'列{i+1}' for i in range(ncols)]

                if len(compressed_names) == 2:
                    lines = []
                    for row in data_rows:
                        if _is_section_row(row):
                            continue
                        cells = [_clean(row[i]) if i < len(row) else '' for i in valid_idx]
                        cells = [c for c in cells if c]
                        if cells:
                            lines.append(' → '.join(cells))
                    if lines:
                        text_chunks.append('\n'.join(lines))
                    continue

                for row in data_rows:
                    if _is_section_row(row):
                        continue
                    cells = [_clean(row[i]) if i < len(row) else '' for i in valid_idx]
                    row_text = _row_to_text(cells, compressed_names)
                    if row_text.strip():
                        text_chunks.append(row_text)

    return text_chunks

def pdf_table_to_documents(pdf_path: str, metadata: dict) -> list:
    if not HAS_PDFPLUMBER:
        return []
    try:
        chunks = _extract_tables_from_pdf(pdf_path)
        return [
            Document(page_content=c, metadata={**metadata, 'type': 'pdf_table'})
            for c in chunks if len(c.strip()) >= 20
        ]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
#  3. pymupdf 路径（快速文本兜底）
# ═══════════════════════════════════════════════════════════

def pymupdf_to_documents(pdf_path: str, metadata: dict, splitter) -> list:
    if not HAS_PYMUPDF:
        return []
    try:
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text("text").strip() for page in doc if page.get_text("text").strip()]
        doc.close()
        full_text = '\n\n'.join(pages_text)
        if not full_text.strip():
            return []
        chunks = splitter.split_text(full_text)
        return [
            Document(page_content=c, metadata={**metadata, 'type': 'pdf_pymupdf'})
            for c in chunks if len(c.strip()) >= 20
        ]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
#  4. 引擎状态检查
# ═══════════════════════════════════════════════════════════

def engine_status() -> str:
    return '\n'.join([
        f"  pdfplumber : {'✅ 已加载' if HAS_PDFPLUMBER else '❌ 未安装 → pip install pdfplumber'}",
        f"  pymupdf    : {'✅ 已加载' if HAS_PYMUPDF else '❌ 未安装 → pip install pymupdf'}",
    ])


# ═══════════════════════════════════════════════════════════
#  5. 结构化表格提取（供 SQLite 写入）
# ═══════════════════════════════════════════════════════════

_CAT_KEYWORDS  = ['类别', '分类', '类型', '大类', '类']
_TASK_KEYWORDS = ['任务', '工作', '内容', '具体', '措施', '事项', '要求', '工作任务', '履职', '职责']
_FREQ_KEYWORDS = ['频次', '频率', '周期', '次数', '频度', '时间要求', '时限', '完成时间', '频']


def _map_col_to_field(col_name: str) -> str:
    """将列名映射到语义字段: category / task / freq / extra"""
    for kw in _CAT_KEYWORDS:
        if kw in col_name:
            return 'category'
    for kw in _TASK_KEYWORDS:
        if kw in col_name:
            return 'task'
    for kw in _FREQ_KEYWORDS:
        if kw in col_name:
            return 'freq'
    return 'extra'


def extract_structured_tables(pdf_path: str) -> list:
    """
    提取结构化表格数据供 SQLite 写入（不改变原有向量化逻辑）。

    返回列表，每项为:
    {
        "table_name": str,           # 第N页-表M 标记
        "headers": [str],            # 原始列名
        "rows": [                    # 数据行列表
            {
                "row_index": int,
                "category": str,
                "task": str,
                "freq": str,
                "extra": str,
            }
        ]
    }
    """
    if not HAS_PDFPLUMBER:
        return []
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue
                for tbl_idx, table in enumerate(tables):
                    if not table:
                        continue
                    # 过滤全空行
                    table = [r for r in table if any(c and str(c).strip() for c in r)]
                    if len(table) < 2:
                        continue

                    # 识别表头
                    if _is_header_row(table[0]):
                        raw_headers = [str(c).strip() if c else '' for c in table[0]]
                        headers   = [h for h in raw_headers if h]
                        valid_idx = [i for i, h in enumerate(raw_headers) if h]
                        data_rows = table[1:]
                    else:
                        ncols     = len(table[0])
                        headers   = [f'列{i+1}' for i in range(ncols)]
                        valid_idx = list(range(ncols))
                        data_rows = table

                    if not headers:
                        continue

                    field_map = {h: _map_col_to_field(h) for h in headers}
                    structured_rows = []

                    for row_idx, row in enumerate(data_rows):
                        if _is_section_row(row):
                            continue
                        cells = [_clean(row[i]) if i < len(row) else '' for i in valid_idx]
                        row_dict = dict(zip(headers, cells))

                        cat   = ' / '.join(v for k, v in row_dict.items() if field_map[k] == 'category' and v)
                        task  = ' / '.join(v for k, v in row_dict.items() if field_map[k] == 'task'     and v)
                        freq  = ' / '.join(v for k, v in row_dict.items() if field_map[k] == 'freq'     and v)
                        extra = ' / '.join(v for k, v in row_dict.items() if field_map[k] == 'extra'    and v)

                        if task or cat:  # 至少有任务或类别才写入
                            structured_rows.append({
                                "row_index": row_idx,
                                "category":  cat,
                                "task":      task,
                                "freq":      freq,
                                "extra":     extra,
                            })

                    if structured_rows:
                        results.append({
                            "table_name": f"第{page_idx}页-表{tbl_idx + 1}",
                            "headers":    headers,
                            "rows":       structured_rows,
                        })
    except Exception:
        pass
    return results
