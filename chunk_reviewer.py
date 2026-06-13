"""
chunk_reviewer.py — EHS 知识库数据质检工具
端口 8085 | 三栏布局：源文件（可框选截图）| 编辑 | 预览效果
用法：venv/bin/python chunk_reviewer.py
"""
import os, re, base64, time, json, pickle, threading
from io import BytesIO
import gradio as gr
import fitz
from PIL import Image
from collections import defaultdict
from langchain_core.documents import Document

BASE_DIR    = os.path.expanduser("~/doc_parser_v12")
KB_DIR      = os.path.join(BASE_DIR, "Local_KB")
MD_DIR      = os.path.join(BASE_DIR, "reviewed_md")
DB_DIR      = os.path.join(BASE_DIR, "chroma_db")
CROPS_DIR   = os.path.join(BASE_DIR, "user_crops")
CROPS_INDEX = os.path.join(CROPS_DIR, "index.json")
os.makedirs(CROPS_DIR, exist_ok=True)


# ── 截图历史持久化（按"源文件::p页码"为 key，绑定到具体页）──────
def _crop_key(meta: dict) -> str:
    """EHS案例块用 source::pN（按页），MD切片块用 source::seqN（按chunk_seq）。"""
    seq = meta.get("chunk_seq")
    if seq is not None:
        return f"{meta.get('source', 'unknown')}::seq{seq}"
    return f"{meta.get('source', 'unknown')}::p{meta.get('page') or 1}"


def _load_crops(key: str) -> list:
    if not os.path.exists(CROPS_INDEX):
        return []
    try:
        with open(CROPS_INDEX, encoding="utf-8") as f:
            idx = json.load(f)
        return [c for c in idx.get(key, [])
                if os.path.exists(os.path.join(BASE_DIR, c["path"]))]
    except Exception:
        return []


def _save_crops(key: str, crops: list):
    idx = {}
    if os.path.exists(CROPS_INDEX):
        try:
            with open(CROPS_INDEX, encoding="utf-8") as f:
                idx = json.load(f)
        except Exception:
            pass
    idx[key] = crops
    with open(CROPS_INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


# ── 懒加载数据库 ──────────────────────────────────────────────
_db = None

def _get_db():
    global _db
    if _db is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    return _db


# ── BM25 索引重建与序列化 ─────────────────────────────────────
BM25_PKL = os.path.join(BASE_DIR, "bm25_index.pkl")

def _rebuild_bm25_async():
    """后台线程：从 ChromaDB 读文档，重建 BM25 语料并序列化到 pkl。"""
    try:
        import jieba
        db = _get_db()
        result = db._collection.get(include=["documents", "metadatas"])
        docs  = result.get("documents", [])
        metas = result.get("metadatas", [])
        if not docs:
            return
        corpus = list(zip(docs, metas))
        # 序列化语料（不序列化 BM25Okapi 对象本身）
        with open(BM25_PKL + ".tmp", "wb") as f:
            pickle.dump(corpus, f)
        os.replace(BM25_PKL + ".tmp", BM25_PKL)
        print(f"[BM25] 索引已重建并写入 {BM25_PKL}，共 {len(corpus)} 条")
    except Exception as e:
        print(f"[BM25] 重建失败: {e}")


def _trigger_bm25_rebuild():
    """非阻塞触发 BM25 重建。"""
    threading.Thread(target=_rebuild_bm25_async, daemon=True).start()


# ── 回收站：删除的数据块存到库外 JSON，可随时恢复 ────────────────
# 设计：删除前把整块（id / 正文 / metadata / 原始向量）导出为一个 JSON 文件，
# 存放在 deleted_chunks/（完全在 ChromaDB 之外，对检索零影响、零噪音）。
# 恢复时优先用保存的原始向量写回，无需重新编码，逐字逐向量还原。
DELETED_DIR     = os.path.join(BASE_DIR, "deleted_chunks")
RESTORED_DIR    = os.path.join(DELETED_DIR, "restored")   # 已恢复的归档于此
os.makedirs(DELETED_DIR, exist_ok=True)


def _backup_deleted_chunk(chunk_id: str) -> str | None:
    """删除前把整块（含原始向量）导出为库外 JSON，返回文件路径；失败返回 None。"""
    try:
        db = _get_db()
        r  = db._collection.get(
            ids=[chunk_id], include=["documents", "metadatas", "embeddings"]
        )
        if not r["ids"]:
            return None
        emb  = r["embeddings"][0] if r.get("embeddings") is not None else None
        meta = r["metadatas"][0] or {}
        data = {
            "id":         chunk_id,
            "document":   r["documents"][0],
            "metadata":   meta,
            "embedding":  [float(x) for x in emb] if emb is not None else None,
            "deleted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source":     meta.get("source", ""),
            "page":       meta.get("page"),
        }
        stem  = os.path.splitext(os.path.basename(data["source"] or "unknown"))[0]
        ts    = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}__{stem}__p{data['page']}__{chunk_id[:8]}.json"
        path  = os.path.join(DELETED_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return path
    except Exception as e:
        print(f"[回收站] 备份失败: {e}")
        return None


def _deleted_label(data: dict) -> str:
    src  = os.path.basename(data.get("source", "") or "?")
    page = data.get("page")
    body = (data.get("document", "") or "").strip().replace("\n", " ")
    return f"[{data.get('deleted_at','')}] {src} p{page} · {body[:24]}"


def list_deleted_chunks():
    """扫描 deleted_chunks/（不含 restored/ 子目录），按删除时间倒序返回 (labels, {label:path})。"""
    items = []
    for fn in os.listdir(DELETED_DIR):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(DELETED_DIR, fn)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items.append((data.get("deleted_at", ""), _deleted_label(data), path))
        except Exception:
            continue
    items.sort(key=lambda x: x[0], reverse=True)
    labels = [it[1] for it in items]
    fmap   = {it[1]: it[2] for it in items}
    return labels, fmap


def preview_deleted_chunk(label, fmap):
    path = (fmap or {}).get(label)
    if not path or not os.path.exists(path):
        return "（选择上方条目以预览被删数据块的内容）"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"读取失败: {e}"
    has_vec = "有（恢复时逐向量还原）" if data.get("embedding") else "无（恢复时重新编码）"
    head = (f"删除时间：{data.get('deleted_at')}\n"
            f"来源：{data.get('source')}　第 {data.get('page')} 页\n"
            f"原始向量：{has_vec}　ID：{(data.get('id') or '')[:16]}…\n"
            + "─" * 42 + "\n")
    return head + (data.get("document", "") or "")


def restore_deleted_chunk(label, fmap):
    """把回收站条目写回 ChromaDB（优先用原始向量），成功后归档到 restored/。
    返回 (状态, dropdown_update, 新map, 预览清空)。"""
    path = (fmap or {}).get(label)
    if not path or not os.path.exists(path):
        labels, m = list_deleted_chunks()
        return "❌ 请先在上方选择要恢复的条目", gr.update(choices=labels, value=None), m, ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        db   = _get_db()
        emb  = data.get("embedding")
        meta = data.get("metadata", {})
        doc  = data.get("document", "")
        if emb:
            db._collection.upsert(ids=[data["id"]], embeddings=[emb],
                                  documents=[doc], metadatas=[meta])
            note = "（原始向量还原）"
        else:
            db.add_documents([Document(page_content=doc, metadata=meta)])
            note = "（重新编码）"
        os.makedirs(RESTORED_DIR, exist_ok=True)
        os.replace(path, os.path.join(RESTORED_DIR, os.path.basename(path)))
        _trigger_bm25_rebuild()
        labels, m = list_deleted_chunks()
        src = os.path.basename(data.get("source", "") or "?")
        return (f"✅ 已恢复 {note}：{src} 第 {data.get('page')} 页"
                f"（重新选择该文件即可看到）",
                gr.update(choices=labels, value=None), m, "")
    except Exception as e:
        labels, m = list_deleted_chunks()
        return f"❌ 恢复失败: {e}", gr.update(choices=labels, value=None), m, ""


def purge_deleted_chunk(label, fmap):
    """永久删除回收站中某条 JSON（不可恢复）。"""
    path = (fmap or {}).get(label)
    if not path or not os.path.exists(path):
        labels, m = list_deleted_chunks()
        return "❌ 请先选择条目", gr.update(choices=labels, value=None), m, ""
    try:
        os.remove(path)
    except Exception as e:
        labels, m = list_deleted_chunks()
        return f"❌ 删除失败: {e}", gr.update(choices=labels, value=None), m, ""
    labels, m = list_deleted_chunks()
    return "✅ 已永久删除该回收站条目", gr.update(choices=labels, value=None), m, ""


# ── 文件列表 ──────────────────────────────────────────────────
def _find_reviewed_md(source_rel: str) -> str | None:
    """根据 chunk 的 source 字段找对应的 reviewed_md 文件。
    兼容两种 source 格式：
      - index_reviewed_md.py 写入的 .md 路径
      - sync_kb_v11.py 写入的原始文件路径（.pdf/.docx/.xlsx）
    """
    # 直接命中
    candidate = os.path.join(MD_DIR, source_rel)
    if os.path.exists(candidate):
        return candidate
    # 换成 .md 扩展名
    base = os.path.splitext(source_rel)[0]
    candidate = os.path.join(MD_DIR, base + ".md")
    if os.path.exists(candidate):
        return candidate
    return None


def _load_md_for_source(source_rel: str) -> tuple[str, str]:
    """返回 (md_file_path_display, md_content)。"""
    path = _find_reviewed_md(source_rel)
    if not path:
        return "（未找到对应 MD 文件）", ""
    try:
        content = open(path, encoding="utf-8").read()
        rel = os.path.relpath(path, MD_DIR)
        return rel, content
    except Exception as e:
        return f"读取失败: {e}", ""


def _extract_chunk_heading(chunk_content: str) -> str:
    """从 chunk 内容提取 【...】 章节路径，供 MD 列显示定位提示。"""
    m = re.match(r'^【([^】]+)】', chunk_content.strip())
    return m.group(1) if m else ""


def load_file_list():
    db = _get_db()
    r  = db._collection.get(include=["metadatas"])
    counter = defaultdict(lambda: {"total": 0, "confirmed": 0})
    for m in r["metadatas"]:
        m = m or {}
        src = m.get("source", "")
        if src:
            counter[src]["total"] += 1
            if m.get("confirmed_at"):
                counter[src]["confirmed"] += 1

    def _sort_key(item):
        src, c = item
        t, cf = c["total"], c["confirmed"]
        if cf == 0:     return (0, src)   # 未确认 → 最前
        elif cf < t:    return (1, src)   # 部分确认 → 中间
        else:           return (2, src)   # 全部确认 → 沉底

    items = sorted(counter.items(), key=_sort_key)
    labels, src_map = [], {}
    for src, c in items:
        t, cf = c["total"], c["confirmed"]
        if cf == t and t > 0:
            label = f"[✓ {t}块] {src}"
        elif cf > 0:
            label = f"[{cf}/{t}块✓] {src}"
        else:
            label = f"[{t}块] {src}"
        labels.append(label)
        src_map[label] = src
    return labels, src_map


# ── Chunks ────────────────────────────────────────────────────
def get_chunks(source_file):
    db = _get_db()
    r  = db._collection.get(
        where={"source": source_file},
        include=["metadatas", "documents"],
    )
    chunks = []
    for i in range(len(r["ids"])):
        chunks.append({
            "id":      r["ids"][i],
            "content": r["documents"][i],
            "meta":    r["metadatas"][i],
        })
    # 排序：MD切片按 chunk_seq，EHS案例按 page，其余按 id
    chunks.sort(key=lambda c: (
        c["meta"].get("chunk_seq") or c["meta"].get("page") or 0,
        c["id"],
    ))
    # 若所有块的 page 均为 1（旧数据，MinerU 页标记注入前写入），
    # 则清零让 _get_source_pil 使用线性估算
    pages = [c["meta"].get("page") for c in chunks]
    if chunks and all(p == 1 for p in pages if p is not None):
        for c in chunks:
            if c["meta"].get("page") == 1:
                c["meta"] = {**c["meta"], "page": None}
    return chunks


# ── 格式检测 ──────────────────────────────────────────────────
def detect_format(content: str) -> str:
    if re.search(r'<table|<html|<tr|<td', content, re.IGNORECASE):
        return "HTML表格"
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '|' in line and i + 1 < len(lines):
            if re.match(r'\s*\|[\s\-|:]+\|\s*$', lines[i + 1]):
                return "Markdown表格"
    return "纯文字"


# ── 预览渲染 ──────────────────────────────────────────────────
def render_preview(content: str) -> str:
    fmt = detect_format(content)
    style = """<style>
      .pv { font-family:-apple-system,sans-serif;font-size:14px;line-height:1.7;
            padding:12px;background:#fff;border-radius:8px; }
      table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
      th{background:#f0f4f8;padding:6px 10px;border:1px solid #d0d7de;font-weight:600}
      td{padding:5px 10px;border:1px solid #d0d7de}
      tr:nth-child(even) td{background:#f8f9fa}
      .badge{display:inline-block;padding:2px 8px;border-radius:4px;
             font-size:11px;font-weight:600;margin-bottom:8px}
      .h{background:#fff3cd;color:#856404}
      .m{background:#d1e7dd;color:#0a5c36}
      .t{background:#e2e3e5;color:#383d41}
      .tip{font-size:12px;color:#888;margin-top:8px}
    </style>"""

    if fmt == "HTML表格":
        badge = '<span class="badge h">HTML表格</span>'
        body  = re.sub(r'<table', '<table style="border-collapse:collapse;width:100%"', content)
        body  = re.sub(r'<td', '<td style="padding:5px 10px;border:1px solid #d0d7de"', body)
        body  = re.sub(r'<th', '<th style="padding:6px 10px;border:1px solid #d0d7de;background:#f0f4f8"', body)
        tip   = '<p class="tip">⚠️ 飞书中 HTML 标签会原样输出为乱码，建议用截图替代</p>'
        return f'{style}<div class="pv">{badge}<br>{body}{tip}</div>'

    elif fmt == "Markdown表格":
        badge = '<span class="badge m">Markdown表格</span>'
        html  = _md_to_html(content)
        tip   = '<p class="tip">⚠️ 飞书 post 消息不渲染 Markdown 表格，用户会看到 | 符号</p>'
        return f'{style}<div class="pv">{badge}<br>{html}{tip}</div>'

    else:
        badge   = '<span class="badge t">纯文字</span>'
        escaped = content.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        body    = escaped.replace('\n', '<br>')
        return f'{style}<div class="pv">{badge}<br>{body}</div>'


def _md_to_html(text: str) -> str:
    lines, result, i = text.split('\n'), [], 0
    while i < len(lines):
        line = lines[i]
        if '|' in line and i+1 < len(lines) and re.match(r'\s*\|[\s\-|:]+\|\s*$', lines[i+1]):
            result.append('<table>')
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            result.append('<tr>'+''.join(f'<th>{c}</th>' for c in cells)+'</tr>')
            i += 2
            while i < len(lines) and '|' in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                result.append('<tr>'+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
                i += 1
            result.append('</table>')
        else:
            esc = lines[i].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            esc = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', esc)
            result.append(esc + '<br>')
            i += 1
    return '\n'.join(result)


# ── 源文件图片获取 ────────────────────────────────────────────
def _find_source_file(source: str) -> tuple[str, str]:
    """根据 chunk source 字段找 Local_KB 中的原始文件。
    处理 .md → 原始扩展名的映射。返回 (full_path, ext) 或 ("", "")。
    """
    ext = os.path.splitext(source)[1].lower()
    if ext == ".md":
        base = os.path.splitext(source)[0]
        for try_ext in (".pdf", ".pptx", ".docx", ".doc", ".xlsx"):
            p = os.path.join(KB_DIR, base + try_ext)
            if os.path.exists(p):
                return p, try_ext
        return "", ""
    p = os.path.join(KB_DIR, source)
    return (p, ext) if os.path.exists(p) else ("", "")


_page_cache: dict = {}   # (file_path, mtime) → [b64_png, ...]
_DOCX_PDF_DIR = os.path.join(BASE_DIR, "batch_output", "docx_pdf_cache")
os.makedirs(_DOCX_PDF_DIR, exist_ok=True)

def _docx_to_pdf(docx_path: str) -> str:
    """用 LibreOffice 将 DOCX 转为 PDF，返回 PDF 路径。失败返回空字符串。
    PDF 以 docx 文件名+mtime hash 命名，避免重复转换。
    """
    import subprocess, hashlib
    mtime = os.path.getmtime(docx_path)
    key   = hashlib.md5(f"{docx_path}{mtime}".encode()).hexdigest()[:12]
    name  = os.path.splitext(os.path.basename(docx_path))[0]
    pdf_path = os.path.join(_DOCX_PDF_DIR, f"{name}_{key}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", _DOCX_PDF_DIR, docx_path],
            capture_output=True, timeout=120
        )
        # LibreOffice 输出的文件名 = 原始名.pdf（不含 hash）
        raw_pdf = os.path.join(_DOCX_PDF_DIR, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
        if os.path.exists(raw_pdf):
            os.rename(raw_pdf, pdf_path)
            return pdf_path
    except Exception as e:
        print(f"[LibreOffice] 转换失败: {e}")
    return ""


def _render_pages_b64(file_path: str, dpi: float = 1.8) -> list[str]:
    """将文件所有页渲染为 PNG base64 列表（带 mtime 缓存，上限 60 页）。
    DOCX 文件先用 LibreOffice 转为 PDF 再渲染，确保页眉页脚和图片完整显示。
    """
    if not file_path:
        return []
    try:
        render_path = file_path
        if file_path.lower().endswith((".docx", ".doc")):
            pdf = _docx_to_pdf(file_path)
            if pdf:
                render_path = pdf
        mtime = os.path.getmtime(file_path)  # 始终用源文件 mtime 作缓存 key
        key   = (file_path, mtime)
        if key in _page_cache:
            return _page_cache[key]
        doc   = fitz.open(render_path)
        pages = []
        for i in range(min(len(doc), 60)):
            pix = doc[i].get_pixmap(matrix=fitz.Matrix(dpi, dpi))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = BytesIO()
            img.save(buf, format="PNG")
            pages.append(base64.b64encode(buf.getvalue()).decode())
        doc.close()
        _page_cache[key] = pages
        return pages
    except Exception as e:
        print(f"[render] 渲染失败 {file_path}: {e}")
        return []


def _pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 截图 + 缩放 JS（放入 gr.Blocks(js=...) 避开 DOMPurify 过滤）─────────────
# gr.HTML 会通过 DOMPurify 过滤掉所有 <script> 和 onclick，
# 所以 JS 必须走 gr.Blocks(js=...) 注入，用 MutationObserver 监听 DOM 变化。
_CROP_JS = """
(function() {
  'use strict';

  // ── 缩放 ────────────────────────────────────────────────────
  window._setZoom = function(pct) {
    pct = Math.max(30, Math.min(800, parseInt(pct) || 100));
    window._currentZoom = pct / 100;
    _applySize();  // 同步所有页面尺寸
    var lbl = document.getElementById('crop-zoom');
    if (lbl) lbl.textContent = pct + '%';
    var inp = document.getElementById('zoom-input');
    if (inp && inp.value != pct) inp.value = pct;
  };

  function _applySize() {
    var zoom = window._currentZoom || 1.0;
    document.querySelectorAll('.crop-page-img').forEach(function(img) {
      var W = img.naturalWidth  * zoom;
      var H = img.naturalHeight * zoom;
      if (!W || !H) return;
      img.style.width  = W + 'px';
      img.style.height = H + 'px';
      var cv = img.nextElementSibling;
      if (cv && cv.tagName === 'CANVAS') {
        cv.width = W; cv.height = H;
        cv.style.width = W + 'px'; cv.style.height = H + 'px';
      }
    });
  }

  // ── 多页截图框选 ─────────────────────────────────────────────
  var _initedCanvases = new WeakSet();

  function initPageCrops() {
    document.querySelectorAll('.crop-page-canvas').forEach(function(cv) {
      if (_initedCanvases.has(cv)) return;
      _initedCanvases.add(cv);

      var page = parseInt(cv.dataset.page || '1');
      var img  = cv.previousElementSibling;  // img 在 canvas 前面
      var ctx  = cv.getContext('2d');
      var drawing = false, sx=0, sy=0, ex=0, ey=0;

      function syncSize() {
        var zoom = window._currentZoom || 1.0;
        var W = img.naturalWidth * zoom, H = img.naturalHeight * zoom;
        if (!W || !H) return;
        img.style.width = W+'px'; img.style.height = H+'px';
        cv.width = W; cv.height = H;
        cv.style.width = W+'px'; cv.style.height = H+'px';
      }
      if (img && img.complete && img.naturalWidth > 0) syncSize();
      else if (img) img.addEventListener('load', syncSize);

      function pos(e) {
        var r = cv.getBoundingClientRect();
        return [(e.touches?e.touches[0].clientX:e.clientX)-r.left,
                (e.touches?e.touches[0].clientY:e.clientY)-r.top];
      }
      function redraw() {
        ctx.clearRect(0,0,cv.width,cv.height);
        var x=Math.min(sx,ex),y=Math.min(sy,ey),w=Math.abs(ex-sx),h=Math.abs(ey-sy);
        ctx.fillStyle='rgba(13,110,253,0.13)'; ctx.fillRect(x,y,w,h);
        ctx.strokeStyle='#0d6efd'; ctx.lineWidth=2; ctx.setLineDash([6,3]);
        ctx.strokeRect(x,y,w,h);
      }
      function onStart(e) {
        e.preventDefault();
        // 清除其他页面的选框
        document.querySelectorAll('.crop-page-canvas').forEach(function(c) {
          if (c !== cv) c.getContext('2d').clearRect(0,0,c.width,c.height);
        });
        ctx.clearRect(0,0,cv.width,cv.height);
        var p=pos(e); sx=ex=p[0]; sy=ey=p[1]; drawing=true;
      }
      function onMove(e) { if(!drawing)return; e.preventDefault(); var p=pos(e); ex=p[0]; ey=p[1]; redraw(); }
      function onEnd(e) {
        if(!drawing)return; drawing=false;
        var W=cv.width, H=cv.height;
        var x1=Math.min(sx,ex)/W, y1=Math.min(sy,ey)/H;
        var x2=Math.max(sx,ex)/W, y2=Math.max(sy,ey)/H;
        var tip=document.getElementById('crop-tip');
        if(x2-x1<0.02||y2-y1<0.02){if(tip)tip.textContent='选区太小，请重新框选';return;}
        // 格式: "页码:x1,y1,x2,y2"
        var v=page+':'+x1.toFixed(4)+','+y1.toFixed(4)+','+x2.toFixed(4)+','+y2.toFixed(4);
        var el=document.querySelector('#crop-coords-box textarea')||
               document.querySelector('#crop-coords-box input[type=text]')||
               document.querySelector('#crop-coords-box input');
        if(el){el.value=v; el.dispatchEvent(new Event('input',{bubbles:true}));}
        if(tip)tip.textContent='第'+page+'页 已选 ('+Math.round(x1*100)+'%,'+Math.round(y1*100)
          +'%)→('+Math.round(x2*100)+'%,'+Math.round(y2*100)+'%) 点「保存截图」';
      }
      cv.addEventListener('mousedown', onStart);
      cv.addEventListener('mousemove', onMove);
      cv.addEventListener('mouseup',   onEnd);
      cv.addEventListener('touchstart',onStart,{passive:false});
      cv.addEventListener('touchmove', onMove, {passive:false});
      cv.addEventListener('touchend',  onEnd);
    });
  }

  // ── 缩放按钮 ─────────────────────────────────────────────────
  function initZoomBtn() {
    var btn = document.getElementById('zoom-apply-btn');
    if (!btn || btn._ready) return;
    btn._ready = true;
    btn.addEventListener('click', function() {
      var inp = document.getElementById('zoom-input');
      if (inp) window._setZoom(inp.value);
    });
    var inp = document.getElementById('zoom-input');
    if (inp && !inp._ready) {
      inp._ready = true;
      inp.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') window._setZoom(this.value);
      });
    }
  }

  // ── 复制按钮（每条截图记录）──────────────────────────────────
  function initCopyBtns() {
    document.querySelectorAll('[id^="cp-"]').forEach(function(btn) {
      if (btn._cpReady) return;
      btn._cpReady = true;
      btn.addEventListener('click', function() {
        var tag = (this.dataset.tag || '').replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>');
        var self = this;
        function flash() {
          var orig = self.textContent;
          self.textContent = '已复制'; self.style.background='#4f7ef5'; self.style.color='#fff';
          setTimeout(function(){ self.textContent=orig; self.style.background='#fff'; self.style.color='#4f7ef5'; }, 1500);
        }
        if (navigator.clipboard) {
          navigator.clipboard.writeText(tag).then(flash);
        } else {
          var ta=document.createElement('textarea'); ta.value=tag;
          document.body.appendChild(ta); ta.select(); document.execCommand('copy');
          document.body.removeChild(ta); flash();
        }
      });
    });
  }

  // ── MutationObserver 监听 DOM 变化 ───────────────────────────
  function tick() { initPageCrops(); initZoomBtn(); initCopyBtns(); }

  var ob = new MutationObserver(tick);
  function boot() {
    ob.observe(document.body, {childList:true, subtree:true});
    tick();
  }
  if (document.readyState==='loading') document.addEventListener('DOMContentLoaded', boot);
  else setTimeout(boot, 200);
})();
"""


def make_all_pages_html(pages_b64: list, info_text: str) -> str:
    """生成多页滚动预览 HTML，每页独立 canvas 可框选截图。"""
    if not pages_b64:
        return f'<div style="padding:20px;color:#888">{info_text}</div>'
    zoom_bar = f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:13px;color:#444">
  <label style="margin:0">缩放:</label>
  <input id="zoom-input" type="number" value="100" min="30" max="800" step="10"
         style="width:68px;padding:2px 6px;border:1px solid #ccc;border-radius:4px;font-size:13px">
  <span>%</span>
  <button id="zoom-apply-btn"
          style="padding:2px 14px;border:1px solid #4f7ef5;border-radius:4px;
                 background:#4f7ef5;color:#fff;cursor:pointer;font-size:13px;line-height:1.6">
    应用
  </button>
  <span style="color:#aaa;font-size:12px">当前 <span id="crop-zoom">100%</span></span>
  <span style="margin-left:auto;color:#aaa;font-size:11px;max-width:40%;overflow:hidden;
               text-overflow:ellipsis;white-space:nowrap">{info_text}</span>
</div>"""
    pages_html = []
    for i, b64 in enumerate(pages_b64):
        p = i + 1
        pages_html.append(
            f'<div style="margin-bottom:8px">'
            f'<div style="font-size:11px;color:#666;padding:2px 6px;background:#e0e0e0;'
            f'border-radius:3px 3px 0 0;user-select:none">第 {p} 页</div>'
            f'<div style="position:relative;display:inline-block;user-select:none">'
            f'<img class="crop-page-img" data-page="{p}" src="data:image/png;base64,{b64}"'
            f' style="display:block;max-width:none" draggable="false">'
            f'<canvas class="crop-page-canvas" data-page="{p}"'
            f' style="position:absolute;top:0;left:0;pointer-events:all;cursor:crosshair"></canvas>'
            f'</div></div>'
        )
    return (
        zoom_bar
        + f'<div id="crop-wrap" style="overflow:auto;max-height:680px;border:1px solid #ddd;'
          f'border-radius:4px;background:#f0f0f0;padding:8px">'
        + ''.join(pages_html)
        + '</div>'
        + '<div id="crop-tip" style="font-size:12px;color:#666;margin-top:6px">'
          '在任意页面拖拽框选，然后点「保存截图」</div>'
    )


# ── 截图列表渲染 ──────────────────────────────────────────────
def _thumb_b64(path: str, max_w: int = 110) -> str:
    """生成缩略图 base64，失败返回空字符串。"""
    try:
        img = Image.open(path)
        r = min(1.0, max_w / img.width)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def render_crops_html(crops: list) -> str:
    if not crops:
        return '<div style="color:#aaa;font-size:12px;padding:8px 0">暂无截图，框选后点「保存截图」</div>'
    rows = []
    for i, c in enumerate(crops):
        tag         = c["tag"]
        tag_escaped = tag.replace('"', '&quot;').replace("<", "&lt;").replace(">", "&gt;")
        full        = os.path.join(BASE_DIR, c["path"])
        thumb       = _thumb_b64(full)
        thumb_html  = (f'<img src="data:image/jpeg;base64,{thumb}" '
                       f'style="max-width:110px;max-height:90px;border:1px solid #ddd;'
                       f'border-radius:3px;flex-shrink:0">'
                       if thumb else
                       '<div style="width:60px;height:40px;background:#eee;border-radius:3px;'
                       'display:flex;align-items:center;justify-content:center;'
                       'font-size:11px;color:#aaa;flex-shrink:0">无图</div>')
        rows.append(
            f'<div style="margin:6px 0;padding:8px;background:#f0f4ff;border-left:3px solid #4f7ef5;'
            f'border-radius:0 4px 4px 0;display:flex;align-items:flex-start;gap:8px">'
            f'{thumb_html}'
            f'<div style="flex:1;min-width:0">'
            f'<span style="color:#777;font-size:11px">#{i+1}&nbsp;</span>'
            f'<code style="font-size:11px;word-break:break-all;user-select:all">{tag_escaped}</code>'
            f'</div>'
            f'<button id="cp-{i}" data-tag="{tag_escaped}" '
            f'style="padding:2px 10px;font-size:12px;border:1px solid #4f7ef5;border-radius:4px;'
            f'background:#fff;color:#4f7ef5;cursor:pointer;white-space:nowrap;flex-shrink:0">'
            f'复制</button>'
            f'</div>'
        )
    hint = '<div style="font-size:11px;color:#888;margin-top:4px">输入序号后点「删除截图」可移除</div>'
    return (
        '<div style="max-height:320px;overflow-y:auto;margin-top:4px">'
        + ''.join(rows)
        + '</div>' + hint
    )


# ── 执行裁剪保存 ──────────────────────────────────────────────
def perform_crop(chunks: list, idx: int, coords_str: str, crops: list):
    """
    按归一化坐标裁剪源文件图片，追加到 crops 列表。
    coords_str 格式：'页码:x1,y1,x2,y2'（多页）或 'x1,y1,x2,y2'（兼容旧格式）
    返回 (状态消息, new_crops, crops_html)
    """
    if not chunks:
        return "❌ 未加载文件", crops, render_crops_html(crops)
    if not coords_str:
        return "❌ 请先在左侧图片上框选区域", crops, render_crops_html(crops)

    # 解析页码前缀
    page_override = None
    if ':' in coords_str:
        head, coords_str = coords_str.split(':', 1)
        try:
            page_override = int(head)
        except ValueError:
            pass

    if coords_str.count(',') != 3:
        return "❌ 请先在左侧图片上框选区域", crops, render_crops_html(crops)
    try:
        x1n, y1n, x2n, y2n = [float(v) for v in coords_str.split(',')]
    except Exception:
        return "❌ 坐标解析失败，请重新框选", crops, render_crops_html(crops)

    c    = chunks[idx]
    meta = c["meta"]
    img_rel = meta.get("image_path", "")
    source  = meta.get("source", "")
    page    = page_override or meta.get("page") or 1

    # 高清渲染（3x DPI）
    pil  = None
    stem = os.path.splitext(os.path.basename(source))[0]

    if img_rel:
        slide_dir = os.path.join(BASE_DIR, os.path.dirname(img_rel))
        if page_override and os.path.isdir(slide_dir):
            # 从预渲染目录找对应页
            fname = f"p{page_override:04d}.png"
            candidate = os.path.join(slide_dir, fname)
            if os.path.exists(candidate):
                pil = Image.open(candidate)
                stem = os.path.splitext(os.path.basename(source))[0]
        if pil is None and not page_override:
            full = os.path.join(BASE_DIR, img_rel)
            if os.path.exists(full):
                pil = Image.open(full)

    if pil is None:
        file_path, _ = _find_source_file(source)
        if file_path:
            try:
                doc  = fitz.open(file_path)
                p    = max(0, min(page - 1, len(doc) - 1))
                pix  = doc[p].get_pixmap(matrix=fitz.Matrix(3, 3))
                pil  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
                stem = os.path.splitext(os.path.basename(file_path))[0]
            except Exception:
                pass

    if pil is None:
        return "❌ 无法获取源文件图片", crops, render_crops_html(crops)

    w, h  = pil.size
    box   = (int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h))
    crop  = pil.crop(box)

    ts    = int(time.time() * 1000) % 1_000_000
    fname = f"{stem}_p{page}_{ts}.png"
    out   = os.path.join(CROPS_DIR, fname)
    crop.save(out, "PNG")

    rel       = os.path.relpath(out, BASE_DIR)
    tag       = f"[📷 {rel}]"
    key       = _crop_key(meta)
    new_crops = crops + [{"tag": tag, "fname": fname, "path": rel}]
    _save_crops(key, new_crops)
    return f"✅ 已保存 #{len(new_crops)} (第{page}页): user_crops/{fname}", new_crops, render_crops_html(new_crops)


def delete_crop(crops: list, del_num: int, crop_key: str):
    """删除第 del_num 个截图（1-based）并从磁盘移除文件。"""
    i = int(del_num or 0) - 1
    if i < 0 or i >= len(crops):
        return crops, render_crops_html(crops), f"❌ 序号 {del_num} 超出范围（共{len(crops)}个）"
    c    = crops[i]
    full = os.path.join(BASE_DIR, c["path"])
    try:
        os.remove(full)
    except Exception:
        pass
    new_crops = [x for j, x in enumerate(crops) if j != i]
    if crop_key:
        _save_crops(crop_key, new_crops)
    return new_crops, render_crops_html(new_crops), f"✅ 已删除 #{del_num}: {c['fname']}"


# ── 保存 / 删除 chunk ─────────────────────────────────────────
def save_chunk(chunk_id: str, new_content: str, chunks: list, idx: int):
    db   = _get_db()
    r    = db._collection.get(ids=[chunk_id], include=["metadatas"])
    if not r["metadatas"]:
        return "❌ 找不到该 chunk", chunks
    meta = {**r["metadatas"][0], "modified_at": time.strftime("%Y-%m-%d %H:%M")}
    db._collection.delete(ids=[chunk_id])
    new_ids = db.add_documents([Document(page_content=new_content, metadata=meta)])
    chunks[idx]["id"]      = new_ids[0]
    chunks[idx]["content"] = new_content
    chunks[idx]["meta"]    = meta
    _trigger_bm25_rebuild()
    return "✅ 已保存并重新嵌入", chunks


def confirm_chunk(chunk_id: str, new_content: str, chunks: list, idx: int):
    """保存内容 + 标记 confirmed_at，已确认块重排到底部，返回下一个未确认块索引。"""
    db = _get_db()
    r  = db._collection.get(ids=[chunk_id], include=["metadatas"])
    if not r["metadatas"]:
        return "❌ 找不到该 chunk", chunks, idx, False
    now  = time.strftime("%Y-%m-%d %H:%M")
    meta = {**r["metadatas"][0], "modified_at": now, "confirmed_at": now}
    db._collection.delete(ids=[chunk_id])
    new_ids = db.add_documents([Document(page_content=new_content, metadata=meta)])
    chunks[idx]["id"]      = new_ids[0]
    chunks[idx]["content"] = new_content
    chunks[idx]["meta"]    = meta

    # 重新排序：MD切片按 chunk_seq，EHS案例按 page，其余按 id
    chunks.sort(key=lambda c: (
        c["meta"].get("chunk_seq") or c["meta"].get("page") or 0,
        c["id"],
    ))

    all_confirmed = all(c["meta"].get("confirmed_at") for c in chunks)
    # 找第一个未确认块
    next_idx = next((i for i, c in enumerate(chunks) if not c["meta"].get("confirmed_at")),
                    len(chunks) - 1)
    _trigger_bm25_rebuild()
    return f"✅ 已确认（{now}）", chunks, next_idx, all_confirmed


def delete_chunk(chunk_id: str, chunks: list, idx: int):
    db = _get_db()
    _backup_deleted_chunk(chunk_id)   # 删除前先导出到库外回收站（含原始向量）
    db._collection.delete(ids=[chunk_id])
    new_chunks = [c for i, c in enumerate(chunks) if i != idx]
    new_idx    = min(idx, len(new_chunks) - 1) if new_chunks else 0
    return new_chunks, new_idx


# ── 统一 chunk 渲染 ───────────────────────────────────────────
def render_chunk_view(chunks: list, idx: int):
    """→ (crop_html, nav, content, meta_txt, preview_html, has_table)"""
    if not chunks:
        return '<div style="padding:20px;color:#888">请先选择文件</div>', "0/0", "", "暂无数据", "<p>—</p>", False

    c    = chunks[idx]
    meta = c["meta"]

    # EHS案例：有 image_path 时，从同目录加载所有预渲染 p*.png（保留视觉质量）
    img_rel = meta.get("image_path", "")
    if img_rel:
        slide_dir = os.path.join(BASE_DIR, os.path.dirname(img_rel))
        if os.path.isdir(slide_dir):
            slide_files = sorted(
                f for f in os.listdir(slide_dir)
                if re.match(r'p\d+\.png$', f)
            )
            pages_b64 = []
            for fname in slide_files:
                try:
                    pil = Image.open(os.path.join(slide_dir, fname))
                    pages_b64.append(_pil_to_b64(pil))
                except Exception:
                    pass
            info = f"{os.path.basename(slide_dir)} | 共 {len(pages_b64)} 页"
        else:
            # slide_dir 不存在，退而显示单张
            full = os.path.join(BASE_DIR, img_rel)
            pages_b64 = [_pil_to_b64(Image.open(full))] if os.path.exists(full) else []
            info = f"图片: {img_rel}"
    else:
        # 无 image_path：从 Local_KB 源文件全页渲染（国家规定/公司内部 MD 切片）
        file_path, _ = _find_source_file(meta.get("source", ""))
        pages_b64 = _render_pages_b64(file_path) if file_path else []
        if pages_b64:
            info = f"{os.path.basename(file_path)} | 共 {len(pages_b64)} 页"
        else:
            info = f"无预览（{meta.get('type','未知')}）"
    crop_html = make_all_pages_html(pages_b64, info)

    content   = c["content"]
    fmt       = detect_format(content)
    nav       = f"{idx+1} / {len(chunks)}"
    confirmed = meta.get("confirmed_at", "")
    modified  = meta.get("modified_at", "")
    ts_parts  = []
    if confirmed:
        ts_parts.append(f"✅ 已确认 {confirmed}")
    elif modified:
        ts_parts.append(f"✏️ 已修改 {modified}")
    ts_str    = " | " + " | ".join(ts_parts) if ts_parts else ""
    meta_txt  = (
        f"格式: {fmt} | 类型: {meta.get('type','—')} | "
        f"页: {meta.get('page','—')} | ID: {c['id'][:16]}…{ts_str}"
    )
    preview   = render_preview(content)
    has_table = fmt in ("HTML表格", "Markdown表格")
    return crop_html, nav, content, meta_txt, preview, has_table


# ── Gradio 界面 ───────────────────────────────────────────────
_UI_CSS = """
/* 预览列默认 430px（iPhone），可通过 JS 切换为 768px Fold7）*/
#col-preview { flex: 0 0 430px; max-width: 430px; min-width: 430px; }
/* 确保前两列拉伸填满剩余空间 */
#col-source { flex: 6 1 0 !important; min-width: 0 !important; }
#col-edit   { flex: 5 1 0 !important; min-width: 0 !important; }
/* 预览列独立滚动 */
#preview-html { max-height: calc(100vh - 120px); overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 6px; }
"""


def build_ui():
    labels, src_map = load_file_list()

    with gr.Blocks(title="KB Chunk Reviewer") as demo:
        gr.Markdown("# KB Chunk Reviewer　　源文件（框选截图）| MD原文 | 编辑 | 飞书预览")

        state_chunks   = gr.State([])
        state_idx      = gr.State(0)
        state_src_map  = gr.State(src_map)
        state_crops    = gr.State([])
        state_crop_key = gr.State("")   # 当前块对应的截图 key（source::p页码）
        state_md_text  = gr.State("")   # 当前文件对应的 reviewed_md 内容
        state_del_map  = gr.State({})   # 回收站：{label: json路径}

        # ── 顶部 ─────────────────────────────────────────────
        with gr.Row():
            file_dd     = gr.Dropdown(choices=labels, label="选择文件", scale=4, interactive=True)
            refresh_btn = gr.Button("刷新列表", scale=1)

        # ── 三栏主体 ──────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ① 源文件 + 截图工具
            with gr.Column(elem_id="col-source"):
                gr.Markdown("### 源文件　*拖拽框选截图区域*")
                src_html    = gr.HTML()
                crop_coords = gr.Textbox(
                    value="", elem_id="crop-coords-box",
                    label="框选坐标（自动填入）", interactive=False, max_lines=1,
                )
                with gr.Row():
                    crop_btn    = gr.Button("💾 保存截图", variant="primary")
                    crop_status = gr.Textbox(label="状态", interactive=False,
                                             max_lines=1, scale=3)
                gr.Markdown("**截图列表**（自动保存，切换文件后仍保留）")
                crops_html  = gr.HTML(render_crops_html([]))
                with gr.Row():
                    del_crop_num = gr.Number(label="删除第N个", minimum=1, step=1,
                                             precision=0, scale=1)
                    del_crop_btn = gr.Button("删除截图", variant="stop", scale=1)
                del_crop_status = gr.Textbox(label="", interactive=False,
                                             max_lines=1, show_label=False)

            # ② MD原文（只读参考）
            with gr.Column(elem_id="col-md"):
                gr.Markdown("### MD 原文")
                md_file_lbl  = gr.Textbox(label="来源文件", interactive=False, max_lines=1)
                md_chunk_loc = gr.Textbox(label="当前块章节路径", interactive=False, max_lines=2)
                md_text_box  = gr.Textbox(
                    label="MD 内容（只读）",
                    lines=28,
                    interactive=False,
                )

            # ③ 编辑
            with gr.Column(elem_id="col-edit"):
                gr.Markdown("### 编辑")
                chunk_nav = gr.Textbox(label="位置", interactive=False, max_lines=1)
                meta_info = gr.Textbox(label="元数据 / 格式", interactive=False, max_lines=2)
                with gr.Row():
                    undo_btn = gr.Button("↩ 撤回", size="sm", scale=1)
                    redo_btn = gr.Button("↪ 重做", size="sm", scale=1)
                    gr.HTML("<div style='flex:6'></div>")
                chunk_txt = gr.Textbox(label="内容", lines=22, interactive=True, elem_id="chunk-edit-box")
                with gr.Row():
                    prev_btn = gr.Button("◀ 上一块")
                    next_btn = gr.Button("下一块 ▶")
                with gr.Row():
                    save_btn    = gr.Button("保存并重新嵌入", variant="secondary")
                    confirm_btn = gr.Button("✅ 确认完成", variant="primary")
                    del_btn     = gr.Button("删除此块", variant="stop")
                status_txt = gr.Textbox(label="状态", interactive=False, max_lines=1)

            # ④ 飞书预览（可切换 430px iPhone / 768px Fold7展开）
            with gr.Column(elem_id="col-preview"):
                gr.Markdown("### 飞书预览")
                screen_radio = gr.Radio(
                    choices=["iPhone 17 PM (430px)", "Fold7 展开 (768px)"],
                    value="iPhone 17 PM (430px)",
                    label="模拟屏幕宽度",
                    interactive=True,
                )
                preview_out = gr.HTML(elem_id="preview-html")

        # ── 回收站（已删除数据块，存于库外 deleted_chunks/，可恢复）──
        with gr.Accordion("🗑 回收站　已删除的数据块（存于库外 deleted_chunks/，零噪音，可随时恢复）", open=False):
            with gr.Row():
                del_list_dd     = gr.Dropdown(choices=[], label="已删除的数据块（按删除时间倒序）",
                                              scale=4, interactive=True)
                del_refresh_btn = gr.Button("刷新回收站", scale=1)
            with gr.Row():
                restore_btn = gr.Button("♻ 恢复此块到数据库", variant="primary", scale=1)
                purge_btn   = gr.Button("✗ 永久删除（不可恢复）", variant="stop", scale=1)
            recycle_status = gr.Textbox(label="回收站状态", interactive=False, max_lines=2)
            del_preview    = gr.Textbox(label="内容预览", lines=12, interactive=False)

        # ── 事件：切换预览宽度（纯 JS）──────────────────────────
        screen_radio.change(
            fn=None, inputs=[screen_radio], outputs=[],
            js="""(choice) => {
              var w = choice.startsWith('Fold7') ? '768px' : '430px';
              var col = document.getElementById('col-preview');
              if (col) {
                col.style.flex = '0 0 ' + w;
                col.style.maxWidth = w;
                col.style.minWidth = w;
              }
            }"""
        )

        # ── 事件：选择文件 ────────────────────────────────────
        def on_select_file(label, src_map):
            src = src_map.get(label, "")
            if not src:
                empty = '<div style="padding:20px;color:#888">请先选择文件</div>'
                return [], 0, empty, "0/0", "", "暂无数据", "<p>—</p>", "", "", [], render_crops_html([]), "", "", "", ""
            chunks    = get_chunks(src)
            ch, nav, content, meta_txt, preview, _ = render_chunk_view(chunks, 0)
            key       = _crop_key(chunks[0]["meta"]) if chunks else ""
            crops     = _load_crops(key)
            md_lbl, md_txt = _load_md_for_source(src)
            heading   = _extract_chunk_heading(content) if chunks else ""
            return chunks, 0, ch, nav, content, meta_txt, preview, "", "", crops, render_crops_html(crops), key, md_lbl, md_txt, heading

        file_dd.change(
            on_select_file,
            inputs=[file_dd, state_src_map],
            outputs=[state_chunks, state_idx, src_html,
                     chunk_nav, chunk_txt, meta_info, preview_out,
                     crop_coords, status_txt, state_crops, crops_html, state_crop_key,
                     md_file_lbl, md_text_box, md_chunk_loc],
        )

        # ── 事件：导航 ────────────────────────────────────────
        def on_nav(chunks, idx, direction, md_txt):
            new_idx = max(0, min(len(chunks)-1, idx+direction)) if chunks else 0
            ch, nav, content, meta_txt, preview, _ = render_chunk_view(chunks, new_idx)
            key     = _crop_key(chunks[new_idx]["meta"]) if chunks else ""
            crops   = _load_crops(key)
            heading = _extract_chunk_heading(content)
            return new_idx, ch, nav, content, meta_txt, preview, "", "", crops, render_crops_html(crops), key, heading

        prev_btn.click(lambda c,i,m: on_nav(c,i,-1,m), inputs=[state_chunks, state_idx, state_md_text],
                       outputs=[state_idx, src_html, chunk_nav, chunk_txt,
                                meta_info, preview_out, crop_coords, status_txt,
                                state_crops, crops_html, state_crop_key, md_chunk_loc])
        next_btn.click(lambda c,i,m: on_nav(c,i,+1,m), inputs=[state_chunks, state_idx, state_md_text],
                       outputs=[state_idx, src_html, chunk_nav, chunk_txt,
                                meta_info, preview_out, crop_coords, status_txt,
                                state_crops, crops_html, state_crop_key, md_chunk_loc])

        # ── 事件：保存 chunk ──────────────────────────────────
        def on_save(chunks, idx, content):
            if not chunks:
                return "❌ 未加载文件", chunks, render_preview(content), "暂无数据"
            msg, new_chunks = save_chunk(chunks[idx]["id"], content, chunks, idx)
            _, _, _, meta_txt, _, _ = render_chunk_view(new_chunks, idx)
            return msg, new_chunks, render_preview(content), meta_txt

        save_btn.click(on_save, inputs=[state_chunks, state_idx, chunk_txt],
                       outputs=[status_txt, state_chunks, preview_out, meta_info])

        # ── 事件：确认完成 ────────────────────────────────────
        def on_confirm(chunks, idx, content, src_map):
            if not chunks:
                return ("❌ 未加载文件", chunks, idx,
                        '<div style="padding:20px;color:#888">请先选择文件</div>',
                        "0/0", content, "暂无数据", "<p>—</p>",
                        "", [], render_crops_html([]), "",
                        gr.update(), src_map, "", gr.update(), gr.update())

            msg, new_chunks, next_idx, all_confirmed = confirm_chunk(
                chunks[idx]["id"], content, chunks, idx
            )
            new_labels, new_src_map = load_file_list()
            file_update = gr.update(choices=new_labels)
            md_lbl_upd = gr.update()
            md_txt_upd = gr.update()

            if all_confirmed:
                next_file = next((l for l in new_labels if not l.startswith("[✓")), None)
                if next_file:
                    new_src = new_src_map.get(next_file, "")
                    new_chunks = get_chunks(new_src) if new_src else []
                    next_idx = 0
                    file_update = gr.update(choices=new_labels, value=next_file)
                    md_lbl_str, md_txt_str = _load_md_for_source(new_src) if new_src else ("", "")
                    md_lbl_upd = gr.update(value=md_lbl_str)
                    md_txt_upd = gr.update(value=md_txt_str)
                    msg = "✅ 文件已全部确认，已跳转至下一个待确认文件"
                else:
                    msg = "🎉 全部文件已确认完毕！"

            ch, nav, cont, meta_txt, preview, _ = (
                render_chunk_view(new_chunks, next_idx) if new_chunks
                else ('<div style="padding:20px;color:#888">请先选择文件</div>',
                      "0/0", "", "暂无数据", "<p>—</p>", False)
            )
            key    = _crop_key(new_chunks[next_idx]["meta"]) if new_chunks else ""
            crops  = _load_crops(key)
            heading = _extract_chunk_heading(cont)

            return (msg, new_chunks, next_idx, ch, nav, cont, meta_txt, preview,
                    "", crops, render_crops_html(crops), key,
                    file_update, new_src_map, heading, md_lbl_upd, md_txt_upd)

        confirm_btn.click(
            on_confirm,
            inputs=[state_chunks, state_idx, chunk_txt, state_src_map],
            outputs=[status_txt, state_chunks, state_idx, src_html,
                     chunk_nav, chunk_txt, meta_info, preview_out,
                     crop_coords, state_crops, crops_html, state_crop_key,
                     file_dd, state_src_map, md_chunk_loc,
                     md_file_lbl, md_text_box],
        )

        # ── 事件：删除 chunk ──────────────────────────────────
        def on_delete(chunks, idx):
            if not chunks:
                labels, dmap = list_deleted_chunks()
                return ("❌ 未加载文件", chunks, idx, '<div></div>', "0/0", "", "暂无数据",
                        "<p>—</p>", gr.update(choices=labels), dmap)
            new_chunks, new_idx = delete_chunk(chunks[idx]["id"], chunks, idx)
            ch, nav, content, meta_txt, preview, _ = render_chunk_view(new_chunks, new_idx)
            labels, dmap = list_deleted_chunks()
            return ("✅ 已删除（已存入库外回收站，可在下方「🗑 回收站」恢复）",
                    new_chunks, new_idx, ch, nav, content, meta_txt, preview,
                    gr.update(choices=labels), dmap)

        del_btn.click(on_delete, inputs=[state_chunks, state_idx],
                      outputs=[status_txt, state_chunks, state_idx,
                               src_html, chunk_nav, chunk_txt, meta_info, preview_out,
                               del_list_dd, state_del_map])

        # ── 事件：撤回 / 重做 ─────────────────────────────────
        _UNDO_JS = "() => { const ta = document.querySelector('#chunk-edit-box textarea'); if(ta){ta.focus();document.execCommand('undo');} }"
        _REDO_JS = "() => { const ta = document.querySelector('#chunk-edit-box textarea'); if(ta){ta.focus();document.execCommand('redo');} }"
        undo_btn.click(fn=None, js=_UNDO_JS)
        redo_btn.click(fn=None, js=_REDO_JS)

        # ── 事件：实时预览 ────────────────────────────────────
        chunk_txt.change(lambda t: render_preview(t), inputs=[chunk_txt], outputs=[preview_out])

        # ── 事件：保存截图 ────────────────────────────────────
        crop_btn.click(
            perform_crop,
            inputs=[state_chunks, state_idx, crop_coords, state_crops],
            outputs=[crop_status, state_crops, crops_html],
        )

        # ── 事件：删除截图 ────────────────────────────────────
        del_crop_btn.click(
            delete_crop,
            inputs=[state_crops, del_crop_num, state_crop_key],
            outputs=[state_crops, crops_html, del_crop_status],
        )

        # ── 事件：刷新列表 ────────────────────────────────────
        def on_refresh():
            new_labels, new_map = load_file_list()
            return gr.update(choices=new_labels, value=None), new_map

        refresh_btn.click(on_refresh, outputs=[file_dd, state_src_map])

        # ── 事件：回收站 ──────────────────────────────────────
        def on_del_refresh():
            labels, m = list_deleted_chunks()
            return gr.update(choices=labels, value=None), m, ""

        del_list_dd.change(preview_deleted_chunk,
                           inputs=[del_list_dd, state_del_map], outputs=[del_preview])
        del_refresh_btn.click(on_del_refresh,
                              outputs=[del_list_dd, state_del_map, del_preview])
        restore_btn.click(restore_deleted_chunk,
                          inputs=[del_list_dd, state_del_map],
                          outputs=[recycle_status, del_list_dd, state_del_map, del_preview])
        purge_btn.click(purge_deleted_chunk,
                        inputs=[del_list_dd, state_del_map],
                        outputs=[recycle_status, del_list_dd, state_del_map, del_preview])

        # 页面加载时填充回收站列表
        demo.load(on_del_refresh, outputs=[del_list_dd, state_del_map, del_preview])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=8085, css=_UI_CSS, js=_CROP_JS)
