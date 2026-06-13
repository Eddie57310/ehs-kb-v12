"""
md_reviewer.py  — MD 文件审核工具，端口 8084
三列布局：源文件（PDF全页自由滚动）| 编辑区（带框）| 实时预览（独立滚动）
"""
import os, re, json, base64, io, shutil, subprocess, hashlib
import gradio as gr
import fitz
import markdown as _md_lib
from PIL import Image
from pathlib import Path
from datetime import datetime

BASE_DIR = os.path.expanduser("~/doc_parser_v12")
MD_DIR   = os.path.join(BASE_DIR, "reviewed_md")
KB_DIR   = os.path.join(BASE_DIR, "Local_KB")
os.makedirs(MD_DIR, exist_ok=True)

_SAVE_HISTORY_FILE = os.path.join(MD_DIR, ".save_history.json")
_DPI = 120   # 全页渲染，低一点省内存


# ── 保存历史 ─────────────────────────────────────────────

def _load_history() -> dict:
    try:
        return json.loads(Path(_SAVE_HISTORY_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}

def _update_history(rel_path: str):
    h = _load_history()
    h[rel_path] = datetime.now().strftime("%m-%d %H:%M")
    Path(_SAVE_HISTORY_FILE).write_text(
        json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 文件列表（带已保存标记）────────────────────────────────

def _all_md_files() -> list[str]:
    result = []
    for root, _, files in os.walk(MD_DIR):
        for f in sorted(files):
            if f.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, f), MD_DIR)
                result.append(rel)
    return sorted(result)

def _build_choices() -> list[tuple[str, str]]:
    files = _all_md_files()
    history = _load_history()
    unsaved = [(f"  {f}", f) for f in files if f not in history]
    saved   = [(f"✓ {history[f]}  {f}", f) for f in files if f in history]
    return unsaved + saved or [("（无 MD 文件）", "")]

def _refresh():
    choices = _build_choices()
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


# ── 源文件查找（Local_KB）────────────────────────────────

_SRC_EXTS = [
    ".pdf", ".PDF", ".docx", ".DOCX", ".doc", ".DOC",
    ".xlsx", ".XLSX", ".xls", ".XLS", ".pptx", ".PPTX",
]

def _find_source(md_rel: str) -> str | None:
    base = os.path.splitext(md_rel)[0]
    for ext in _SRC_EXTS:
        c = os.path.join(KB_DIR, base + ext)
        if os.path.exists(c):
            return c
    parent = str(Path(md_rel).parent)
    for ext in [".xlsx", ".XLSX", ".xls", ".XLS"]:
        c = os.path.join(KB_DIR, parent + ext)
        if os.path.exists(c):
            return c
    return None


# ── PDF 全页渲染 → 可滚动 HTML ────────────────────────────

_BOX_H = "780px"

_SCROLL_STYLE = (
    f"height:{_BOX_H};overflow-y:auto;overflow-x:auto;"
    "border:1px solid #d0d0d0;border-radius:6px;background:#888;"
)

def _pdf_to_scroll_html(pdf_path: str) -> str:
    """将 PDF 所有页面渲染为 base64 图片，拼成可滚动 HTML。"""
    try:
        doc = fitz.open(pdf_path)
        imgs = []
        for i in range(len(doc)):
            page = doc[i]
            mat  = fitz.Matrix(_DPI / 72, _DPI / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf  = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            b64  = base64.b64encode(buf.getvalue()).decode()
            imgs.append(
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:100%;display:block;margin-bottom:4px;">'
            )
        doc.close()
        return f'<div style="{_SCROLL_STYLE}">{"".join(imgs)}</div>'
    except Exception as e:
        return f'<div style="{_SCROLL_STYLE};color:#fff;padding:16px;">渲染失败：{e}</div>'

def _no_preview(msg: str = "无预览") -> str:
    return (
        f'<div style="{_SCROLL_STYLE};display:flex;align-items:center;'
        f'justify-content:center;color:#ccc;font-size:14px;">{msg}</div>'
    )


# ── 加载 / 保存 MD ───────────────────────────────────────

def _load_md(rel_path: str) -> tuple[str, str]:
    if not rel_path:
        return "请先选择文件", ""
    full = os.path.join(MD_DIR, rel_path)
    if not os.path.exists(full):
        return f"文件不存在: {rel_path}", ""
    content = Path(full).read_text(encoding="utf-8")
    h = _load_history()
    saved_info = f"  上次保存：{h[rel_path]}" if rel_path in h else "  （未保存）"
    info = f"{rel_path}  |  {len(content):,} 字符  |  {content.count(chr(10))+1} 行{saved_info}"
    return info, content

def _save_md(rel_path: str, content: str) -> tuple[str, list, str]:
    if not rel_path:
        return "未加载文件", _build_choices(), rel_path
    try:
        Path(os.path.join(MD_DIR, rel_path)).write_text(content, encoding="utf-8")
        _update_history(rel_path)
        choices = _build_choices()
        # 下一个未处理文件 = choices 里第一个不带 ✓ 的，且不是当前文件
        next_val = next(
            (v for _, v in choices if not _.startswith("✓") and v != rel_path),
            choices[0][1] if choices else rel_path,
        )
        return f"已保存  {datetime.now().strftime('%H:%M:%S')}  ({len(content):,} 字符)", choices, next_val
    except Exception as e:
        return f"保存失败: {e}", _build_choices(), rel_path


# ── 弃用：MD + 源文件移入 _archived ──────────────────────

ARCHIVE_DIR = os.path.join(BASE_DIR, "archived_files")

def _archive_file(rel_path: str) -> tuple[str, list, str]:
    if not rel_path:
        return "未加载文件", _build_choices(), rel_path
    try:
        # MD 和源文件放在同一个子目录下
        dest_dir = Path(os.path.join(ARCHIVE_DIR, os.path.dirname(rel_path)))
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 移 MD 文件
        src_md = Path(os.path.join(MD_DIR, rel_path))
        src_md.rename(dest_dir / src_md.name)

        # 移源文件（Local_KB）——若源文件在子文件夹里，整个文件夹一起移
        src_file = _find_source(rel_path)
        if src_file:
            rel_src = os.path.relpath(src_file, KB_DIR)
            parts   = Path(rel_src).parts
            if len(parts) >= 4:
                # domain/category/package_folder/file → 移整个 package_folder
                pkg_folder = Path(os.path.join(KB_DIR, *parts[:3]))
                if pkg_folder.is_dir():
                    shutil.move(str(pkg_folder), str(dest_dir / pkg_folder.name))
                else:
                    Path(src_file).rename(dest_dir / os.path.basename(src_file))
            else:
                Path(src_file).rename(dest_dir / os.path.basename(src_file))

        # 从保存历史里清除
        h = _load_history()
        h.pop(rel_path, None)
        Path(_SAVE_HISTORY_FILE).write_text(
            json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        choices = _build_choices()
        next_val = choices[0][1] if choices else None
        return f"已弃用：{rel_path}", choices, next_val
    except Exception as e:
        return f"弃用失败: {e}", _build_choices(), rel_path


# ── LibreOffice 转 PDF 缓存 ──────────────────────────────

_LO_CACHE = os.path.join(BASE_DIR, "batch_output", "lo_cache")
os.makedirs(_LO_CACHE, exist_ok=True)

def _to_pdf_via_lo(src: str) -> str | None:
    """用 LibreOffice 把 Office 文件转成 PDF，缓存后返回路径。"""
    key      = hashlib.md5(src.encode()).hexdigest()[:12]
    out_pdf  = os.path.join(_LO_CACHE, f"{key}.pdf")
    if os.path.exists(out_pdf):
        return out_pdf
    try:
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", _LO_CACHE, src],
            capture_output=True, timeout=120,
        )
        # LibreOffice 输出文件名 = 原文件名改后缀
        lo_out = os.path.join(_LO_CACHE,
                              os.path.splitext(os.path.basename(src))[0] + ".pdf")
        if os.path.exists(lo_out):
            os.rename(lo_out, out_pdf)
            return out_pdf
    except Exception:
        pass
    return None


# ── 源文件加载 ───────────────────────────────────────────

def _load_source(md_rel: str) -> tuple[str, str]:
    src = _find_source(md_rel)
    if not src:
        return "（未找到源文件）", _no_preview("未找到源文件")
    ext = os.path.splitext(src)[1].lower()
    rel = os.path.relpath(src, KB_DIR)

    if ext == ".pdf":
        doc = fitz.open(src)
        n   = len(doc)
        doc.close()
        return f"PDF  {n} 页  |  {rel}", _pdf_to_scroll_html(src)

    elif ext in (".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"):
        pdf = _to_pdf_via_lo(src)
        if pdf:
            doc = fitz.open(pdf)
            n   = len(doc)
            doc.close()
            return f"{ext[1:].upper()}  {n} 页  |  {rel}", _pdf_to_scroll_html(pdf)
        return f"{ext[1:].upper()}  |  {rel}", _no_preview("转换中，请稍后重试")

    else:
        return f"{ext.upper()}  |  {rel}", _no_preview(f"{ext.upper()} 文件，无预览")


# ── 文件导航 ─────────────────────────────────────────────

def _step_file(rel_path, delta):
    files = _all_md_files()
    if not files or rel_path not in files:
        return gr.Dropdown(choices=_build_choices())
    idx = max(0, min(len(files) - 1, files.index(rel_path) + delta))
    choices = _build_choices()
    return gr.Dropdown(choices=choices, value=files[idx])


# ── MD → HTML（带独立滚动容器）──────────────────────────

_TABLE_CSS = """
<style>
.md-preview table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 13px;
}
.md-preview th, .md-preview td {
    border: 1px solid #bbb;
    padding: 5px 10px;
    text-align: left;
    vertical-align: top;
}
.md-preview th { background: #f0f0f0; font-weight: bold; }
.md-preview tr:nth-child(even) td { background: #fafafa; }
</style>
"""

_PREVIEW_STYLE = (
    f"height:{_BOX_H};overflow-y:auto;padding:12px 16px;"
    "border:1px solid #d0d0d0;border-radius:6px;"
    "font-size:14px;line-height:1.7;background:#fff;"
    "font-family:'PingFang SC','Microsoft YaHei',sans-serif;"
)

def _to_html(text: str) -> str:
    html = _md_lib.markdown(
        text or "",
        extensions=["tables", "nl2br", "fenced_code"],
    )
    return f'{_TABLE_CSS}<div class="md-preview" style="{_PREVIEW_STYLE}">{html}</div>'


# ── on_load ───────────────────────────────────────────────

def on_load(rel_path):
    status, md_content   = _load_md(rel_path)
    src_info, src_html   = _load_source(rel_path)
    return status, src_info, md_content, src_html, _to_html(md_content)


# ── CSS ──────────────────────────────────────────────────

_CSS = """
.gradio-container { max-width: 100% !important; padding: 8px 12px !important; }
#editor-col textarea {
    height: 780px !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 6px !important;
    font-family: 'PingFang SC', 'Microsoft YaHei', monospace;
    font-size: 13px;
    line-height: 1.6;
}
"""

# ── UI ───────────────────────────────────────────────────

with gr.Blocks(title="MD 文件审核") as demo:
    gr.Markdown("## MD 文件审核　`reviewed_md/` 浏览 · 编辑 · 保存")

    with gr.Row():
        file_dd     = gr.Dropdown(label="选择 MD 文件（✓=已保存）",
                                  choices=_build_choices(),
                                  value=(_build_choices()[0][1] if _build_choices() else None),
                                  scale=6)
        btn_prev    = gr.Button("◀ 上一个", scale=1)
        btn_next    = gr.Button("下一个 ▶", scale=1)
        btn_refresh = gr.Button("刷新列表", scale=1)
        btn_load    = gr.Button("加载", variant="primary", scale=1)

    status_box   = gr.Textbox(label="文件信息", interactive=False, lines=1)
    src_info_box = gr.Textbox(label="来源", interactive=False, lines=1)

    with gr.Row():
        # ① 源文件（全页可滚动）
        with gr.Column(scale=3):
            gr.Markdown("**源文件**")
            src_html = gr.HTML(value=_no_preview())

        # ② 编辑区
        with gr.Column(scale=3, elem_id="editor-col"):
            with gr.Row():
                btn_undo = gr.Button("↩ 撤回", size="sm", scale=1)
                btn_redo = gr.Button("↪ 重做", size="sm", scale=1)
                gr.HTML("<div style='flex:6'></div>")
            editor = gr.Textbox(
                label="编辑区（修改后点保存）",
                lines=42, max_lines=2000,
            )

        # ③ 预览（独立滚动）
        with gr.Column(scale=3):
            gr.Markdown("**实时预览**")
            preview = gr.HTML(value=_to_html(""))

    with gr.Row():
        btn_save    = gr.Button("保存", variant="primary", scale=1)
        btn_archive = gr.Button("弃用（移出知识库）", variant="stop", scale=1)
        save_result = gr.Textbox(label="", interactive=False,
                                 lines=1, scale=6, show_label=False)

    # ── 事件 ──
    _load_outs = [status_box, src_info_box, editor, src_html, preview]

    btn_load.click(on_load, inputs=[file_dd], outputs=_load_outs)
    file_dd.select(on_load, inputs=[file_dd], outputs=_load_outs)
    btn_refresh.click(_refresh, outputs=[file_dd])

    btn_prev.click(lambda r: _step_file(r, -1), inputs=[file_dd], outputs=[file_dd]).then(
        on_load, inputs=[file_dd], outputs=_load_outs
    )
    btn_next.click(lambda r: _step_file(r, +1), inputs=[file_dd], outputs=[file_dd]).then(
        on_load, inputs=[file_dd], outputs=_load_outs
    )

    editor.change(lambda x: _to_html(x), inputs=[editor], outputs=[preview])

    _UNDO_JS = "() => { const ta = document.querySelector('#editor-col textarea'); if(ta){ta.focus();document.execCommand('undo');} }"
    _REDO_JS = "() => { const ta = document.querySelector('#editor-col textarea'); if(ta){ta.focus();document.execCommand('redo');} }"
    btn_undo.click(fn=None, js=_UNDO_JS)
    btn_redo.click(fn=None, js=_REDO_JS)

    btn_save.click(
        _save_md, inputs=[file_dd, editor],
        outputs=[save_result, file_dd, file_dd],
    ).then(on_load, inputs=[file_dd], outputs=_load_outs)

    btn_archive.click(
        _archive_file, inputs=[file_dd],
        outputs=[save_result, file_dd, file_dd],
    ).then(on_load, inputs=[file_dd], outputs=_load_outs)



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8084, css=_CSS)
