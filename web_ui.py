import os
import re
import numpy as np
import gradio as gr
import requests
import yaml
from PIL import Image

# 你的高级网关地址
API_URL = "http://127.0.0.1:8080/api/ask"

BASE_DIR      = os.path.expanduser("~/doc_parser_v12")
CONFIRM_DIR   = os.path.join(BASE_DIR, "org_confirm")
REVIEW_DIR    = os.path.join(BASE_DIR, "review_pending")
SLIDE_IMG_DIR = os.path.join(BASE_DIR, "slide_images")
os.makedirs(CONFIRM_DIR,   exist_ok=True)
os.makedirs(REVIEW_DIR,    exist_ok=True)
os.makedirs(SLIDE_IMG_DIR, exist_ok=True)

# ChromaDB 懒加载（仅在写入数据库时初始化）
_db = None
def _get_db():
    global _db
    if _db is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cpu'})
        _db = Chroma(persist_directory=os.path.join(BASE_DIR, "chroma_db"), embedding_function=emb)
    return _db


# ─── 组织架构审核辅助函数 ─────────────────────────────────

def _list_yaml_files():
    files = sorted(f for f in os.listdir(CONFIRM_DIR) if f.endswith("_org_confirm.yaml"))
    return files or ["（无待确认文件）"]


def _yaml_path(filename):
    return os.path.join(CONFIRM_DIR, filename)


def refresh_yaml_list():
    files = _list_yaml_files()
    return gr.Dropdown(choices=files, value=files[0])


def load_yaml_file(filename):
    """返回 (info, node_rows, rel_rows, node_names)"""
    empty = ("请先选择文件", [], [], [])
    if not filename or filename == "（无待确认文件）":
        return empty
    path = _yaml_path(filename)
    if not os.path.exists(path):
        return ("文件不存在", [], [], [])
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    status = data.get("status", "pending")
    source = data.get("source", "")
    method = data.get("method", "")
    pages  = data.get("pages", "")
    info   = f"来源: {source}\n方法: {method}   页码: {pages}   状态: {status}"
    nodes  = data.get("nodes", [])
    node_rows = [
        [n.get("name", ""), n.get("parent") or "", str(n.get("level", 1)),
         f'{n.get("confidence", "")}', n.get("note", "")]
        for n in nodes
    ]
    node_names = [n.get("name", "") for n in nodes if n.get("name", "").strip()]
    rels = data.get("relations", [])
    rel_rows = [
        [r.get("from", ""), r.get("to", ""), r.get("type", "governs"), r.get("method", "")]
        for r in rels
    ]
    return info, node_rows, rel_rows, node_names


def _normalize_rows(data):
    """将 pandas DataFrame 或 list 统一转为 list[list]"""
    if data is None:
        return []
    if hasattr(data, 'values'):
        return [list(r) for r in data.values.tolist()]
    return [list(r) for r in data]


def save_and_apply(filename, node_data, rel_data, action):
    """
    action: "save" 仅保存  |  "apply" 保存并写入SQLite
    """
    if not filename or filename == "（无待确认文件）":
        return "❌ 请先选择文件"
    path = _yaml_path(filename)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    node_data = _normalize_rows(node_data)
    rel_data  = _normalize_rows(rel_data)

    # 重建 nodes
    new_nodes = []
    for row in node_data:
        name, parent, level, conf, note = (list(row) + ["", "", "1", "", ""])[:5]
        name = str(name).strip()
        if not name:
            continue
        new_nodes.append({
            "name":       name,
            "parent":     str(parent).strip() or None,
            "level":      int(str(level).strip() or 1),
            "confidence": conf,
            "note":       str(note).strip(),
        })
    data["nodes"] = new_nodes

    # 重建 relations
    new_rels = []
    for row in rel_data:
        frm, to, typ, method = (list(row) + ["", "", "governs", ""])[:4]
        frm = str(frm).strip()
        to  = str(to).strip()
        if frm and to:
            new_rels.append({"from": frm, "to": to, "type": str(typ).strip(), "method": str(method).strip()})
    data["relations"] = new_rels

    if action == "apply":
        data["status"] = "confirmed"

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)

    return f"✅ 已保存 {len(new_nodes)} 个节点 + {len(new_rels)} 条关联（status 仍为 {data.get('status')}）"

def ask_gateway(question, history, db_choice, model_choice):
    payload = {
        "question": question,
        "db": db_choice.lower(),  # 动态传入选中的数据库 (v3-v10)
        "model": model_choice
    }
    
    try:
        res = requests.post(API_URL, json=payload, timeout=300).json()
        if res.get("status") == "success":
            return res.get("answer")
        else:
            return f"❌ 网关报错: {res.get('message')}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 8080 网关，请确认 api_server_v11.py 是否已启动！"
    except Exception as e:
        return f"❌ 请求异常: {str(e)}"

# 构建高科技感网页界面
with gr.Blocks(title="智慧工地核心终端") as demo:
    gr.Markdown("## 🏗️ 智慧工地断网知识大脑 (全库直连版)")

    with gr.Tabs():

        # ══════════════════════════════════════════
        #  Tab 1: 知识库问答（原有功能）
        # ══════════════════════════════════════════
        with gr.Tab("💬 知识库问答"):
            with gr.Row():
                db_dropdown = gr.Dropdown(
                    choices=["V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"],
                    value="V10",
                    label="📁 目标数据库路由",
                    interactive=True,
                )
                model_dropdown = gr.Dropdown(
                    choices=["qwen2.5:32b", "deepseek-r1:32b", "qwen2.5:14b"],
                    value="qwen2.5:32b",
                    label="🧠 本地推理引擎",
                    interactive=True,
                )

            gr.ChatInterface(
                fn=ask_gateway,
                additional_inputs=[db_dropdown, model_dropdown],
                chatbot=gr.Chatbot(height=560),
                textbox=gr.Textbox(
                    placeholder="请输入工程相关问题，引擎将自动检索上方选中的数据库...",
                    container=False, scale=7,
                ),
            )

        # ══════════════════════════════════════════
        #  Tab 2: 组织架构审核
        # ══════════════════════════════════════════
        with gr.Tab("🌳 组织架构审核"):
            gr.Markdown(
                "### 审核视觉LLM自动识别的组织架构层级关系\n"
                "检查每行的 **parent**（父节点名）和 **level**（层级数）是否正确，"
                "然后点击「确认并写入SQLite」。"
            )

            with gr.Row():
                yaml_dd = gr.Dropdown(
                    label="待确认文件",
                    choices=_list_yaml_files(),
                    value=_list_yaml_files()[0],
                    interactive=True,
                    scale=4,
                )
                load_btn    = gr.Button("📂 加载", size="sm", scale=1)
                refresh_btn = gr.Button("🔄 刷新列表", size="sm", scale=1)

            file_info = gr.Textbox(
                label="文件信息",
                interactive=False,
                lines=2,
            )

            node_table = gr.Dataframe(
                headers=["节点名称(name)", "父节点(parent)", "层级(level)", "置信度", "备注"],
                datatype=["str", "str", "number", "str", "str"],
                column_count=(5, "fixed"),
                interactive=True,
                wrap=True,
                label="层级节点（可直接编辑 parent / level，空name行自动忽略）",
            )

            gr.Markdown("#### 自定义关联（非父子关系）")
            rel_state = gr.State([])   # 真正的数据源，避免 pandas 问题
            rel_display = gr.Markdown("*（尚未加载）*", label="已有自定义关联")

            gr.Markdown("""
**添加自定义关联**

> **填写方向：来源节点（上位） → 目标节点（下位）**
>
> 例：「开发业务EHS管理指引」指导「地区公司EHS管理体系」
> → 来源节点填「开发业务EHS管理指引」，目标节点填「地区公司EHS管理体系」
>
> | 关系类型 | 含义 | 适用场景 |
> |----------|------|----------|
> | `governs`（上位指导） | 上位对下位具有管控或约束力 | 集团规定约束地区细则 |
> | `applies_to`（适用于） | 某规定/体系适用于特定对象或范围 | 某指引适用于特定项目类型 |
> | `references`（互参） | 两节点内容相互引用，无明显上下级 | 两本手册互相参照 |
> | `shares`（共享内容） | 两节点共用同一部分内容或条款 | 两个子体系共用同一附录 |
""")

            with gr.Row():
                from_dd  = gr.Dropdown(label="来源节点", choices=[], interactive=True, scale=3)
                to_dd    = gr.Dropdown(label="目标节点", choices=[], interactive=True, scale=3)
                type_dd  = gr.Dropdown(
                    label="关系类型",
                    choices=[
                        "governs（上位指导）",
                        "applies_to（适用于）",
                        "references（互参）",
                        "shares（共享）",
                    ],
                    value="governs（上位指导）", interactive=True, scale=2,
                )
                add_rel_btn = gr.Button("➕ 添加", scale=1)

            del_chk = gr.CheckboxGroup(label="勾选要删除的关联（支持多选）", choices=[], interactive=True)
            with gr.Row():
                sel_all_btn = gr.Button("☑️ 全选", scale=1)
                del_rel_btn = gr.Button("🗑️ 删除所选", variant="stop", scale=1)

            with gr.Row():
                save_btn  = gr.Button("💾 仅保存修改", variant="secondary")
                apply_btn = gr.Button("✅ 确认并写入SQLite", variant="primary")

            result_box = gr.Textbox(label="操作结果", interactive=False, lines=2)

            # ── 辅助函数 ──
            def _rel_choices(rows):
                return [f"{i+1}: {r[0]} → {r[1]} ({r[2]})" for i, r in enumerate(rows)] if rows else []

            def _chk_update(rows):
                choices = _rel_choices(rows)
                return gr.CheckboxGroup(choices=choices, value=[])

            def _rows_to_md(rows):
                if not rows:
                    return "*（暂无自定义关联）*"
                lines = ["| 序号 | 来源节点 | 目标节点 | 关系类型 | 方法 |",
                         "|------|---------|---------|---------|------|"]
                for i, r in enumerate(rows):
                    lines.append(f"| {i+1} | {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
                return "\n".join(lines)

            def _load(fn):
                info, node_rows, rel_rows, node_names = load_yaml_file(fn)
                node_dd = gr.Dropdown(choices=node_names)
                return (info, node_rows, rel_rows,
                        _rows_to_md(rel_rows),
                        node_dd, node_dd,
                        _chk_update(rel_rows))

            def _save(fn, nd, state):
                return save_and_apply(fn, nd, state, "save")

            def _apply(fn, nd, state):
                return save_and_apply(fn, nd, state, "apply")

            def _add_rel(frm, to, typ, state):
                rows = list(state or [])
                if not frm or not to:
                    return (rows, _rows_to_md(rows), _chk_update(rows),
                            gr.Dropdown(value=frm), gr.Dropdown(value=to),
                            "❌ 请先选择来源节点和目标节点")
                typ_key = str(typ).split("（")[0].strip()
                rows.append([frm, to, typ_key, "manual"])
                return (rows, _rows_to_md(rows), _chk_update(rows),
                        gr.Dropdown(value=frm), gr.Dropdown(value=to),
                        f"✅ 已添加: {frm} → {to} ({typ_key})")

            def _del_rel(selected, state):
                rows = list(state or [])
                # 解析选中项的序号（从大到小删，避免下标偏移）
                indices = []
                for item in (selected or []):
                    try:
                        indices.append(int(str(item).split(":")[0]) - 1)
                    except (ValueError, TypeError):
                        pass
                for i in sorted(set(indices), reverse=True):
                    if 0 <= i < len(rows):
                        rows.pop(i)
                return (rows, _rows_to_md(rows), _chk_update(rows),
                        f"✅ 已删除 {len(indices)} 条关联")

            def _sel_all(state):
                rows = list(state or [])
                choices = _rel_choices(rows)
                return gr.CheckboxGroup(choices=choices, value=choices)

            _load_outputs = [file_info, node_table, rel_state,
                             rel_display, from_dd, to_dd, del_chk]

            refresh_btn.click(refresh_yaml_list, outputs=yaml_dd)
            load_btn.click(_load, inputs=yaml_dd, outputs=_load_outputs)
            yaml_dd.change(_load, inputs=yaml_dd, outputs=_load_outputs)
            add_rel_btn.click(_add_rel, inputs=[from_dd, to_dd, type_dd, rel_state],
                              outputs=[rel_state, rel_display, del_chk, from_dd, to_dd, result_box])
            sel_all_btn.click(_sel_all, inputs=rel_state, outputs=del_chk)
            del_rel_btn.click(_del_rel, inputs=[del_chk, rel_state],
                              outputs=[rel_state, rel_display, del_chk, result_box])
            save_btn.click(_save, inputs=[yaml_dd, node_table, rel_state], outputs=result_box)
            apply_btn.click(_apply, inputs=[yaml_dd, node_table, rel_state], outputs=result_box)

        # ══════════════════════════════════════════
        #  Tab 3: 图文内容审核
        # ══════════════════════════════════════════
        with gr.Tab("📄 图文内容审核"):
            gr.Markdown(
                "### 审核 Vision LLM 提取的幻灯片/文档图文内容\n"
                "**第一步**：设置全文件排除区域（页眉页脚等）。"
                "**第二步**：逐页核对图片与提取文字，确认后写入数据库。"
            )

            # ── 文件选择 ──
            def _list_review_files():
                files = sorted(f for f in os.listdir(REVIEW_DIR) if f.endswith('_img_review.yaml'))
                return files or ["（无待审核文件）"]

            with gr.Row():
                rv_dd = gr.Dropdown(
                    label="待审核文件",
                    choices=_list_review_files(),
                    value=_list_review_files()[0],
                    interactive=True, scale=4,
                )
                rv_load_btn    = gr.Button("📂 加载", size="sm", scale=1)
                rv_refresh_btn = gr.Button("🔄 刷新列表", size="sm", scale=1)

            rv_info = gr.Textbox(label="文件信息", interactive=False, lines=2)

            # ── Step 1: 排除区域 ──
            with gr.Accordion("【第一步】设置排除区域（页眉页脚等，设置一次应用全文件）", open=True):
                gr.Markdown(
                    "用画笔在图片上**涂抹**要排除的区域（页眉、页脚、水印等），"
                    "点击「应用到全文件」。系统自动检测涂抹范围，并以背景色填充。"
                )
                zone_editor = gr.ImageEditor(
                    label="在此处涂抹排除区域（建议用矩形画笔横向刷过页眉/页脚）",
                    type="numpy",
                    height=420,
                    brush=gr.Brush(colors=["#FF0000"], default_color="#FF0000", default_size=20),
                )
                zone_apply_btn = gr.Button("✅ 应用到全文件", variant="primary")
                zone_status    = gr.Textbox(label="排除区域状态", interactive=False, lines=1)

            # ── Step 2: 逐页审核 ──
            with gr.Accordion("【第二步】逐页审核", open=True):
                # 批量操作行
                with gr.Row():
                    rv_batch_text_btn   = gr.Button("✅ 批量通过所有「文字提取」页", variant="primary", scale=2)
                    rv_batch_blank_btn  = gr.Button("⏭ 批量跳过所有「近空白」页",   variant="secondary", scale=2)
                    rv_jump_vision_btn  = gr.Button("🔍 跳到下一个 Vision 页",       scale=1)
                rv_batch_result = gr.Textbox(label="批量操作结果", interactive=False, lines=1)

                with gr.Row():
                    rv_page_info = gr.Textbox(label="当前进度", interactive=False, scale=4)
                    rv_prev_btn  = gr.Button("⬅ 上一页", scale=1)
                    rv_next_btn  = gr.Button("下一页 ➡", scale=1)

                with gr.Row():
                    rv_img  = gr.Image(label="幻灯片图片（已应用排除区域）",
                                       type="numpy", height=420, scale=1, interactive=False)
                    with gr.Column(scale=1):
                        rv_text = gr.Textbox(
                            label="提取内容（可直接编辑修改）",
                            lines=16, interactive=True,
                        )
                        with gr.Row():
                            rv_approve_btn = gr.Button("✅ 通过", variant="primary", scale=1)
                            rv_skip_btn    = gr.Button("⏭ 跳过此页", variant="secondary", scale=1)

                rv_action_result = gr.Textbox(label="操作提示", interactive=False, lines=1)

            # ── 写入数据库 ──
            rv_write_btn    = gr.Button("💾 全部审核完毕，写入数据库", variant="primary")
            rv_write_result = gr.Textbox(label="写入结果", interactive=False, lines=2)

            # ── State ──
            rv_state      = gr.State({})   # 当前 review YAML 数据
            rv_page_state = gr.State(0)    # 当前页 index (0-based)

            # ── 辅助函数 ──
            def _rv_list():
                files = _list_review_files()
                return gr.Dropdown(choices=files, value=files[0])

            def _load_review(filename):
                """加载审核YAML，返回 (info, state, page_idx=0, img, text, page_info)"""
                empty = ("请先选择文件", {}, 0, None, "", "")
                if not filename or filename == "（无待审核文件）":
                    return empty
                path = os.path.join(REVIEW_DIR, filename)
                if not os.path.exists(path):
                    return ("文件不存在", {}, 0, None, "", "")
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                pages = data.get('pages', [])
                total = len(pages)
                pending = sum(1 for p in pages if p.get('status') == 'pending')
                info = (f"来源: {data.get('source_file', '')}  |  "
                        f"共 {total} 页  |  待审核 {pending} 页  |  状态: {data.get('status', 'pending')}")
                # 加载第一页
                img, text, page_info = _render_page(data, 0)
                # 同步 zone_editor 显示第一页图片
                first_img = _load_raw_img(data, 0)
                return info, data, 0, img, text, page_info, first_img

            def _load_raw_img(data, idx):
                """加载原始图片（供 zone_editor 使用）"""
                pages = data.get('pages', [])
                if not pages or idx >= len(pages):
                    return None
                img_rel = pages[idx].get('image_path', '')
                img_path = os.path.join(BASE_DIR, img_rel)
                if not os.path.exists(img_path):
                    return None
                return np.array(Image.open(img_path).convert('RGB'))

            def _apply_zones_to_arr(img_arr, zones):
                """对 numpy 图片数组应用排除区域（以背景色填充）。"""
                if not zones or img_arr is None:
                    return img_arr
                arr = img_arr.copy()
                h, w = arr.shape[:2]
                for z in zones:
                    x1 = int(z['rx1'] * w); y1 = int(z['ry1'] * h)
                    x2 = int(z['rx2'] * w); y2 = int(z['ry2'] * h)
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    # 自动取周边背景色（上下各3行取中位）
                    samples = []
                    if y1 >= 3:
                        samples.append(arr[y1-3:y1, x1:x2].reshape(-1, 3))
                    if y2 + 3 <= h:
                        samples.append(arr[y2:y2+3, x1:x2].reshape(-1, 3))
                    if samples:
                        bg = tuple(int(c) for c in np.median(np.vstack(samples), axis=0))
                    else:
                        bg = (255, 255, 255)
                    arr[y1:y2, x1:x2] = bg
                return arr

            def _render_page(data, idx):
                """渲染当前页：加载图片 + 应用排除区域，返回 (img_arr, text, page_info)。"""
                pages = data.get('pages', [])
                if not pages:
                    return None, "", "无页面数据"
                idx = max(0, min(idx, len(pages) - 1))
                page = pages[idx]
                img_rel  = page.get('image_path', '')
                img_path = os.path.join(BASE_DIR, img_rel)
                img_arr  = None
                if os.path.exists(img_path):
                    img_arr = np.array(Image.open(img_path).convert('RGB'))
                    zones   = data.get('exclusion_zones', [])
                    img_arr = _apply_zones_to_arr(img_arr, zones)

                text = page.get('final_text') or page.get('raw_text', '')
                total   = len(pages)
                done    = sum(1 for p in pages if p.get('status') in ('approved', 'edited', 'skipped'))
                status  = page.get('status', 'pending')
                method  = page.get('extraction_method', '')
                page_info = (f"第 {idx+1}/{total} 页  |  状态: {status}  |  提取方式: {method}  "
                             f"|  已完成: {done}/{total}")
                return img_arr, text, page_info

            def _detect_zones(editor_output, data):
                """从 ImageEditor 输出检测涂抹区域，存入 data['exclusion_zones']。"""
                if editor_output is None or not data:
                    return data, "⚠️ 请先加载文件并在图片上涂抹排除区域"
                layers = editor_output.get('layers', [])
                if not layers:
                    return data, "⚠️ 未检测到涂抹内容，请用画笔在图片上涂抹需要排除的区域"
                layer = layers[0]
                if layer is None:
                    return data, "⚠️ 图层为空"
                layer_arr = np.array(layer) if not isinstance(layer, np.ndarray) else layer
                if layer_arr.ndim != 3 or layer_arr.shape[2] < 4:
                    return data, "⚠️ 图层格式异常，请重试"
                alpha = layer_arr[:, :, 3]
                mask  = alpha > 10
                if not mask.any():
                    return data, "⚠️ 未检测到涂抹内容"
                lh, lw = layer_arr.shape[:2]
                # 分行扫描：连续有涂抹的行视为一个区域（支持上下多个排除带）
                row_has_paint = np.any(mask, axis=1)
                zones = []
                in_zone = False
                zone_start = 0
                for r, has in enumerate(row_has_paint):
                    if has and not in_zone:
                        in_zone = True; zone_start = r
                    elif not has and in_zone:
                        in_zone = False
                        # 找此行范围内左右边界
                        col_mask = np.any(mask[zone_start:r, :], axis=0)
                        cmin = int(np.where(col_mask)[0][0])  if col_mask.any() else 0
                        cmax = int(np.where(col_mask)[0][-1]) if col_mask.any() else lw
                        zones.append({
                            'rx1': 0.0, 'ry1': zone_start / lh,
                            'rx2': 1.0, 'ry2': r / lh,
                        })
                if in_zone:
                    zones.append({'rx1': 0.0, 'ry1': zone_start / lh, 'rx2': 1.0, 'ry2': 1.0})
                data = dict(data)
                data['exclusion_zones'] = zones
                # 保存回 YAML
                fname = _find_review_fname(data)
                if fname:
                    _save_review_yaml(data, fname)
                msg = f"✅ 检测到 {len(zones)} 个排除区域，已应用到全部 {len(data.get('pages',[]))} 页"
                return data, msg

            def _find_review_fname(data):
                src = data.get('source_file', '')
                if not src:
                    return None
                base = re.sub(r'[^\w\u4e00-\u9fa5-]', '_', os.path.splitext(os.path.basename(src))[0])
                return base + '_img_review.yaml'

            def _save_review_yaml(data, fname):
                path = os.path.join(REVIEW_DIR, fname)
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            def _navigate(data, idx, direction):
                """direction: +1 / -1"""
                pages = data.get('pages', [])
                new_idx = max(0, min(idx + direction, len(pages) - 1))
                img, text, page_info = _render_page(data, new_idx)
                return new_idx, img, text, page_info

            def _approve_page(data, idx, text):
                """标记当前页为 approved，文字使用用户编辑后的内容。"""
                data  = dict(data)
                pages = list(data.get('pages', []))
                page  = dict(pages[idx])
                page['status']     = 'approved' if text == page.get('raw_text', '') else 'edited'
                page['final_text'] = text
                pages[idx] = page
                data['pages'] = pages
                fname = _find_review_fname(data)
                if fname:
                    _save_review_yaml(data, fname)
                done  = sum(1 for p in pages if p.get('status') in ('approved', 'edited', 'skipped'))
                total = len(pages)
                # 自动跳下一页
                new_idx = min(idx + 1, total - 1)
                img, new_text, page_info = _render_page(data, new_idx)
                msg = f"✅ 第{idx+1}页已{page['status']}  |  进度 {done}/{total}"
                return data, new_idx, img, new_text, page_info, msg

            def _skip_page(data, idx):
                """标记当前页为 skipped，不入库。"""
                data  = dict(data)
                pages = list(data.get('pages', []))
                page  = dict(pages[idx])
                page['status'] = 'skipped'
                pages[idx] = page
                data['pages'] = pages
                fname = _find_review_fname(data)
                if fname:
                    _save_review_yaml(data, fname)
                done  = sum(1 for p in pages if p.get('status') in ('approved', 'edited', 'skipped'))
                total = len(pages)
                new_idx = min(idx + 1, total - 1)
                img, text, page_info = _render_page(data, new_idx)
                msg = f"⏭ 第{idx+1}页已跳过  |  进度 {done}/{total}"
                return data, new_idx, img, text, page_info, msg

            def _batch_approve_text(data, idx):
                """批量通过所有 extraction_method='text' 且状态为 pending 的页面。"""
                if not data:
                    return data, idx, None, "", "", "⚠️ 请先加载文件"
                data  = dict(data)
                pages = list(data.get('pages', []))
                count = 0
                for i, page in enumerate(pages):
                    if page.get('status') == 'pending' and page.get('extraction_method') == 'text':
                        page = dict(page)
                        page['status']     = 'approved'
                        page['final_text'] = page.get('raw_text', '')
                        pages[i] = page
                        count += 1
                data['pages'] = pages
                fname = _find_review_fname(data)
                if fname:
                    _save_review_yaml(data, fname)
                # 跳到第一个仍 pending 的页
                new_idx = next((i for i, p in enumerate(pages) if p.get('status') == 'pending'), idx)
                new_idx = max(0, min(new_idx, len(pages) - 1))
                img, text, page_info = _render_page(data, new_idx)
                done = sum(1 for p in pages if p.get('status') in ('approved', 'edited', 'skipped'))
                msg = f"✅ 批量通过 {count} 个文字提取页  |  剩余待审 {len(pages)-done} 页"
                return data, new_idx, img, text, page_info, msg

            def _batch_skip_blank(data, idx, blank_threshold=20):
                """批量跳过所有文字少于 blank_threshold 字且状态为 pending 的页面。"""
                if not data:
                    return data, idx, None, "", "", "⚠️ 请先加载文件"
                data  = dict(data)
                pages = list(data.get('pages', []))
                count = 0
                for i, page in enumerate(pages):
                    if page.get('status') == 'pending':
                        text = page.get('raw_text', '') or ''
                        if len(text.strip()) < blank_threshold:
                            page = dict(page)
                            page['status'] = 'skipped'
                            pages[i] = page
                            count += 1
                data['pages'] = pages
                fname = _find_review_fname(data)
                if fname:
                    _save_review_yaml(data, fname)
                new_idx = next((i for i, p in enumerate(pages) if p.get('status') == 'pending'), idx)
                new_idx = max(0, min(new_idx, len(pages) - 1))
                img, text, page_info = _render_page(data, new_idx)
                done = sum(1 for p in pages if p.get('status') in ('approved', 'edited', 'skipped'))
                msg = f"⏭ 批量跳过 {count} 个近空白页  |  剩余待审 {len(pages)-done} 页"
                return data, new_idx, img, text, page_info, msg

            def _jump_to_next_vision(data, idx):
                """跳转到当前页之后第一个 extraction_method='vision' 且 pending 的页面。"""
                if not data:
                    return idx, None, "", "", "⚠️ 请先加载文件"
                pages = data.get('pages', [])
                # 从 idx+1 开始找，找不到则从头找
                for i in list(range(idx + 1, len(pages))) + list(range(0, idx + 1)):
                    p = pages[i]
                    if p.get('extraction_method') == 'vision' and p.get('status') == 'pending':
                        img, text, page_info = _render_page(data, i)
                        return i, img, text, page_info, f"🔍 已跳转到第 {i+1} 页（Vision 待审）"
                img, text, page_info = _render_page(data, idx)
                return idx, img, text, page_info, "✅ 没有更多待审的 Vision 页面了"

            def _write_to_db(data):
                """将 approved/edited 页写入 ChromaDB，标记文件为 completed。"""
                if not data:
                    return "⚠️ 请先加载并审核文件"
                pages = data.get('pages', [])
                pending = [p for p in pages if p.get('status') == 'pending']
                if pending:
                    return f"⚠️ 还有 {len(pending)} 页未审核，请全部处理后再写入"
                to_write = [p for p in pages if p.get('status') in ('approved', 'edited')]
                if not to_write:
                    return "⚠️ 没有通过审核的页面，不写入数据库"
                try:
                    from langchain_core.documents import Document as LCDoc
                    db  = _get_db()
                    meta_base = data.get('metadata', {})
                    chunks = []
                    for page in to_write:
                        text = page.get('final_text') or page.get('raw_text', '')
                        if not text.strip():
                            continue
                        meta = {
                            **meta_base,
                            'type':       'slide_vision',
                            'page_num':   page.get('page_num', 0),
                            'image_path': page.get('image_path', ''),
                        }
                        chunks.append(LCDoc(page_content=text, metadata=meta))
                    if chunks:
                        db.add_documents(chunks)
                    # 更新 YAML 状态
                    data  = dict(data)
                    data['status'] = 'completed'
                    fname = _find_review_fname(data)
                    if fname:
                        _save_review_yaml(data, fname)
                    skipped = len([p for p in pages if p.get('status') == 'skipped'])
                    return (f"✅ 成功写入 {len(chunks)} 个文本块  |  "
                            f"跳过 {skipped} 页  |  文件状态已标记为 completed")
                except Exception as e:
                    return f"❌ 写入失败: {e}"

            # ── 连线 ──
            _load_outputs = [rv_info, rv_state, rv_page_state,
                             rv_img, rv_text, rv_page_info, zone_editor]

            _batch_outputs = [rv_state, rv_page_state, rv_img, rv_text, rv_page_info, rv_batch_result]

            rv_refresh_btn.click(_rv_list, outputs=rv_dd)
            rv_load_btn.click(_load_review, inputs=rv_dd, outputs=_load_outputs)
            rv_dd.change(_load_review,  inputs=rv_dd, outputs=_load_outputs)

            rv_batch_text_btn.click(
                _batch_approve_text,
                inputs=[rv_state, rv_page_state],
                outputs=_batch_outputs,
            )
            rv_batch_blank_btn.click(
                _batch_skip_blank,
                inputs=[rv_state, rv_page_state],
                outputs=_batch_outputs,
            )
            rv_jump_vision_btn.click(
                _jump_to_next_vision,
                inputs=[rv_state, rv_page_state],
                outputs=[rv_page_state, rv_img, rv_text, rv_page_info, rv_batch_result],
            )

            zone_apply_btn.click(
                _detect_zones,
                inputs=[zone_editor, rv_state],
                outputs=[rv_state, zone_status],
            )

            rv_prev_btn.click(
                lambda d, i: _navigate(d, i, -1),
                inputs=[rv_state, rv_page_state],
                outputs=[rv_page_state, rv_img, rv_text, rv_page_info],
            )
            rv_next_btn.click(
                lambda d, i: _navigate(d, i, +1),
                inputs=[rv_state, rv_page_state],
                outputs=[rv_page_state, rv_img, rv_text, rv_page_info],
            )
            rv_approve_btn.click(
                _approve_page,
                inputs=[rv_state, rv_page_state, rv_text],
                outputs=[rv_state, rv_page_state, rv_img, rv_text, rv_page_info, rv_action_result],
            )
            rv_skip_btn.click(
                _skip_page,
                inputs=[rv_state, rv_page_state],
                outputs=[rv_state, rv_page_state, rv_img, rv_text, rv_page_info, rv_action_result],
            )
            rv_write_btn.click(
                _write_to_db,
                inputs=[rv_state],
                outputs=[rv_write_result],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8081)

