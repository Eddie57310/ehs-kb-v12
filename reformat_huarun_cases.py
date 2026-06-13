#!/usr/bin/env python3
"""
reformat_huarun_cases.py
把华润置地示范区提质增效案例集 content_list.json 重新解析为干净 MD。
"""

import json
import re
import os
from collections import defaultdict, OrderedDict
from bs4 import BeautifulSoup

# ── 路径配置 ──────────────────────────────────────────────
INPUT_JSON = os.path.expanduser(
    "~/doc_parser_v12/batch_output/工管部/"
    "20260601_华润置地示范区提质增效案例集-2026年V1/"
    "华润置地示范区提质增效案例集-2026年V1/auto/"
    "华润置地示范区提质增效案例集-2026年V1_content_list.json"
)
OUTPUT_MD = os.path.expanduser(
    "~/doc_parser_v12/reviewed_md/工管部/"
    "20260601_华润置地示范区提质增效案例集-2026年V1/"
    "华润置地示范区提质增效案例集-2026年V1.md"
)

LEFT_LABELS = ["方法论归属", "核心成果", "适用条件", "具体措施", "开发阶段"]


def clean_text(s: str) -> str:
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── 手动 rowspan 追踪，正确提取左右列 ─────────────────────

def _extract_columns(soup):
    rows = soup.find_all('tr')
    rowspan_spill = []

    left_cells = []
    right_cells_raw = []
    seen_left = set()
    seen_right = set()

    for tr in rows:
        tds = tr.find_all('td')

        new_spill = []
        for rem, col, txt in rowspan_spill:
            if rem > 1:
                new_spill.append((rem - 1, col, txt))
        rowspan_spill = new_spill

        col = 0
        right_texts_this_row = []

        for td in tds:
            while any(col == sc for _, sc, _ in rowspan_spill):
                for rem, sc, txt in rowspan_spill:
                    if sc == col and sc >= 1 and txt:
                        if txt not in seen_right:
                            seen_right.add(txt)
                            right_texts_this_row.append(txt)
                col += 1

            text = clean_text(td.get_text())
            rowspan = int(td.get('rowspan', 1)) if td.has_attr('rowspan') else 1
            colspan = int(td.get('colspan', 1)) if td.has_attr('colspan') else 1

            if col == 0:
                if text and text not in seen_left:
                    seen_left.add(text)
                    left_cells.append(text)
            else:
                if text and text not in seen_right:
                    seen_right.add(text)
                    right_texts_this_row.append(text)

            if rowspan > 1:
                rowspan_spill.append((rowspan, col, text))

            col += colspan

        while any(col == sc for _, sc, _ in rowspan_spill):
            for rem, sc, txt in rowspan_spill:
                if sc == col and sc >= 1 and txt:
                    if txt not in seen_right:
                        seen_right.add(txt)
                        right_texts_this_row.append(txt)
            col += 1

        if right_texts_this_row:
            right_cells_raw.append(' '.join(right_texts_this_row))

    return left_cells, right_cells_raw


# ── 左列解析（状态机） ────────────────────────────────────

def extract_left_fields(soup):
    left_cells, _ = _extract_columns(soup)
    fields = OrderedDict()
    current_label = None

    for text in left_cells:
        if text in ("【方法论部分】", "方法论标题", "【案例部分】",):
            continue

        if text in ("核心成果", "适用条件", "具体措施", "开发阶段"):
            current_label = text
            continue

        # 检测所有 LEFT_LABELS 的匹配（可能多个在同文本中）
        best_match = None
        best_pos = len(text)  # 最早的匹配位置

        for lbl in LEFT_LABELS:
            m = re.search(re.escape(lbl), text)
            if m and m.start() < best_pos:
                best_match = (lbl, m.start(), m.end())
                best_pos = m.start()

        if best_match:
            lbl, start, end = best_match
            before = clean_text(text[:start])
            after = clean_text(text[end:])

            # 标签之前的内容 → 追加到 current_label
            if before and current_label:
                prev = fields.get(current_label, "")
                fields[current_label] = (prev + " " + before).strip() if prev else before

            # 标签之后的内容 → 新字段
            if after:
                fields[lbl] = after

            current_label = lbl
        elif text.strip().rstrip("：:") in LEFT_LABELS:
            current_label = text.strip().rstrip("：:")
        elif current_label:
            prev = fields.get(current_label, "")
            fields[current_label] = (prev + " " + text).strip() if prev else text

    # 将 "开发阶段" 合并到 "方法论归属"（如果后者不存在）
    if "开发阶段" in fields and "方法论归属" not in fields:
        fields["方法论归属"] = fields.pop("开发阶段")

    return fields


# ── 右列解析（状态机） ────────────────────────────────────

def _is_header_cell(text: str) -> bool:
    if re.match(r'^【案例部分】\s*(案例标识)?\s*$', text):
        return True
    if text in ("案例标识", "关键数据", "案例详述", "案例详述（续）",
                "案例详述（续)", "案例详述(续)", "案例详述 (续)",
                "【案例部分】",):
        return True
    return False


def extract_right_fields(soup):
    _, right_cells_raw = _extract_columns(soup)

    cells = []
    seen = set()
    for text in right_cells_raw:
        appended = False
        for prev in seen:
            if text.startswith(prev) and len(text) > len(prev):
                extra = clean_text(text[len(prev):])
                if extra:
                    cells.append(extra)
                appended = True
                break
        if not appended and text not in seen:
            seen.add(text)
            cells.append(text)

    fields = OrderedDict()
    current_label = None

    for text in cells:
        if _is_header_cell(text):
            continue

        if re.match(r'^案例详述\s*[（(]续[）)]\s*$', text):
            current_label = "案例详述"
            continue

        m = re.match(r'案例标识及关键数据\s*(.+)', text)
        if m:
            content = m.group(1).strip()
            # 在内容中查找 "关键数据" 和 "案例详述" 分隔符
            kd = re.search(r'关键数据\s*', content)
            xq = re.search(r'案例详述\s*', content)
            if kd and xq:
                # 取最小位置作为第一个分隔
                if kd.start() < xq.start():
                    fields["案例标识"] = clean_text(content[:kd.start()])
                    after_kd = clean_text(content[kd.end():xq.start()])
                    if after_kd:
                        fields["关键数据"] = after_kd
                    after_xq = clean_text(content[xq.end():]).lstrip("（续）").lstrip("(续)").strip()
                    if after_xq:
                        fields["案例详述"] = after_xq
                else:
                    fields["案例标识"] = clean_text(content[:xq.start()])
                    after_xq = clean_text(content[xq.end():]).lstrip("（续）").lstrip("(续)").strip()
                    if after_xq:
                        fields["案例详述"] = after_xq
            elif kd:
                fields["案例标识"] = clean_text(content[:kd.start()])
                kd_content = clean_text(content[kd.end():])
                if kd_content:
                    fields["关键数据"] = kd_content
            elif xq:
                fields["案例标识"] = clean_text(content[:xq.start()])
                after_xq = clean_text(content[xq.end():]).lstrip("（续）").lstrip("(续)").strip()
                if after_xq:
                    fields["案例详述"] = after_xq
            else:
                fields["案例标识"] = content
            current_label = None  # 案例标识已完整，后续不再续写
            continue

        m = re.match(r'案例标识[：:]*\s*(.+)', text)
        if m and m.group(1).strip():
            fields["案例标识"] = m.group(1).strip()
            current_label = "案例标识"
            continue

        m = re.match(r'关键数据[：:]*\s*(.+)', text)
        if m and m.group(1).strip():
            fields["关键数据"] = m.group(1).strip()
            current_label = "关键数据"
            continue

        m_xq = re.search(r'案例详述\s*[（(]续[）)]\s*', text)
        if m_xq:
            before = clean_text(text[:m_xq.start()])
            after = clean_text(text[m_xq.end():])
            if before and current_label:
                prev = fields.get(current_label, "")
                fields[current_label] = (prev + " " + before).strip() if prev else before
            if after:
                prev = fields.get("案例详述", "")
                fields["案例详述"] = (prev + " " + after).strip() if prev else after
            current_label = "案例详述"
            continue

        m = re.match(r'案例详述[：:]*\s*(.+)', text)
        if m and m.group(1).strip():
            fields["案例详述"] = m.group(1).strip()
            current_label = "案例详述"
            continue

        if "案例标识" not in fields:
            kd_pos = text.find("关键数据")
            xq_pos = text.find("案例详述")
            if kd_pos >= 0 or xq_pos >= 0:
                split_pos = min(p for p in [kd_pos, xq_pos] if p >= 0)
                fields["案例标识"] = clean_text(text[:split_pos])
                remaining = clean_text(text[split_pos:])
                if kd_pos >= 0 and kd_pos == split_pos:
                    kd_m = re.match(r'关键数据[：:]*\s*', remaining)
                    if kd_m:
                        remaining = remaining[kd_m.end():]
                    xq_pos2 = remaining.find("案例详述")
                    if xq_pos2 >= 0:
                        fields["关键数据"] = clean_text(remaining[:xq_pos2])
                        xq_detail = remaining[xq_pos2 + 4:]
                        xq_detail = re.sub(r'^[（(]续[）)]\s*', '', xq_detail).strip()
                        if xq_detail:
                            fields["案例详述"] = xq_detail
                    else:
                        fields["关键数据"] = remaining
                elif xq_pos >= 0 and xq_pos == split_pos:
                    xq_m = re.match(r'案例详述[（(]?续[）)]?[：:]*\s*', remaining)
                    if xq_m:
                        remaining = remaining[xq_m.end():]
                    fields["案例详述"] = remaining
            else:
                fields["案例标识"] = text
            current_label = "案例标识"
            continue

        if current_label:
            prev = fields.get(current_label, "")
            fields[current_label] = (prev + " " + text).strip() if prev else text
        elif text.strip() and not re.match(r'^[（(]续[）)]\s*$', text):
            # 无当前标签但有文本 → 兜底追加到案例详述
            prev = fields.get("案例详述", "")
            fields["案例详述"] = (prev + " " + text).strip() if prev else text

    return fields


# ── 案例编号 ──────────────────────────────────────────────

def get_case_number(page_entries):
    for entry in page_entries:
        if entry["type"] == "text" and entry.get("text_level") == 1:
            t = clean_text(entry["text"])
            if t:
                return t
    for entry in page_entries:
        if entry["type"] == "table":
            captions = entry.get("table_caption", [])
            if captions:
                t = clean_text(captions[0])
                if t:
                    return t
    return None


def get_images(page_entries):
    imgs = []
    for entry in page_entries:
        if entry["type"] == "image":
            path = entry.get("img_path", "")
            fname = os.path.basename(path)
            if fname:
                imgs.append(fname)
    return imgs


def parse_page(page_entries):
    main_table = None
    for entry in page_entries:
        if entry["type"] == "table":
            main_table = entry
            break

    if main_table is None:
        case_no = get_case_number(page_entries)
        return {"case_number": case_no, "fields": {}, "images": get_images(page_entries),
                "skip": True}

    soup = BeautifulSoup(main_table["table_body"], "html.parser")
    left_fields = extract_left_fields(soup)
    right_fields = extract_right_fields(soup)

    fields = OrderedDict()
    fields.update(left_fields)
    fields.update(right_fields)

    case_no = get_case_number(page_entries)
    images = get_images(page_entries)

    return {"case_number": case_no, "fields": fields, "images": images, "skip": False}


def infer_case_numbers(cases):
    cat_counters = {}
    for case in cases:
        if case.get("skip"):
            continue
        cn = case.get("case_number")
        if cn:
            m = re.match(r'^(.+?-(\d{3}))$', cn)
            if m:
                prefix = m.group(1)
                num = int(m.group(2))
                cat_counters.setdefault(prefix, set()).add(num)

    for case in cases:
        if case.get("skip"):
            continue
        if case.get("case_number"):
            continue
        guishu = case["fields"].get("方法论归属", "")
        if not guishu:
            # 最后尝试从其他字段推断一个分类
            case["case_number"] = "UNKNOWN"
            continue
        cat_raw = guishu.split("：")[0].split(":")[0].strip()
        cat_clean = re.sub(r'[【】""（）()\s]', '', cat_raw)
        prefix = f"示范区-{cat_clean}-2026"
        used = cat_counters.get(prefix, set())
        n = 1
        while n in used:
            n += 1
        used.add(n)
        cat_counters[prefix] = used
        case["case_number"] = f"{prefix}-{n:03d}"

    return cases


def cleanup_fields(cases):
    garbage_patterns = [
        (r'^[：:.。，,、]\s*', ''),
        (r'^\s*[（(]续[）)]\s*', ''),
        (r'^案例详述\s*', ''),
        (r'^关键数据\s*', ''),
        (r'^案例标识\s*', ''),
        (r'^案例标识及关键数据\s*', ''),
        (r'\s*案例详述[（(]?续[）)]?\s*$', ''),
        (r'\s*关键数据\s*$', ''),
        (r'\s*案例标识\s*$', ''),
        (r'\s*案例标识及关键数据\s*$', ''),
        (r'\s*照片[：:]*\s*(如有|无)?\s*$', ''),
        (r'\s*中华人民共和国.*$', ''),
        (r'\s*建设工程规划许可证\s*$', ''),
        (r'\s*华人民共和国城乡规划法.*$', ''),
    ]

    for case in cases:
        keys_to_del = []
        for key, val in list(case["fields"].items()):
            for pat, repl in garbage_patterns:
                val = re.sub(pat, repl, val).strip()
            val = re.sub(r'\s+', ' ', val).strip()
            if not val:
                keys_to_del.append(key)
            else:
                case["fields"][key] = val
        for k in keys_to_del:
            del case["fields"][k]


def render_case(case: dict) -> str:
    lines = []
    cn = case.get("case_number", "UNKNOWN")
    if cn in ("UNKNOWN", None):
        cn = case["fields"].get("案例标识", "案例")
    lines.append(f"## {cn}")
    lines.append("")

    field_order = [
        "案例标识", "方法论归属", "核心成果", "适用条件",
        "具体措施", "关键数据", "案例详述",
    ]
    for key in field_order:
        val = case["fields"].get(key, "")
        if not val:
            continue
        lines.append(f"**{key}**：{val}")
        lines.append("")

    for img in case.get("images", []):
        lines.append(f"[📷 images/{img}]")
    if case.get("images"):
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = defaultdict(list)
    for entry in data:
        groups[entry["page_idx"]].append(entry)

    cases = []
    for page_idx in sorted(groups.keys()):
        if page_idx == 0:
            continue
        entries = groups[page_idx]
        cases.append(parse_page(entries))

    # 先清洗字段（去除冒号等杂讯），再推断编号（依赖干净的字段值）
    cleanup_fields(cases)
    cases = infer_case_numbers(cases)

    # 过滤 skip 的案例（无表格的空壳）
    valid_cases = [c for c in cases if not c.get("skip")]

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# 华润置地示范区提质增效案例集 v1（2026年6月）\n\n")
        for case in valid_cases:
            f.write(render_case(case))
            f.write("\n\n")

    print(f"共处理 {len(valid_cases)} 个案例")
    print(f"输出: {OUTPUT_MD}")
    print()

    for i, case in enumerate(valid_cases[:3]):
        print(f"=== 案例 {i+1} ===")
        print(render_case(case))
        print()


if __name__ == "__main__":
    main()
