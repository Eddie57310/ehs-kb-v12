"""
解析 sync_kb_v11 日志，生成处理报告。
用法：venv/bin/python parse_sync_log.py [日志文件路径]
不传路径则自动取最新的 sync_rebuild_*.log
"""
import re, sys, os
from datetime import datetime
from pathlib import Path

LOG_DIR = os.path.expanduser("~/doc_parser_v12/logs")


def find_latest_log():
    logs = sorted(Path(LOG_DIR).glob("sync_kb_v11_*.log"), key=lambda p: p.stat().st_mtime)
    return str(logs[-1]) if logs else None


def parse_log(log_path):
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    files = []       # 每个文件的处理记录
    current = None   # 当前文件状态

    def _save():
        if current:
            files.append(dict(current))

    for line in lines:
        line = line.rstrip()

        # ── 新文件开始 ──
        m = re.search(r'处理: (.+?)  \[(.+?) \| (.+?)\]', line)
        if m:
            _save()
            current = {
                "name":     m.group(1),
                "category": m.group(2),
                "date":     m.group(3),
                "method":   [],
                "chunks":   0,
                "ocr_total": 0,
                "ocr_fixed": 0,
                "ocr_calls": 0,
                "ocr_fail":  0,
                "vision_pages": 0,
                "vision_fail": 0,
                "status":   "processing",
                "notes":    [],
            }
            continue

        if current is None:
            continue

        # ── 处理方法检测 ──
        if "MinerU 正在解析" in line:
            current["method"].append("MinerU")
        elif "分批校对" in line or "逐块校对" in line:
            m2 = re.search(r'共(\d+)块', line)
            if m2:
                current["ocr_total"] = int(m2.group(1))
            if "MinerU+OCR" not in current["method"]:
                current["method"] = [m for m in current["method"] if m != "MinerU"]
                current["method"].append("MinerU+OCR校对")
        elif "切换 pdfplumber" in line or "pdfplumber 提取" in line:
            if "pdfplumber" not in current["method"]:
                current["method"].append("pdfplumber")
        elif "切换 pymupdf" in line or "pymupdf 快速提取" in line or "pymupdf 提取" in line:
            if "pymupdf" not in current["method"]:
                current["method"].append("pymupdf")
        elif "Vision LLM" in line and "页文字不足" in line:
            current["vision_pages"] += 1
            if "Vision LLM" not in current["method"]:
                current["method"].append("Vision LLM")
        elif "审核文件已生成" in line:
            m2 = re.search(r'\((\d+)页', line)
            if m2 and "vision_total" not in current:
                current["vision_total"] = int(m2.group(1))
            current["status"] = "pending_review"

        # ── OCR 校对统计 ──
        elif "LLM 校对完成" in line:
            m2 = re.search(r'调用(\d+)次，修正了\s*(\d+)/(\d+)', line)
            if m2:
                current["ocr_calls"] = int(m2.group(1))
                current["ocr_fixed"] = int(m2.group(2))
                current["ocr_total"] = int(m2.group(3))
        elif "LLM校对失败" in line:
            current["ocr_fail"] += 1

        # ── Vision LLM 失败 ──
        elif "Vision LLM 调用失败" in line:
            current["vision_fail"] += 1
            m2 = re.search(r'第(\d+)/\d+页', lines[lines.index(line+"\n")-1] if line+"\n" in lines else "")
            if m2:
                current["notes"].append(f"第{m2.group(1)}页Vision失败")

        # ── 入库结果 ──
        elif "入库" in line and "块" in line and "✅" in line:
            m2 = re.search(r'入库\s*(\d+)\s*块', line)
            if m2:
                current["chunks"] = int(m2.group(1))
            if current["status"] == "processing":
                current["status"] = "success"

        # ── 失败 ──
        elif "三引擎全部失败" in line or "移入失败隔离区" in line:
            current["status"] = "failed"

        # ── 额外备注 ──
        elif "检测到乱码" in line:
            current["notes"].append("检测到乱码，切换pdfplumber")
        elif "弱页救援" in line:
            current["notes"].append("有弱页，启动救援")
        elif "过滤目录块" in line:
            m2 = re.search(r'过滤目录块\s*(\d+)', line)
            if m2:
                current["notes"].append(f"过滤TOC块{m2.group(1)}个")

    _save()
    return files


def generate_report(files, log_path):
    total       = len(files)
    success     = [f for f in files if f["status"] == "success"]
    pending     = [f for f in files if f["status"] == "pending_review"]
    failed      = [f for f in files if f["status"] == "failed"]
    ocr_files   = [f for f in files if f["ocr_total"] > 0]
    ocr_fail_f  = [f for f in files if f["ocr_fail"] > 0]
    vision_fail_f = [f for f in files if f["vision_fail"] > 0]
    total_chunks = sum(f["chunks"] for f in files)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# Sync 处理报告")
    lines.append(f"**生成时间**: {now}  |  **日志文件**: {os.path.basename(log_path)}\n")

    lines.append("## 一、汇总")
    lines.append(f"| 项目 | 数量 |")
    lines.append(f"|------|------|")
    lines.append(f"| 处理文件总数 | {total} |")
    lines.append(f"| 成功入库 | {len(success)} |")
    lines.append(f"| 待人工审核 | {len(pending)} |")
    lines.append(f"| 失败 | {len(failed)} |")
    lines.append(f"| 总入库块数 | {total_chunks} |")
    lines.append(f"| 触发OCR校对文件数 | {len(ocr_files)} |")
    lines.append(f"| OCR校对有失败的文件 | {len(ocr_fail_f)} |")
    lines.append(f"| Vision LLM识别失败的文件 | {len(vision_fail_f)} |")
    lines.append("")

    # ── 需重点关注 ──
    attention = ocr_fail_f + vision_fail_f + failed
    attention = list({f["name"]: f for f in attention}.values())  # 去重
    if attention:
        lines.append("## 二、需重点关注（人工复查优先级高）")
        lines.append("| 文件 | 问题 |")
        lines.append("|------|------|")
        for f in attention:
            issues = []
            if f["ocr_fail"] > 0:
                issues.append(f"OCR校对{f['ocr_fail']}次失败")
            if f["vision_fail"] > 0:
                issues.append(f"Vision识别{f['vision_fail']}次失败")
            if f["status"] == "failed":
                issues.append("三引擎全部失败")
            lines.append(f"| {f['name']} | {' / '.join(issues)} |")
        lines.append("")

    # ── 待审核文件 ──
    if pending:
        lines.append("## 三、待人工审核文件（Web UI Tab3）")
        lines.append("| 文件 | 类别 | Vision识别页数 | Vision失败 |")
        lines.append("|------|------|--------------|------------|")
        for f in pending:
            vp = f.get("vision_pages", 0)
            vf = f["vision_fail"]
            lines.append(f"| {f['name']} | {f['category']} | {vp} | {'有' if vf else '无'} |")
        lines.append("")

    # ── 成功入库明细 ──
    lines.append("## 四、成功入库文件明细")
    lines.append("| 文件 | 方法 | 块数 | OCR修正 | OCR失败次数 | 备注 |")
    lines.append("|------|------|------|---------|------------|------|")
    for f in success:
        method = " + ".join(f["method"]) if f["method"] else "未知"
        if f["ocr_total"] > 0:
            ocr_info = f"{f['ocr_fixed']}/{f['ocr_total']}"
            if f["ocr_fail"]:
                ocr_info += f" ⚠️{f['ocr_fail']}失败"
        else:
            ocr_info = "—"
        notes = " / ".join(f["notes"]) if f["notes"] else "—"
        lines.append(f"| {f['name']} | {method} | {f['chunks']} | {ocr_info} | {f['ocr_fail'] or '—'} | {notes} |")
    lines.append("")

    # ── 失败文件 ──
    if failed:
        lines.append("## 五、失败文件")
        lines.append("| 文件 | 类别 |")
        lines.append("|------|------|")
        for f in failed:
            lines.append(f"| {f['name']} | {f['category']} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    if not log_path or not os.path.exists(log_path):
        print("找不到日志文件，请指定路径。")
        sys.exit(1)

    print(f"解析日志: {log_path}")
    files = parse_log(log_path)
    report = generate_report(files, log_path)

    out_name = "sync_report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
    out_path = os.path.join(LOG_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成: {out_path}")
    print()
    print(report)
