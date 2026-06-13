"""
Telegram 机器人 v11
复用同一知识库（ChromaDB + SQLite + BGE-M3 + doubao LLM）
"""
import os, re, json, time, logging, threading
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from concurrent.futures import ThreadPoolExecutor
import requests
from pathlib import Path

# ── 复用飞书机器人的核心组件 ──
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feishu_ws_server_v12 import (
    db, call_llm_engine, _SYSTEM_PROMPT, _SYSTEM_PROMPT_ACCIDENT, _ACCIDENT_LEGAL_QUERIES,
    _CROP_TAG_RE, BASE_DIR, KB_DIR, QA_LOG_DIR,
    _detect_dir_query, _list_kb_dir,
    write_qa_log,
    CURRENT_LLM, CURRENT_MODEL, _state_lock,
    _bm25_search, _rrf_merge, _rerank, _auto_scope_keywords,
)

# ── 配置 ──
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_API   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
MAX_IMAGES     = 5
MAX_WORKERS    = 10

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{BASE_DIR}/logs/telegram_{time.strftime('%Y%m%d')}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── 阈值解析（与飞书保持一致）──
_thresh_re = re.compile(r'阈值\s*([0-9]+\.?[0-9]*)')


# ════════════════════════════════════════════════════════
# Telegram 发送工具
# ════════════════════════════════════════════════════════

def tg_send_text(chat_id, text, reply_to=None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        res = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=30)
        if res.json().get("ok"):
            logger.info("📤 Telegram 文字回复成功")
        else:
            logger.error(f"sendMessage 失败: {res.text}")
    except Exception as e:
        logger.error(f"sendMessage 异常: {e}")


def tg_send_photo(chat_id, img_path, reply_to=None):
    abs_path = os.path.join(BASE_DIR, img_path) if not os.path.isabs(img_path) else img_path
    if not os.path.exists(abs_path):
        logger.warning(f"图片不存在: {abs_path}")
        return
    payload = {"chat_id": chat_id}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        with open(abs_path, "rb") as f:
            res = requests.post(f"{TELEGRAM_API}/sendPhoto",
                data=payload, files={"photo": f}, timeout=60)
        if not res.json().get("ok"):
            logger.error(f"sendPhoto 失败: {res.text}")
    except Exception as e:
        logger.error(f"sendPhoto 异常: {e}")



# ════════════════════════════════════════════════════════
# 核心处理逻辑
# ════════════════════════════════════════════════════════

def process_tg_message(chat_id, message_id, question):
    logger.info(f"💬 [TG] 收到问题: {question[:80]}")

    # ── /start ──
    if question.strip() in ("/start", "/help"):
        tg_send_text(chat_id, (
            "👋 EHS 智慧知识库机器人\n\n"
            "直接发问即可，支持：\n"
            "• EHS案例 / 公司内部制度 / 国家规定\n"
            "• 超精准匹配 / 精准匹配 / 宽松匹配\n"
            "• 开启分析（附引用来源）\n"
            "• /目录 — 查看知识库目录\n\n"
            "示例：高处作业安全要求是什么"
        ), reply_to=message_id)
        return

    # ── /目录 ──
    if question.strip().startswith("/目录") or question.strip() == "目录":
        hint = question.replace("/目录", "").strip()
        listing = _list_kb_dir(hint)
        tg_send_text(chat_id, listing, reply_to=message_id)
        return

    # ── 自然语言目录意图 ──
    is_dir, path_hint = _detect_dir_query(question)
    if is_dir:
        listing = _list_kb_dir(path_hint)
        tg_send_text(chat_id, listing, reply_to=message_id)
        return

    # ── 匹配精度 ──
    _tm = _thresh_re.search(question)
    if _tm:
        score_threshold = float(_tm.group(1))
    elif "超精准匹配" in question:
        score_threshold = 0.5
    elif "精准匹配" in question:
        score_threshold = 0.65
    elif "宽松匹配" in question:
        score_threshold = 1.1
    else:
        score_threshold = 1.0   # 默认与飞书一致

    # ── domain 路由 + scope（可叠加）──
    domain_keywords = {
        "负面案例": "EHS案例", "正面案例": "EHS案例", "案例库": "EHS案例",
        "警示手册": "EHS案例", "标准化手册": "EHS案例", "EHS案例": "EHS案例",
        "公司内部": "公司内部", "内部制度": "公司内部",
        "国家规定": "国家规定", "法律法规": "国家规定",
        "国家标准": "国家规定", "行业标准": "国家规定",
    }
    detected_domain = next((v for k, v in domain_keywords.items() if k in question), None)
    detected_scope  = next((kw for kw in _auto_scope_keywords if kw in question), None)

    # ── BM25 + 向量混合检索 ──
    chroma_res      = db.similarity_search_with_score(question, k=20)
    bm25_res        = _bm25_search(question, k=20)
    docs_and_scores = _rrf_merge(chroma_res, bm25_res) if bm25_res else chroma_res

    def _passes(doc):
        ok = True
        if detected_domain:
            ok = ok and (doc.metadata.get("domain") == detected_domain)
        if detected_scope:
            ok = ok and (detected_scope in doc.metadata.get("source", ""))
        return ok

    is_accident_mode = question.strip().startswith("真实案例分析")
    _k_match = re.search(r'\bK\s*=\s*(\d+)', question, re.IGNORECASE) if is_accident_mode else None
    accident_top_k = min(int(_k_match.group(1)), 40) if _k_match else 15
    top_k = accident_top_k if is_accident_mode else 10

    filtered       = [(d, s) for d, s in docs_and_scores if _passes(d) and s < score_threshold]
    valid_with_scores = _rerank(question, filtered, top_k=top_k)

    # ── 事故模式：追加国家规定法律条款检索 ──
    if is_accident_mode:
        import re as _re
        legal_queries = []
        for pattern, query in _ACCIDENT_LEGAL_QUERIES:
            if _re.search(pattern, question) and query not in legal_queries:
                legal_queries.append(query)
            if len(legal_queries) == 3:
                break
        if legal_queries:
            existing_keys = {
                (d.metadata.get("source"), d.metadata.get("chunk_seq"))
                for d, _ in valid_with_scores
            }
            extra_candidates = []
            for lq in legal_queries:
                for doc, score in db.similarity_search_with_score(lq, k=5, filter={"domain": "国家规定"}):
                    key = (doc.metadata.get("source"), doc.metadata.get("chunk_seq"))
                    if key not in existing_keys:
                        extra_candidates.append((doc, score))
                        existing_keys.add(key)
            if extra_candidates:
                extra_reranked = _rerank(question, extra_candidates, top_k=5)
                valid_with_scores = list(valid_with_scores) + extra_reranked
                logger.info(f"⚖️  法律扩展追加 {len(extra_reranked)} 块，合计 {len(valid_with_scores)} 块")

    valid_docs = [doc for doc, _ in valid_with_scores]
    logger.info(f"📚 命中 {len(valid_docs)} 条（domain={detected_domain} scope={detected_scope}）")

    if not valid_docs:
        tg_send_text(chat_id, "知识库中未找到符合条件的相关规定。可尝试宽松匹配或换个关键词。", reply_to=message_id)
        return

    # ── XML 结构化上下文 ──
    xml_parts = [
        f'<参考资料 序号="{i}" 来源="{d.metadata.get("source","未知")}" 生效时间="{d.metadata.get("date_str","未知")}">\n{d.page_content}\n</参考资料>'
        for i, (d, _) in enumerate(valid_with_scores, 1)
    ]
    context_str = "\n".join(xml_parts)

    ctx_crops = _CROP_TAG_RE.findall(context_str)
    if ctx_crops:
        img_instruction = (
            "部分参考资料中含有图片标记 [📷 路径]。"
            "仅当图片内容能直接佐证你对用户问题的回答时，才将对应的 [📷 路径] 标记插入到答案中；"
            "与问题无直接关联的图片一律忽略，不要引用。\n"
        )
    else:
        img_instruction = ""

    if is_accident_mode or "开启分析" in question:
        user_prompt = (
            f"参考资料如下：\n{context_str}\n\n"
            f"问题：{question}\n"
            f"{img_instruction}"
            f"请筛选与问题直接相关的参考资料，整合分析并给出结论。"
            f"与问题无关的参考资料不要引入答案。"
            f"在末尾列出实际引用的【引用来源和生效时间】清单。使用 Markdown 排版。"
        )
    else:
        user_prompt = (
            f"参考资料如下：\n{context_str}\n\n"
            f"问题：{question}\n"
            f"{img_instruction}"
            f"请先判断哪些参考资料与问题直接相关，只基于直接相关的部分作答，"
            f"不相关的内容不要引入答案。"
            f"直接相关的要点不要遗漏，整合为结构清晰的答案。"
            f"正文不出现【来源:】字样。使用 Markdown 排版。"
        )

    sys_prompt = _SYSTEM_PROMPT_ACCIDENT if is_accident_mode else _SYSTEM_PROMPT

    try:
        with _state_lock:
            llm_log = CURRENT_LLM
            model_log = CURRENT_MODEL
        logger.info(f"🚀 [LLM] {llm_log} ({model_log})")
        answer = call_llm_engine(sys_prompt, user_prompt)
    except Exception as e:
        answer = f"❌ 响应失败: {str(e)}"

    write_qa_log(question, docs_and_scores, valid_with_scores, user_prompt, answer)

    # ── 按 [📷] 标签交错发文字和图片（顺序正确）──
    parts = _CROP_TAG_RE.split(answer)
    img_sent = 0
    first = True
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                tg_send_text(chat_id, text, reply_to=message_id if first else None)
                first = False
        else:
            if img_sent < MAX_IMAGES:
                tg_send_photo(chat_id, part.strip(), reply_to=message_id if first else None)
                first = False
                img_sent += 1


# ════════════════════════════════════════════════════════
# 长轮询主循环
# ════════════════════════════════════════════════════════

def poll():
    offset = 0
    logger.info("🤖 Telegram 机器人启动！@qagent2026_bot")
    while True:
        try:
            res = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            ).json()
            for update in res.get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                if not msg:
                    continue
                chat_id    = msg["chat"]["id"]
                message_id = msg["message_id"]
                text = msg.get("text", "").strip()
                if not text:
                    continue
                # 过滤纯表情
                clean = re.sub(r'\[[^\]]*\]', '', text).strip()
                if len(clean) < 2:
                    continue
                _executor.submit(
                    process_tg_message,
                    chat_id, message_id, text,
                )
        except Exception as e:
            logger.error(f"getUpdates 异常: {e}")
            time.sleep(5)


if __name__ == "__main__":
    poll()
