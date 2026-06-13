import os, json, requests, threading, time, logging, re, glob
from collections import OrderedDict
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as _LCDoc  # 聚合用，保证全局可用
import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

# ================= 1. 核心路径与日志配置 =================
BASE_DIR = os.path.expanduser("~/doc_parser_v12")
DB_DIR   = f"{BASE_DIR}/chroma_db"
LOG_DIR    = f"{BASE_DIR}/logs"
QA_LOG_DIR = f"{BASE_DIR}/qa_logs"
KB_DIR     = f"{BASE_DIR}/Local_KB"

WHITELIST_FILE = f"{BASE_DIR}/authorized_users.json"
QA_RECORD_DIR  = f"{BASE_DIR}/qa_records"

os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(QA_LOG_DIR, exist_ok=True)
os.makedirs(QA_RECORD_DIR, exist_ok=True)
current_date = datetime.now().strftime('%Y-%m-%d')
log_file = os.path.join(LOG_DIR, f"feishu_ws_server_v11_{current_date}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= 2. 飞书通信配置 =================
FEISHU_APP_ID     = os.environ["FEISHU_APP_ID"]
FEISHU_APP_SECRET = os.environ["FEISHU_APP_SECRET"]

# ================= 3. API 密钥与模型配置 =================
# 火山引擎 Coding Plan（OpenAI兼容协议）
VOLCODING_API_KEY  = os.environ["VOLCODING_API_KEY"]
VOLCODING_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
VOLCODING_MODEL    = "doubao-seed-2.0-pro"   # 旗舰模型

# 本地 Ollama（断网可用）
LOCAL_LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"

# ================= 4. 动态状态机（默认火山旗舰）=================
CURRENT_LLM   = "cloud"       # cloud | local
CURRENT_MODEL = VOLCODING_MODEL

_token_cache = {"token": "", "expire": 0}
_token_lock = threading.Lock()
_state_lock = threading.Lock()

# ── 消息去重 + 并发上限 ──
# 飞书事件可能重投（网络抖动/超时重试），同一 message_id 只处理一次；
# CPU 上 embedding/rerank 多线程互相拖慢，限制同时处理 2 条。
_seen_msgs: OrderedDict = OrderedDict()
_seen_lock  = threading.Lock()
_worker_sem = threading.BoundedSemaphore(2)

def _is_duplicate_msg(msg_id: str) -> bool:
    with _seen_lock:
        if msg_id in _seen_msgs:
            return True
        _seen_msgs[msg_id] = time.time()
        while len(_seen_msgs) > 500:
            _seen_msgs.popitem(last=False)
        return False

logger.info("🔍 正在连接本地向量知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cpu'})
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
logger.info("✅ 知识库（ChromaDB）挂载完毕！")

# ── BM25 混合检索（后台线程构建索引）──────────────────────
try:
    import jieba
    from rank_bm25 import BM25Okapi
    _BM25_READY = True
except ImportError:
    _BM25_READY = False
    logger.warning("⚠️ rank-bm25 / jieba 未安装，BM25 混合检索不可用")

_bm25_index  = None
_bm25_corpus = []   # [(page_content, metadata), ...]
_bm25_lock   = threading.Lock()
BM25_PKL     = os.path.join(BASE_DIR, "bm25_index.pkl")
_bm25_pkl_mtime = 0  # 上次加载时的 mtime

def _save_bm25_pkl(index, corpus):
    """以 v2 格式（含已建好的索引对象）原子写入 pkl，下次启动免重新分词。"""
    global _bm25_pkl_mtime
    try:
        import pickle
        with open(BM25_PKL + ".tmp", "wb") as f:
            pickle.dump({"version": 2, "index": index, "corpus": corpus}, f)
        os.replace(BM25_PKL + ".tmp", BM25_PKL)
        _bm25_pkl_mtime = os.path.getmtime(BM25_PKL)
        logger.info("💾 BM25 索引已存盘（v2 格式，含索引对象）")
    except Exception as e:
        logger.warning(f"⚠️ BM25 pkl 存盘失败: {e}")

def _load_bm25_from_pkl() -> bool:
    """从 pkl 加载 BM25 索引，成功返回 True。
    v2 格式（dict，含索引对象）秒级加载；
    旧格式（list，仅语料，chunk_reviewer 等工具仍在写）则现场分词重建，并升级存盘为 v2。"""
    global _bm25_index, _bm25_corpus, _bm25_pkl_mtime
    if not _BM25_READY or not os.path.exists(BM25_PKL):
        return False
    try:
        import pickle
        mtime = os.path.getmtime(BM25_PKL)
        with open(BM25_PKL, "rb") as f:
            data = pickle.load(f)
        upgraded = False
        if isinstance(data, dict) and data.get("version") == 2:
            corpus    = data.get("corpus")
            new_index = data.get("index")
            if not corpus or new_index is None:
                return False
        elif isinstance(data, list) and data:
            corpus = data
            tokenized = [list(jieba.cut(item[0])) for item in corpus]
            new_index = BM25Okapi(tokenized)
            upgraded = True
        else:
            return False
        with _bm25_lock:
            _bm25_index  = new_index
            _bm25_corpus = corpus
            _bm25_pkl_mtime = mtime
        logger.info(f"✅ BM25 索引从 pkl 加载完成，共 {len(corpus)} 条")
        if upgraded:
            _save_bm25_pkl(new_index, corpus)
        return True
    except Exception as e:
        logger.error(f"⚠️ BM25 pkl 加载失败: {e}")
        return False

def _build_bm25_index():
    if not _BM25_READY:
        return
    global _bm25_index, _bm25_corpus
    # 优先从 pkl 加载
    if _load_bm25_from_pkl():
        return
    try:
        logger.info("🔨 构建 BM25 索引（从 ChromaDB）...")
        result = db.get(include=["documents", "metadatas"])
        docs   = result.get("documents", [])
        metas  = result.get("metadatas", [])
        if not docs:
            logger.info("⚠️ ChromaDB 为空，BM25 索引跳过")
            return
        tokenized = [list(jieba.cut(d)) for d in docs]
        new_index = BM25Okapi(tokenized)
        corpus    = list(zip(docs, metas))
        with _bm25_lock:
            _bm25_index  = new_index
            _bm25_corpus = corpus
        logger.info(f"✅ BM25 索引完成，共 {len(corpus)} 条")
        _save_bm25_pkl(new_index, corpus)
    except Exception as e:
        logger.error(f"⚠️ BM25 索引构建失败: {e}")

def _bm25_pkl_watcher():
    """后台监控 bm25_index.pkl 变化，热加载新索引。"""
    global _bm25_pkl_mtime
    while True:
        time.sleep(10)
        try:
            if not os.path.exists(BM25_PKL):
                continue
            mtime = os.path.getmtime(BM25_PKL)
            if mtime > _bm25_pkl_mtime:
                logger.info("🔄 检测到 bm25_index.pkl 变化，热加载中...")
                _load_bm25_from_pkl()
        except Exception as e:
            logger.error(f"⚠️ BM25 pkl watcher 异常: {e}")

threading.Thread(target=_build_bm25_index, daemon=True).start()
threading.Thread(target=_bm25_pkl_watcher, daemon=True).start()

def _bm25_search(question: str, k: int = 20, predicate=None) -> list:
    """BM25 检索，返回 [(doc, bm25_score), ...]。
    predicate(doc)->bool 可选：在取 top-k 前先按 domain/时间等过滤，
    保证定向提问时 BM25 一路的 k 个名额不被库外内容占掉。"""
    if not _BM25_READY:
        return []
    with _bm25_lock:
        if _bm25_index is None:
            return []
        tokens = list(jieba.cut(question))
        scores = _bm25_index.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0 or len(hits) >= k:
                break
            doc = _LCDoc(page_content=_bm25_corpus[i][0],
                         metadata=_bm25_corpus[i][1] or {})
            if predicate is not None and not predicate(doc):
                continue
            hits.append((doc, scores[i]))
        return hits

def _rrf_merge(chroma_res: list, bm25_res: list, rrf_k: int = 60) -> list:
    """Reciprocal Rank Fusion 合并两路检索结果，保留 chroma 原始得分。"""
    rrf_scores = {}
    doc_map     = {}
    chroma_scores = {}

    for rank, (doc, score) in enumerate(chroma_res):
        key = doc.metadata.get('source','') + '§' + doc.page_content[:80]
        rrf_scores[key]   = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key]       = doc
        chroma_scores[key] = score

    for rank, (doc, score) in enumerate(bm25_res):
        key = doc.metadata.get('source','') + '§' + doc.page_content[:80]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        if key not in doc_map:
            doc_map[key]       = doc
            chroma_scores[key] = 0.6   # BM25-only命中，给一个通过阈值的默认分

    merged = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    return [(doc_map[k], chroma_scores[k]) for k in merged]

# ── Reranker（二阶段精排）────────────────────────────────
try:
    from FlagEmbedding import FlagReranker
    _reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)  # CPU 下 fp16 反而更慢
    logger.info("✅ Reranker (bge-reranker-v2-m3) 加载完毕")
    _RERANKER_READY = True
except Exception as e:
    _reranker = None
    _RERANKER_READY = False
    logger.warning(f"⚠️ Reranker 未加载: {e}")

def _rerank(question: str, candidates: list, top_k: int = 10, min_score: float = 0.05) -> list:
    """用 cross-encoder 精排，返回 top_k 条 (doc, original_score)。
    min_score：rerank 归一化分（0~1）门槛，低于此分的候选视为弱相关直接淘汰，
    避免 top_k 永远填满导致无关内容硬塞进 prompt。"""
    if not _RERANKER_READY or not candidates:
        return candidates[:top_k]
    try:
        pairs  = [[question, doc.page_content] for doc, _ in candidates]
        scores = _reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):  # 单候选时 FlagReranker 返回标量
            scores = [scores]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        kept    = [(item, s) for item, s in ranked[:top_k] if s >= min_score]
        dropped = min(len(ranked), top_k) - len(kept)
        if dropped:
            logger.info(f"  🎯 Rerank 门槛({min_score})淘汰 {dropped} 条弱相关候选")
        return [item for item, _ in kept]
    except Exception as e:
        logger.warning(f"⚠️ Reranker 精排失败，降级: {e}")
        return candidates[:top_k]

# ── scope_keywords 从 Local_KB 目录自动提取 ──────────────
def _build_scope_keywords() -> list[str]:
    from pathlib import Path as _Path
    keywords = []
    kb = _Path(KB_DIR)
    if not kb.exists():
        return []
    for domain in kb.iterdir():
        if not domain.is_dir():
            continue
        for sub in domain.iterdir():
            name = re.sub(r'^\d+_', '', sub.name)
            if sub.is_dir() and len(name) >= 4:
                keywords.append(name)
                for third in sub.iterdir():
                    if third.is_dir():
                        n3 = re.sub(r'^\d+_', '', third.name)
                        if len(n3) >= 4:
                            keywords.append(n3)
    return list(set(keywords))

_auto_scope_keywords = _build_scope_keywords()
logger.info(f"📂 自动提取 scope 关键词 {len(_auto_scope_keywords)} 条")

def get_tenant_access_token():
    with _token_lock:
        if time.time() < _token_cache["expire"]: return _token_cache["token"]
        try:
            res = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
                timeout=15
            ).json()
            _token_cache["token"] = res.get("tenant_access_token", "")
            _token_cache["expire"] = time.time() + 7000
            logger.info("🔑 成功刷新飞书 Token")
        except Exception as e:
            logger.error(f"⚠️ Token 刷新失败: {e}")
        return _token_cache["token"]

def reply_feishu_message(message_id, text_content):
    token = get_tenant_access_token()
    if not token: return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"msg_type": "text", "content": json.dumps({"text": text_content})}
    try:
        requests.post(f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply", headers=headers, json=payload, timeout=30)
        logger.info("📤 回复飞书消息成功！")
    except Exception as e:
        logger.error(f"⚠️ 飞书推送异常: {e}")


def upload_image_to_feishu(img_path: str, token: str) -> str:
    """上传本地图片到飞书，返回 image_key；失败返回空字符串。"""
    abs_path = os.path.join(BASE_DIR, img_path) if not os.path.isabs(img_path) else img_path
    if not os.path.exists(abs_path):
        # 文件名可能带哈希后缀（如 _p4.png → _p4_923536.png），尝试 glob 匹配
        base, ext = os.path.splitext(abs_path)
        matches = sorted(glob.glob(f"{base}_*{ext}"))
        if matches:
            abs_path = matches[0]
            logger.info(f"🔍 图片路径 glob 匹配: {os.path.basename(abs_path)}")
        else:
            logger.warning(f"⚠️ 图片文件不存在: {abs_path}")
            return ""
    try:
        with open(abs_path, 'rb') as f:
            res = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/images",
                headers={"Authorization": f"Bearer {token}"},
                data={"image_type": "message"},
                files={"image": f},
                timeout=60
            ).json()
        if res.get("code") == 0:
            key = res["data"]["image_key"]
            logger.info(f"🖼️  图片上传成功: {key}")
            return key
        logger.warning(f"⚠️ 图片上传失败: {res}")
    except Exception as e:
        logger.error(f"⚠️ 图片上传异常: {e}")
    return ""


_CROP_TAG_RE = re.compile(r'\[📷\s*([^\]]+?)\s*\]')

def reply_feishu_with_images(message_id: str, text_content: str, image_paths: list, chat_id: str = None):
    """发消息卡片（Markdown 表格 + 内联图片），失败降级为纯文本+独立图片。"""
    token = get_tenant_access_token()
    if not token: return

    crop_paths = _CROP_TAG_RE.findall(text_content)
    all_paths = crop_paths + (image_paths or [])
    clean_text = _CROP_TAG_RE.sub('', text_content).strip()

    # 上传所有图片
    image_keys = []
    for path in all_paths:
        key = upload_image_to_feishu(path, token)
        if key:
            image_keys.append(key)

    # 优先发卡片（支持 Markdown 表格 + 内联图片）
    if not _send_card_reply(message_id, clean_text, image_keys, token):
        # 降级：纯文本 + 逐张发图
        reply_feishu_message(message_id, _convert_md_tables(clean_text))
        for key in image_keys:
            _send_image_message(message_id, key, token)


def _send_image_message(message_id: str, image_key: str, token: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"msg_type": "image", "content": json.dumps({"image_key": image_key})}
    try:
        res = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers=headers, json=payload, timeout=60
        )
        resp = res.json()
        if resp.get("code") != 0:
            logger.error(f"⚠️ 图片消息发送失败: {resp}")
    except Exception as e:
        logger.error(f"⚠️ 图片消息推送异常: {e}")


def _send_post_via_create(chat_id: str, reply_to_id: str, zh_content: list, token: str, img_count: int = 0) -> bool:
    """用发消息接口发 post，reply 接口不支持 post 类型。"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    post_content = {"post": {"zh_cn": {"content": zh_content}}}
    body = json.dumps({
        "receive_id": chat_id,
        "msg_type": "post",
        "content": json.dumps(post_content, ensure_ascii=False),
        "reply_in_thread_id": reply_to_id,
    }, ensure_ascii=False)
    try:
        res = requests.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            headers=headers, data=body.encode("utf-8"), timeout=30
        )
        resp = res.json()
        if resp.get("code") == 0:
            logger.info(f"📤 富文本发送成功（含{img_count}张图）")
            return True
        else:
            logger.warning(f"⚠️ 富文本发送失败 code={resp.get('code')} msg={resp.get('msg')}，降级处理")
            return False
    except Exception as e:
        logger.error(f"⚠️ 富文本发送异常: {e}")
        return False


def _send_post(message_id: str, zh_content: list, token: str, img_count: int = 0) -> bool:
    post_body = {"zh_cn": {"content": zh_content}}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"msg_type": "post", "content": json.dumps({"post": post_body}, ensure_ascii=False)}
    try:
        res = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers=headers, json=payload, timeout=30
        )
        resp = res.json()
        if resp.get("code") == 0:
            logger.info(f"📤 富文本回复成功（含{img_count}张图）")
            return True
        else:
            logger.warning(f"⚠️ 富文本回复失败 code={resp.get('code')}，降级处理")
            return False
    except Exception as e:
        logger.error(f"⚠️ 富文本推送异常: {e}")
        return False



def _send_card_reply(message_id: str, text_content: str, image_keys: list, token: str) -> bool:
    """用消息卡片回复（lark_md 支持 Markdown 表格，图片内联）。"""
    elements = []
    if text_content.strip():
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": text_content}
        })
    for key in image_keys:
        elements.append({
            "tag": "img",
            "img_key": key,
            "alt": {"tag": "plain_text", "content": "图片"}
        })
    if not elements:
        return False
    card = {"config": {"wide_screen_mode": True}, "elements": elements}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)}
    try:
        res = requests.post(
            f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
            headers=headers, json=payload, timeout=30
        )
        resp = res.json()
        if resp.get("code") == 0:
            logger.info(f"📤 卡片回复成功（含{len(image_keys)}张图）")
            return True
        logger.warning(f"⚠️ 卡片回复失败 code={resp.get('code')} msg={resp.get('msg')}，降级纯文本")
        return False
    except Exception as e:
        logger.error(f"⚠️ 卡片回复异常: {e}")
        return False

def _convert_md_tables(text: str) -> str:
    """将 Markdown 表格转为飞书可读的纯文本格式。"""
    table_re = re.compile(r'((?:\|[^\n]*\|\n?)+)', re.MULTILINE)
    def _to_text(m):
        lines = [l.strip() for l in m.group(0).strip().splitlines()]
        rows = []
        for l in lines:
            if re.match(r'^\|[-:\s|]+\|$', l):
                continue  # 跳过分隔行
            cells = [c.strip() for c in l.strip('|').split('|')]
            rows.append(cells)
        if not rows:
            return m.group(0)
        # 第一行作为表头，其余行用缩进列出
        header = '  '.join(rows[0])
        body = '\n'.join('  '.join(r) for r in rows[1:])
        return header + '\n' + body if body else header
    return table_re.sub(_to_text, text)


def clean_for_feishu(text: str) -> str:
    """清理 LLM 输出中影响飞书显示的符号（卡片模式保留 Markdown 表格）。"""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # 去掉 # 标题符
    text = re.sub(r'[\uf000-\uf8ff]', '', text)                  # 去掉 Wingdings 私有区字符
    return text.strip()


_SYSTEM_PROMPT = (
    "你是工程项目EHS安全管理专家助手。请严格基于提供的参考资料回答问题，"
    "禁止编造任何规定、数据或要求。"
    "如果参考资料中没有相关内容，请明确回答'未在知识库中找到相关规定'，不得凑答案。"
    "如果用户发送的是评论、感谢、闲聊、表扬或与EHS知识查询无关的内容（不是在提问），"
    "请只回复：'有EHS相关问题请直接提问 😊'，不要引用任何参考资料。"
)

_SYSTEM_PROMPT_ACCIDENT = (
    "你是工程项目EHS安全管理专家助手。用户提交的是一起真实事故案例，需要你基于知识库中"
    "现有的参考资料进行综合分析。请遵守以下规则：\n"
    "1. 充分利用参考资料中所有相关内容，尽力给出完整的责任界定与处置分析；\n"
    "2. 禁止编造法规条款编号或具体数字，但可以基于参考资料中涉及的原则和要求进行推断分析；\n"
    "3. 如果某一方面（如具体罚款金额、刑事条款）在参考资料中确实没有依据，"
    "请在该部分明确注明【知识库中暂无对应条款，建议参照安全生产法律法规处理】，"
    "而不是整体放弃回答；\n"
    "4. 在末尾列出【引用来源和生效时间】清单。使用 Markdown 排版。"
)

# 检索前从问题中剥离的控制词：这些是给系统看的开关，不是语义内容，
# 留在 query 里会污染 embedding / BM25 检索（LLM 收到的仍是原问题）。
# 注意：裸年份"2024年"不剥离（文档正文里也有年份，对 BM25 有用）。
_CONTROL_WORD_RE = re.compile(
    r'^真实案例分析'
    r'|开启分析'
    r'|超?精准匹配|普通匹配|宽松匹配'
    r'|阈值\s*[0-9]+\.?[0-9]*'
    r'|[Kk]\s*=\s*\d+'
    r'|近[0-9一二三四五六七八九十]{1,3}年'
    r'|\d{4}年(?:之后|以后|以来)'
)

_CN_NUM = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
           '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

def _parse_cn_number(s: str):
    """'3'→3，'三'→3，'十五'→15，'三十'→30；解析失败返回 None。"""
    if s.isdigit():
        return int(s)
    if s in _CN_NUM:
        return _CN_NUM[s]
    if len(s) == 2 and s[0] == '十' and s[1] in _CN_NUM:
        return 10 + _CN_NUM[s[1]]
    if len(s) == 2 and s[0] in _CN_NUM and s[1] == '十':
        return _CN_NUM[s[0]] * 10
    return None

# 事故类型 → 法律检索语句（仅在真实案例分析模式下使用）
_ACCIDENT_LEGAL_QUERIES = [
    ("动火|焊接|切割|明火",   "动火作业违规施工单位行政处罚责任追究"),
    ("火灾|燃烧|起火",        "火灾事故建设单位监理单位责任赔偿处罚"),
    ("死亡|坠亡|身亡|伤亡",   "生产安全事故死亡工伤认定赔偿"),
    ("高处|坠落|高空",        "高处坠落安全事故施工单位刑事行政责任"),
    ("触电|漏电|电气",        "触电事故施工用电违规责任追究处罚"),
    ("爆炸|起爆|燃爆",        "爆炸事故危险品违规施工责任处罚"),
    ("坍塌|垮塌|塌方",        "基坑坍塌施工安全事故责任追究"),
    ("窒息|中毒|有限空间",    "有限空间作业事故责任处罚工伤"),
]

def call_llm_engine(system_prompt: str, user_prompt: str):
    with _state_lock:
        llm = CURRENT_LLM
        model = CURRENT_MODEL

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if llm == "cloud":
                headers = {"Authorization": f"Bearer {VOLCODING_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 4096}
                res = requests.post(VOLCODING_BASE_URL, headers=headers, json=payload, timeout=300)
                if res.status_code == 429:
                    wait = 5 * (attempt + 1)
                    logger.warning(f"⚠️ 限流 429，等待 {wait}s 重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                if res.status_code >= 500:
                    logger.warning(f"⚠️ 服务端错误 {res.status_code}，重试 ({attempt+1}/{max_retries})")
                    time.sleep(3)
                    continue
                try:
                    data = res.json()
                except Exception:
                    raise Exception(f"火山引擎 API 返回非 JSON (HTTP {res.status_code}): {res.text[:200]}")
                res.raise_for_status()
                if "choices" not in data:
                    raise Exception(f"火山引擎 API 返回异常: {data.get('error', data.get('message', '未知'))}")
                return data["choices"][0]["message"]["content"]

            elif llm == "local":
                headers = {"Content-Type": "application/json"}
                payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 4096}
                res = requests.post(LOCAL_LLM_URL, headers=headers, json=payload, timeout=300)
                try:
                    data = res.json()
                except Exception:
                    raise Exception(f"本地模型返回非 JSON (HTTP {res.status_code}): {res.text[:200]}")
                if "choices" not in data:
                    raise Exception(f"本地模型返回异常: {data}")
                return data["choices"][0]["message"]["content"]

            return "⚠️ 未知的 LLM 配置！"

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"⚠️ 网络连接被远端断开 (尝试 {attempt + 1}/{max_retries})...")
            if attempt == max_retries - 1:
                raise Exception("远端服务器强制断开了连接，请稍后再试。")
            time.sleep(2)
        except Exception as e:
            raise e

# 目录查询意图检测：需要"目录类词"+"疑问类词"组合触发，避免误判内容问题
# 注意：触发词不含"分类/类别"——"重大危险源包含哪些类别"是内容问题，不是目录查询
_DIR_QUERY_RE = re.compile(
    r'(主目录|次级目录|二级目录|子目录|目录结构|文件夹列表|文件列表)'          # 明确目录词
    r'|'
    r'(有哪些|有什么|都有|包含|列出|给我看|查看).{0,15}(目录|文件夹)'          # 疑问+目录
    r'|'
    r'(目录|文件夹).{0,10}(有哪些|有什么|是什么|给我|列出|查看)'              # 目录+疑问
)


def _detect_dir_query(question: str):
    """检测目录查询意图，返回 (is_dir_query: bool, path_hint: str)"""
    if not _DIR_QUERY_RE.search(question):
        return False, ""

    from pathlib import Path
    kb = Path(KB_DIR)
    if not kb.exists():
        return True, ""

    # 扫描顶级域和二级目录，看问题里有没有提到
    for domain in sorted(kb.iterdir()):
        if not domain.is_dir():
            continue
        # 先检查二级目录（更具体，优先）
        for sub in sorted(domain.iterdir()):
            if sub.is_dir() and sub.name in question:
                return True, sub.name
        # 再检查域名本身
        if domain.name in question:
            return True, domain.name

    # 只有通用目录意图，没有具体路径 → 返回顶级
    return True, ""


def _list_kb_dir(path_hint: str = "") -> str:
    """
    扫描 Local_KB 返回目录清单。
    path_hint 为空   → 顶级三大类
    path_hint=域名   → 该域下的二级目录
    path_hint=二级名 → 该目录下的三级文件/文件夹
    """
    from pathlib import Path
    kb = Path(KB_DIR)

    def _clean(name: str) -> str:
        return re.sub(r'^\d{8}_', '', name)  # 去掉日期前缀

    path_hint = path_hint.strip()

    # ── 顶级 ──
    if not path_hint:
        domains = sorted(d.name for d in kb.iterdir() if d.is_dir())
        lines = ["📚 知识库包含以下三大类："]
        for d in domains:
            lines.append(f"• 📁 {d}")
        lines.append("\n发送「/目录 EHS案例」可查看该类下的目录，以此类推。")
        return "\n".join(lines)

    # ── 查找目标路径（先顶级，再在各域下找）──
    target = kb / path_hint
    if not (target.exists() and target.is_dir()):
        target = None
        for domain in sorted(kb.iterdir()):
            if not domain.is_dir():
                continue
            candidate = domain / path_hint
            if candidate.exists() and candidate.is_dir():
                target = candidate
                break
    if target is None:
        return f"⚠️ 未找到目录「{path_hint}」，请检查名称是否正确。"

    rel = target.relative_to(kb)
    depth = len(rel.parts)  # 1=域级，2=二级
    items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    dirs  = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file() and not i.name.startswith('.')]

    lines = [f"📁 {rel} 的内容："]

    if depth == 1:
        # 域级 → 只列二级目录
        if dirs:
            for d in dirs:
                lines.append(f"  📁 {_clean(d.name)}")
            lines.append(f"\n发送「/目录 {dirs[0].name}」可查看该目录下的文件。")
        else:
            lines.append("  （该类下无子目录，直接存放文件）")
            for f in files:
                lines.append(f"  📄 {_clean(f.stem)}")
    else:
        # 二级及以下 → 列文件和子文件夹
        if dirs:
            lines.append("📁 子目录：")
            for d in dirs:
                lines.append(f"  • {_clean(d.name)}")
        if files:
            lines.append("📄 文件：")
            for f in files:
                lines.append(f"  • {_clean(f.stem)}")
        if not dirs and not files:
            lines.append("  （空目录）")

    return "\n".join(lines)


def write_qa_log(question: str, all_candidates: list, valid_with_scores: list, prompt: str, answer: str):
    """每次问答后写详细日志，用于检查检索质量和 LLM 输入输出。"""
    ts = time.strftime('%Y%m%d_%H%M%S')
    safe_q = re.sub(r'[\\/:*?"<>|\s]', '_', question[:30]).strip('_')
    log_path = os.path.join(QA_LOG_DIR, f"{ts}_{safe_q}.log")

    lines = [
        "=" * 60,
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"问题: {question}",
        "",
        f"=== 全部候选（k=20，共 {len(all_candidates)} 条）===",
    ]
    for i, (doc, score) in enumerate(all_candidates, 1):
        src    = doc.metadata.get('source', '未知')
        date_s = doc.metadata.get('date_str', '未知')
        domain = doc.metadata.get('domain', '未知')
        lines += [
            "",
            f"[{i:02d}] 得分:{score:.4f}  domain:{domain}  日期:{date_s}",
            f"     来源: {src}",
            "--- 内容 ---",
            doc.page_content,
            "---",
        ]

    lines += [
        "",
        f"=== 命中块（阈值过滤后，共 {len(valid_with_scores)} 条）===",
    ]
    for i, (doc, score) in enumerate(valid_with_scores, 1):
        src    = doc.metadata.get('source', '未知')
        date_s = doc.metadata.get('date_str', '未知')
        lines += [
            "",
            f"【{i}】得分:{score:.4f}  日期:{date_s}",
            f"    来源: {src}",
            "--- 内容 ---",
            doc.page_content,
            "---",
        ]

    lines += [
        "",
        "=" * 60,
        "=== 提交给 LLM 的 Prompt ===",
        prompt,
        "",
        "=" * 60,
        "=== LLM 返回 ===",
        answer,
    ]

    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"📝 QA日志: {os.path.basename(log_path)}")
    except Exception as e:
        logger.warning(f"QA日志写入失败: {e}")


def answer_for(question):
    """平台无关核心：输入问题，返回 (答案文本, 图片路径列表)。
    命令与目录查询也在此处理，统一返回纯文本。具体平台的回复由调用方负责，
    使得飞书 / 企业微信等多入口能复用同一套检索+LLM 大脑。"""
    global CURRENT_LLM, CURRENT_MODEL
    cmd = question.strip()
    
    if cmd.startswith("/"):
        if cmd == "/说明" or cmd == "/模型":
            with _state_lock:
                status_msg  = "🤖 **智慧工地 EHS + 专业部门知识大脑**\n"
                status_msg += "⚠️ 思考上限5分钟，请排队提问 | `/说明` 查看此菜单\n"
                status_msg += "📂 **公开库**：EHS案例 / 公司内部 / 国家规定\n"
                status_msg += "🔒 **封闭库**（须在提问中明确写出部门名称才会检索）：\n"
                status_msg += "   客关部（排雷库、规定）/ 设计部（惊喜库、用心库、沧海桑田）\n"
                status_msg += "   工管部（工艺标准、案例集）/ 合约部（缺陷案例）\n"
                status_msg += "---\n"
                status_msg += "**📂 【技能1：锁定搜索范围】**\n"
                status_msg += "提问时带入以下关键词，系统会严格限定在对应范围内检索：\n"
                status_msg += "• 「**EHS案例**」→ 仅在 EHS 案例库中搜索\n"
                status_msg += "• 「**公司内部**」→ 仅在公司内部制度中搜索\n"
                status_msg += "• 「**国家规定**」→ 仅在国家法规标准库中搜索\n"
                status_msg += "• 「**客关部**」→ 仅在客关部封闭库中搜索（排雷库等）\n"
                status_msg += "• 「**设计部**」→ 仅在设计部封闭库中搜索（惊喜库/用心库/沧海桑田）\n"
                status_msg += "• 「**工管部**」→ 仅在工管部封闭库中搜索\n"
                status_msg += "• 「**合约部**」→ 仅在合约部封闭库中搜索\n"
                status_msg += "⚠️ 封闭库须写**完整部门名**（如「客关部」），写「客关」不会触发\n\n"
                status_msg += "**⏱️ 【技能2：时间过滤】**\n"
                status_msg += "提问中加入「近三年」「近五年」或具体年份（如2024年）\n\n"
                status_msg += "**📝 【技能3：引用来源 / 案例分析模式】**\n"
                status_msg += "• 默认 → 简洁答案\n"
                status_msg += "• 句尾加「**开启分析**」→ 附带引用来源和生效时间清单\n"
                status_msg += "• 问题**最前面**加「**真实案例分析**」→ 案例综合分析模式\n"
                status_msg += "  └ 自动追加国家法规责任条款检索，知识库缺失的条款会单独注明\n"
                status_msg += "  └ 默认返回 15 条参考资料；加「**K=30**」可提升至最多 40 条\n"
                status_msg += "  ⚠️ 注意：「真实案例分析」**必须写在问题最开头**\n\n"
                status_msg += "**🎚️ 【技能4：匹配精度控制】**\n"
                status_msg += "提问中加入以下词语，可控制搜索严格程度（不写默认普通匹配）：\n"
                status_msg += "• `超精准匹配` — 仅返回高度匹配内容；找不到直接告知，不凑答案\n"
                status_msg += "• `精准匹配` — 较严格过滤，适合查具体条款\n"
                status_msg += "• `普通匹配` — 默认模式，平衡覆盖与精度\n"
                status_msg += "• `宽松匹配` — 扩大搜索范围，适合概念性问题\n"
                status_msg += "• `阈值0.7` — 自定义阈值（建议范围 **0.3 ～ 1.2**）\n"
                status_msg += "  └ 阈值越小 → 要求越严 → 返回内容越少但越精准\n"
                status_msg += "  └ 阈值越大 → 搜索越宽 → 覆盖更多但可能引入相关性较弱的内容\n\n"
                status_msg += "---\n"
                status_msg += "**💡 提问示例**：\n"
                status_msg += "✅ 「高处坠落事故有哪些典型原因？负面案例」\n"
                status_msg += "✅ 「应知应会手册中1+N+4+N体系是什么意思」\n"
                status_msg += "✅ 「管理十条：项目总经理对安全事故需承担哪些责任。开启分析」\n"
                status_msg += "✅ 「脚手架搭设有哪些强制性要求 国家规定 精准匹配」\n"
                status_msg += "✅ 「客关部 排雷库中有哪些泳池相关注意事项」\n"
                status_msg += "✅ 「设计部 惊喜库有多少案例」\n"
                status_msg += "✅ 「近三年关于承包商安全管理有哪些新要求？宽松匹配 开启分析」\n"
                status_msg += "✅ 「真实案例分析 ...案例描述文字...该如何界定责任？」"

            return (status_msg, [])

        # ── 隐藏暗号：切换模型 ──
        elif cmd == "/唤醒深度":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "cloud", "deepseek-v3.2"
            return ("🤫 已切换至火山引擎 DeepSeek-V3.2。", [])
        elif cmd == "/唤醒火山":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "cloud", VOLCODING_MODEL
            return (f"🤫 已切换至火山引擎旗舰 ({VOLCODING_MODEL})。", [])
        elif cmd == "/唤醒v4":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "cloud", "ark-code-latest"  # DeepSeek-V4-Pro-Beta
            return ("🤫 已切换至 DeepSeek-V4-Pro（尝鲜版，遇限流请切回 /唤醒火山）。", [])
        elif cmd == "/唤醒深海":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "local", "deepseek-r1:32b"
            return ("🤫 已切断云端，本地 DeepSeek-R1 32B 接管！", [])
        elif cmd == "/唤醒千问":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "local", "qwen2.5:32b"
            return ("🤫 已切断云端，本地 Qwen2.5 32B 接管！", [])
        elif cmd == "/唤醒极速":
            with _state_lock:
                CURRENT_LLM, CURRENT_MODEL = "local", "qwen2.5:14b"
            return ("🤫 已切断云端，本地 Qwen2.5 14B 接管！", [])
        elif cmd.startswith("/目录"):
            path_hint = cmd[len("/目录"):].strip()
            return (_list_kb_dir(path_hint), [])
        else:
            return ("⚠️ 未知指令。请输入 `/说明` 查看可用指令。", [])

    # ── 目录查询意图：自然语言模糊匹配，不走 RAG ──
    _is_dir, _path = _detect_dir_query(question)
    if _is_dir:
        logger.info(f"📂 检测到目录查询意图，路径提示: [{_path or '顶级'}]")
        return (_list_kb_dir(_path), [])

    # ====== 混合检索流程：ChromaDB 向量 + BM25 关键词 ======
    logger.info(f"🧠 [RAG] 开始检索知识库，关键词长度: {len(question)} 字符")

    # ── 时间过滤（动态年份识别，向量路下推 where，BM25 路在 _passes_filter 里检查）──
    time_cutoff = None
    _year_re = re.compile(r'(近([0-9一二三四五六七八九十]{1,3})年|(\d{4})年(?:之后|以后|以来)?)')
    _ym = _year_re.search(question)
    if _ym:
        if _ym.group(2):  # 近N年（支持中文数字：近三年/近十五年）
            n_years = _parse_cn_number(_ym.group(2))
            if n_years:
                time_cutoff = int(time.time()) - n_years * 365 * 24 * 3600
                logger.info(f"⏳ 时间过滤: 近{n_years}年 (timestamp > {time_cutoff})")
        elif _ym.group(3):  # 具体年份
            year = int(_ym.group(3))
            import calendar
            time_cutoff = int(calendar.timegm((year, 1, 1, 0, 0, 0, 0, 0, 0)))
            logger.info(f"⏳ 时间过滤: {year}年以后 (timestamp > {time_cutoff})")

    # ── domain 级路由 ──
    domain_keywords = {
        # EHS案例
        "EHS案例": "EHS案例", "负面案例": "EHS案例", "正面案例": "EHS案例",
        "案例库": "EHS案例", "警示手册": "EHS案例", "标准化手册": "EHS案例",
        "事故案例": "EHS案例",
        # 公司内部
        "公司内部": "公司内部", "内部制度": "公司内部", "内部规定": "公司内部",
        "公司规定": "公司内部", "公司制度": "公司内部", "内部文件": "公司内部",
        "公司文件": "公司内部",
        # 国家规定
        "国家规定": "国家规定", "法律法规": "国家规定", "国家标准": "国家规定",
        "行业标准": "国家规定", "国家法规": "国家规定", "法规标准": "国家规定",
        # 设计部
        "设计部": "设计部",
        # 客关部
        "客关部": "客关部",
        # 工管部
        "工管部": "工管部",
        # 合约部
        "合约部": "合约部",
    }
    # 封闭域：必须显式提到才搜索，通用问题不搜
    EXCLUSIVE_DOMAINS = {"设计部", "客关部", "工管部", "合约部"}

    # ── scope 关键词：硬编码 + 从目录自动提取 ──
    _hardcoded_scope = [
        "工程项目开发业务EHS管理指引", "工程项目EHS事故事件管理细则",
        "EHS年度考核实施细则", "承包商安全管理细则",
        "安全风险分级管控细则", "事故隐患排查治理细则",
        "项目总经理EHS管理十条", "管理十条",
        "项目负责人岗位实践", "应知应会手册", "安全治本攻坚",
    ]
    scope_keywords = list(set(_hardcoded_scope + _auto_scope_keywords))

    is_accident_mode = question.strip().startswith("真实案例分析")
    _k_match = re.search(r'\bK\s*=\s*(\d+)', question, re.IGNORECASE) if is_accident_mode else None
    accident_top_k = min(int(_k_match.group(1)), 40) if _k_match else 15

    detected_domains = set()
    for kw, dm in domain_keywords.items():
        if kw in question:
            detected_domains.add(dm)

    # domain 和 scope 可叠加
    detected_scope = None
    for kw in scope_keywords:
        if kw in question:
            detected_scope = kw
            break

    # ── 匹配精度档位 ──
    _thresh_re = re.compile(r'阈值\s*([0-9]+\.?[0-9]*)')
    _tm = _thresh_re.search(question)
    if _tm:
        score_threshold = float(_tm.group(1))
        use_fallback = True
        match_mode = f"自定义阈值 {score_threshold}"
    elif "超精准匹配" in question:
        score_threshold = 0.5
        use_fallback = False
        match_mode = "超精准匹配"
    elif "精准匹配" in question:
        score_threshold = 0.65
        use_fallback = True
        match_mode = "精准匹配"
    elif "宽松匹配" in question:
        score_threshold = 1.1
        use_fallback = True
        match_mode = "宽松匹配"
    else:
        score_threshold = 1.0   # 默认从 0.8 放宽到 1.0
        use_fallback = True
        match_mode = "普通匹配"
    logger.info(f"🎚️  匹配模式: {match_mode} (阈值 < {score_threshold})")

    # ── 检索 query：剥离控制词，只留语义内容（LLM 收到的仍是原问题）──
    retrieval_query = _CONTROL_WORD_RE.sub(' ', question).strip() or question

    # ── Rerank 分数门槛（cross-encoder 归一化分 0~1），与距离阈值档位联动 ──
    if "超精准匹配" in question:
        rerank_floor = 0.35
    elif "精准匹配" in question:
        rerank_floor = 0.12
    elif "宽松匹配" in question:
        rerank_floor = 0.0
    else:
        rerank_floor = 0.05

    # ── domain / scope / 时间 统一过滤器（BM25 路谓词 + 合并后复核）──
    def _passes_filter(doc):
        meta = doc.metadata or {}
        if time_cutoff and meta.get('timestamp', 0) <= time_cutoff:
            return False
        doc_domain = meta.get('domain', '')
        if detected_domains:
            if doc_domain not in detected_domains:
                return False
        else:
            if doc_domain in EXCLUSIVE_DOMAINS:
                return False
        if detected_scope:
            if detected_scope not in meta.get('source', ''):
                return False
        return True

    # ── Chroma where：时间 + domain 下推到向量检索 ──
    # 库中 7 成是国家规定，封闭库占比极小；不下推的话定向提问的 top-k 名额
    # 会被其他 domain 占满，过滤后所剩无几。
    # 注意：只用 $in，不用 $nin —— 当前 chromadb 版本对本库执行 $nin 会报
    # "Error executing plan: Error finding id"；未点名封闭库的通用提问
    # 仍走全局检索 + _passes_filter 合并后排除封闭域（与旧行为一致）。
    _conds = []
    if time_cutoff:
        _conds.append({"timestamp": {"$gt": time_cutoff}})
    if detected_domains:
        _conds.append({"domain": {"$in": sorted(detected_domains)}})
    chroma_where = None
    if _conds:
        chroma_where = _conds[0] if len(_conds) == 1 else {"$and": _conds}

    if detected_domains or detected_scope:
        tag = f"domain={detected_domains}" if detected_domains else ""
        tag += f" scope={detected_scope}" if detected_scope else ""
        logger.info(f"🎯 定向过滤: [{tag.strip()}]")

    # ── 向量检索 + BM25 混合（scope 是检索后过滤，命中 scope 时加大候选量）──
    retrieval_k = 30 if detected_scope else 20
    chroma_res = db.similarity_search_with_score(retrieval_query, k=retrieval_k, filter=chroma_where)
    bm25_res   = _bm25_search(retrieval_query, k=retrieval_k, predicate=_passes_filter)
    docs_and_scores = _rrf_merge(chroma_res, bm25_res) if bm25_res else chroma_res
    logger.info(f"🔀 RRF 合并后候选: {len(docs_and_scores)} 条")

    filtered = [(doc, score) for doc, score in docs_and_scores
                if _passes_filter(doc) and score < score_threshold]

    for doc, score in chroma_res[:5]:
        logger.info(f"  📊 向量得分: {score:.4f} | {doc.metadata.get('source','未知')}")

    # ── Reranker 精排 → top8（普通模式）/ top15（事故模式）──
    top_k = accident_top_k if is_accident_mode else 8
    reranked = _rerank(retrieval_query, filtered, top_k=top_k, min_score=rerank_floor)
    valid_with_scores = reranked

    if not valid_with_scores and use_fallback:
        # 兜底：放宽距离阈值，但仍保持封闭域/时间过滤，rerank 门槛不放
        all_filtered = [(doc, score) for doc, score in docs_and_scores if _passes_filter(doc)]
        valid_with_scores = _rerank(retrieval_query, all_filtered, top_k=top_k, min_score=rerank_floor)

    if not valid_with_scores and docs_and_scores:
        logger.info(f"  ℹ️  阈值过滤后为空，最近得分: {docs_and_scores[0][1]:.4f}，不兜底")

    # ── 事故模式：追加国家规定法律条款检索 ──
    if is_accident_mode:
        legal_queries = []
        for pattern, query in _ACCIDENT_LEGAL_QUERIES:
            if re.search(pattern, question) and query not in legal_queries:
                legal_queries.append(query)
            if len(legal_queries) == 3:
                break
        if legal_queries:
            logger.info(f"⚖️  事故模式法律扩展检索: {legal_queries}")
            existing_keys = {
                (d.metadata.get("source"), d.metadata.get("chunk_seq"))
                for d, _ in valid_with_scores
            }
            extra_candidates = []
            for lq in legal_queries:
                extra = db.similarity_search_with_score(
                    lq, k=5, filter={"domain": "国家规定"}
                )
                for doc, score in extra:
                    key = (doc.metadata.get("source"), doc.metadata.get("chunk_seq"))
                    if key not in existing_keys:
                        extra_candidates.append((doc, score))
                        existing_keys.add(key)
            if extra_candidates:
                # 法律条款是按事故类型模板查的，与原问题字面差异大，不设 rerank 门槛只排序
                extra_reranked = _rerank(retrieval_query, extra_candidates, top_k=5, min_score=0.0)
                valid_with_scores = list(valid_with_scores) + extra_reranked
                logger.info(f"⚖️  法律扩展追加 {len(extra_reranked)} 块，合计 {len(valid_with_scores)} 块")

    # ── 索引清单聚合：命中 is_index 文件任一块 → 整份按 chunk_seq 带出 ──
    index_srcs = {}
    for doc, score in valid_with_scores:
        if doc.metadata.get('is_index'):
            src = doc.metadata.get('source')
            if src not in index_srcs:
                index_srcs[src] = score
    if index_srcs:
        kept = [(d, s) for d, s in valid_with_scores if not d.metadata.get('is_index')]
        aggregated = []
        for src, score in index_srcs.items():
            full = db.get(where={'source': src}, include=['documents', 'metadatas'])
            blocks = sorted(zip(full['documents'], full['metadatas']),
                            key=lambda x: x[1].get('chunk_seq', 0))
            for content, meta in blocks:
                aggregated.append((_LCDoc(page_content=content, metadata=meta), score))
        # 索引整份置前（含总览+各分项），其余命中块跟后
        valid_with_scores = aggregated + kept
        logger.info(f"📑 索引聚合: {list(index_srcs)} → 整份带出 {len(aggregated)} 块，合计 {len(valid_with_scores)} 块")
    has_index_aggregation = bool(index_srcs)

    valid_docs = [doc for doc, _ in valid_with_scores]

    if not valid_docs:
        return ("知识库中未找到符合条件的相关规定。您可以尝试放宽时间或目录限制，或回复 `/说明` 查看支持的检索功能。", [])

    logger.info(f"📚 [RAG] 最终命中 {len(valid_docs)} 条，准备请求 LLM...")

    # ── XML 结构化上下文 ──
    xml_parts = []
    for i, (d, score) in enumerate(valid_with_scores, 1):
        src  = d.metadata.get('source', '未知')
        date = d.metadata.get('date_str', '未知')
        xml_parts.append(
            f'<参考资料 序号="{i}" 来源="{src}" 生效时间="{date}">\n{d.page_content}\n</参考资料>'
        )
    chroma_ctx = "\n".join(xml_parts)

    context_str = chroma_ctx

    ctx_crops = _CROP_TAG_RE.findall(context_str)
    if ctx_crops:
        img_instruction = (
            "部分参考资料中含有图片标记 [📷 路径]。"
            "仅当图片内容能直接佐证你对用户问题的回答时，才将对应的 [📷 路径] 标记插入到答案中；"
            "与问题无直接关联的图片一律忽略，不要引用。"
            "若多个来源（如不同版本）的图片内容高度相似，只保留生效时间最新的一张，避免重复引用。\n"
        )
    else:
        img_instruction = ""

    index_hint = (
        '【重要】参考资料中含有本库的索引/清单文件（标题含"总览""清单""介绍"等），'
        '其中列出了各系列/版本的案例或条目明细。'
        '回答数量或枚举类问题时，请从该索引的明细列表中统计，'
        '不要仅凭正文案例卡片片段来估算总数。\n'
    ) if has_index_aggregation else ""

    if is_accident_mode or "开启分析" in question:
        user_prompt = (
            f"参考资料如下：\n{context_str}\n\n"
            f"问题：{question}\n"
            f"{index_hint}"
            f"{img_instruction}"
            f"请筛选与问题直接相关的参考资料，整合分析并给出结论。"
            f"与问题无关的参考资料不要引入答案。"
            f"在末尾列出实际引用的【引用来源和生效时间】清单。"
            f"使用 Markdown 格式排版。"
        )
    else:
        user_prompt = (
            f"参考资料如下：\n{context_str}\n\n"
            f"问题：{question}\n"
            f"{index_hint}"
            f"{img_instruction}"
            f"请先判断哪些参考资料与问题直接相关，只基于直接相关的部分作答，"
            f"不相关的内容不要引入答案。"
            f"直接相关的要点不要遗漏，整合为结构清晰的答案。"
            f"正文中不要出现【来源:】字样。"
            f"使用 Markdown 格式排版。"
        )

    sys_prompt = _SYSTEM_PROMPT_ACCIDENT if is_accident_mode else _SYSTEM_PROMPT

    try:
        with _state_lock:
            current_llm_log = CURRENT_LLM
            current_model_log = CURRENT_MODEL
        logger.info(f"🚀 [LLM] 请求发送至: {current_llm_log} ({current_model_log})")

        answer = call_llm_engine(sys_prompt, user_prompt)
    except Exception as e:
        answer = f"❌ 响应失败: {str(e)}"

    answer = clean_for_feishu(answer)
    write_qa_log(question, docs_and_scores, valid_with_scores, user_prompt, answer)
    return (answer, [])

# ================= 私聊白名单 + 用户名查询 + 全量问答记录 =================
_wl = {"users": set(), "open": False, "mtime": -1}
_wl_lock = threading.Lock()

def _load_whitelist():
    with _wl_lock:
        try:
            if not os.path.exists(WHITELIST_FILE):
                _wl["users"], _wl["open"], _wl["mtime"] = set(), False, 0
                return
            mtime = os.path.getmtime(WHITELIST_FILE)
            if mtime == _wl["mtime"]:
                return
            with open(WHITELIST_FILE, encoding="utf-8") as f:
                data = json.load(f)
            _wl["users"] = set(data.get("feishu_authorized_open_ids", []))
            _wl["open"]  = bool(data.get("allow_all", False))   # 应急临时全放行
            _wl["mtime"] = mtime
            logger.info(f"🔐 飞书白名单已加载: {len(_wl['users'])} 人 allow_all={_wl['open']}")
        except Exception as e:
            logger.error(f"⚠️ 白名单加载失败，按全部拒绝处理: {e}")
            _wl["users"], _wl["open"] = set(), False

def is_authorized(open_id: str) -> bool:
    _load_whitelist()
    with _wl_lock:
        return _wl["open"] or open_id in _wl["users"]

_name_cache = {}
_name_lock = threading.Lock()

def get_user_name(open_id: str) -> str:
    """用 open_id 查飞书昵称（需通讯录权限 contact:user.base:readonly）；查不到回退 open_id。"""
    if not open_id:
        return ""
    with _name_lock:
        if open_id in _name_cache:
            return _name_cache[open_id]
    name = ""
    token = get_tenant_access_token()
    if token:
        try:
            r = requests.get(f"https://open.feishu.cn/open-apis/contact/v3/users/{open_id}",
                             headers={"Authorization": f"Bearer {token}"},
                             params={"user_id_type": "open_id"}, timeout=10).json()
            if r.get("code") == 0:
                name = r.get("data", {}).get("user", {}).get("name", "")
            else:
                logger.info(f"  查询用户名返回 code={r.get('code')}（可能未开通通讯录权限）")
        except Exception as e:
            logger.warning(f"⚠️ 查询用户名异常: {e}")
    name = name or open_id
    with _name_lock:
        _name_cache[open_id] = name
    return name

def log_qa_record(open_id: str, name: str, question: str, answer: str, authorized: bool, chat_type: str = ""):
    """全量问答审计：每条一行 JSON（含 id/昵称/时间/场景/问题/答案/是否授权），按月分文件，便于导出。"""
    rec = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "open_id": open_id,
        "name": name,
        "chat": chat_type,          # p2p=私聊  group=群
        "authorized": authorized,
        "question": question,
        "answer": answer,
    }
    path = os.path.join(QA_RECORD_DIR, f"qa_records_{datetime.now().strftime('%Y-%m')}.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"⚠️ 问答记录写入失败: {e}")


def _safe_process(message_id, question, open_id, chat_id=None, chat_type=""):
    """飞书线程入口（私聊 + 内部群）：白名单鉴权 → answer_for → 回复，并全量记录问答。"""
    with _worker_sem:
        name = get_user_name(open_id)
        # ── 授权白名单：仅名单内 open_id 放行，否则拒答并记录（allow_all=true 时全放行）──
        if not is_authorized(open_id):
            logger.warning(f"⛔ 未授权 open_id={open_id} ({name}) 提问被拒：{question[:40]}")
            log_qa_record(open_id, name, question, "[未授权，已拒绝]", False, chat_type)
            try:
                reply_feishu_message(message_id,
                    f"🔒 您暂无权限使用本知识库，请联系管理员开通。\n（您的用户ID：{open_id}，请提供给管理员）")
            except Exception:
                pass
            return
        try:
            text, images = answer_for(question)
            log_qa_record(open_id, name, question, text, True, chat_type)
            reply_feishu_with_images(message_id, text, images, chat_id=chat_id)
        except Exception:
            logger.exception("❌ answer_for 未捕获异常")
            log_qa_record(open_id, name, question, "[系统异常]", True, chat_type)
            try:
                reply_feishu_message(message_id, "❌ 系统处理异常，请稍后重试；若持续出现请联系管理员查看日志。")
            except Exception:
                pass


def _extract_post_text(content_json: str) -> str:
    """从飞书 post 类型消息中提取纯文本。"""
    try:
        data = json.loads(content_json)
        post = data.get("post", {})
        body = post.get("zh_cn") or next(iter(post.values()), {})
        parts = []
        for line in body.get("content", []):
            for elem in line:
                if elem.get("tag") == "text":
                    parts.append(elem.get("text", ""))
        return " ".join(parts)
    except Exception:
        return ""


def handle_feishu_message(data: P2ImMessageReceiveV1) -> None:
    event = data.event
    msg_type = event.message.message_type
    chat_type = getattr(event.message, "chat_type", "") or ""
    logger.info(f"📨 收到飞书消息 type={msg_type} chat={chat_type} id={event.message.message_id}")

    if _is_duplicate_msg(event.message.message_id):
        logger.info(f"  ↩️ 重复事件，已处理过，忽略: {event.message.message_id}")
        return

    # ── 私聊 + 内部群都答（群内外泄已由「内部群 + 禁止加到外部群」从结构上堵住）──
    # 群里只有 @ 机器人的消息才会被推送（im:message.group_at_msg:readonly），无需再判 @。
    # 发消息人的 open_id（白名单鉴权 + 全量记录用）
    try:
        open_id = event.sender.sender_id.open_id or ""
    except Exception:
        open_id = ""

    if msg_type == "text":
        user_question = json.loads(event.message.content).get("text", "")
    elif msg_type == "post":
        user_question = _extract_post_text(event.message.content)
    else:
        logger.info(f"  忽略不支持的消息类型: {msg_type}")
        return

    user_question = re.sub(r'@\S+\s*', '', user_question).strip().replace("**", "")
    # 过滤：空、纯表情（[赞][微笑]等）、有效字符不足3个
    user_question_clean = re.sub(r'\[[^\]]*\]', '', user_question).strip()
    if not user_question or not user_question_clean or len(user_question_clean) < 3:
        logger.info(f"  消息被过滤（空/表情/过短）: {repr(user_question)}")
        return
    logger.info(f"  处理问题: {user_question_clean[:80]}")
    threading.Thread(target=_safe_process,
                     args=(event.message.message_id, user_question, open_id,
                           event.message.chat_id, chat_type)).start()

if __name__ == "__main__":
    event_handler = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(handle_feishu_message).build()
    client = lark.ws.Client(FEISHU_APP_ID, FEISHU_APP_SECRET, event_handler=event_handler)
    logger.info("🎉 飞书 V11 机器人服务启动！(混合检索: ChromaDB + BM25)")
    client.start()
