"""
企业微信自建应用入口（v12）
====================================
设计要点：
  • 复用 feishu_ws_server_v12.answer_for —— 同一套混合检索 + Rerank + 豆包大脑，
    只在 I/O 层换成企业微信的回调/推送，不重复加载模型（只跑本服务时不连飞书）。
  • 回调 5 秒限制：收到消息立即 ACK 空串，后台线程算完答案再用
    「发送应用消息」API 主动推回给该成员（自建应用可随时推，无 48h 窗口限制）。
  • 授权白名单：authorized_users.json 里的 userid 才放行，其余直接拒答 + 记日志，
    防止未授权人员套取知识库内容。改名单热加载，无需重启。

凭据未配齐时本服务可正常导入/启动并跑自检，但收发会被跳过 —— 等你回去
在企业微信管理后台建好自建应用，拿到 corpid/agentid/secret/token/aeskey
填入 .env 即可。

启动：  venv/bin/python wecom_server.py     （默认 0.0.0.0:8090）
回调URL：https://<你的域名>/wecom/callback   （经云服务器 frp/反代转发到本机）
"""
import os, json, time, struct, socket, base64, hashlib, logging, threading, requests
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, Request, Response
import uvicorn
from Crypto.Cipher import AES

# 复用飞书服务里的平台无关大脑（导入即完成 embedding/BM25/reranker 初始化，但不启动飞书 WS）
import feishu_ws_server_v12 as brain

# ================= 配置 =================
BASE_DIR   = os.path.expanduser("~/doc_parser_v12")
LOG_DIR    = f"{BASE_DIR}/logs"
WHITELIST  = f"{BASE_DIR}/authorized_users.json"
os.makedirs(LOG_DIR, exist_ok=True)

_date = datetime.now().strftime('%Y-%m-%d')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, f"wecom_server_{_date}.log"), encoding='utf-8'),
              logging.StreamHandler()],
)
logger = logging.getLogger("wecom")

CORPID   = os.environ.get("WECOM_CORPID", "")
AGENTID  = os.environ.get("WECOM_AGENTID", "")
SECRET   = os.environ.get("WECOM_SECRET", "")
TOKEN    = os.environ.get("WECOM_TOKEN", "")
AES_KEY  = os.environ.get("WECOM_AES_KEY", "")   # EncodingAESKey，43 位
PORT     = int(os.environ.get("WECOM_PORT", "8090"))

_CONFIGURED = all([CORPID, AGENTID, SECRET, TOKEN, AES_KEY])
if not _CONFIGURED:
    logger.warning("⚠️ 企业微信凭据未配齐(.env 的 WECOM_*)，服务可启动但收发会跳过。"
                   "回去建好自建应用拿到凭据后填入 .env 即可。")

# ================= 企业微信消息加解密（WXBizMsgCrypt 等价实现）=================
class WeComCrypto:
    """AES-256-CBC + PKCS7，密文结构: random(16) | msglen(4,BE) | msg | receiveid。"""
    def __init__(self, token: str, encoding_aes_key: str, receive_id: str):
        self.token = token
        self.key = base64.b64decode(encoding_aes_key + "=") if encoding_aes_key else b""
        self.receive_id = receive_id

    def _sign(self, timestamp: str, nonce: str, encrypt: str) -> str:
        arr = sorted([self.token, timestamp, nonce, encrypt])
        return hashlib.sha1("".join(arr).encode()).hexdigest()

    def verify_sign(self, msg_signature: str, timestamp: str, nonce: str, encrypt: str) -> bool:
        return self._sign(timestamp, nonce, encrypt) == msg_signature

    def decrypt(self, encrypt_b64: str) -> str:
        cipher = AES.new(self.key, AES.MODE_CBC, self.key[:16])
        plain = cipher.decrypt(base64.b64decode(encrypt_b64))
        pad = plain[-1]
        plain = plain[:-pad]                       # 去 PKCS7 填充
        msg_len = struct.unpack(">I", plain[16:20])[0]
        msg = plain[20:20 + msg_len].decode("utf-8")
        recv = plain[20 + msg_len:].decode("utf-8")
        if self.receive_id and recv != self.receive_id:
            raise ValueError(f"receiveid 校验失败: {recv!r} != {self.receive_id!r}")
        return msg

_crypto = WeComCrypto(TOKEN, AES_KEY, CORPID) if _CONFIGURED else None

# ================= access_token 缓存 =================
_tok = {"token": "", "expire": 0}
_tok_lock = threading.Lock()

def get_access_token() -> str:
    with _tok_lock:
        if time.time() < _tok["expire"] and _tok["token"]:
            return _tok["token"]
        try:
            r = requests.get("https://qyapi.weixin.qq.com/cgi-bin/gettoken",
                             params={"corpid": CORPID, "corpsecret": SECRET}, timeout=15).json()
            if r.get("errcode") == 0:
                _tok["token"] = r["access_token"]
                _tok["expire"] = time.time() + r.get("expires_in", 7200) - 200
                logger.info("🔑 刷新企业微信 access_token 成功")
            else:
                logger.error(f"⚠️ 获取 access_token 失败: {r}")
        except Exception as e:
            logger.error(f"⚠️ access_token 请求异常: {e}")
        return _tok["token"]

# ================= 主动推送（含长文本分段）=================
_MAX_BYTES = 2000   # 企业微信文本上限约 2048 字节，留余量

def _split_text(text: str, limit: int = _MAX_BYTES):
    out, buf = [], ""
    for line in text.split("\n"):
        # 单行就超限，按字节硬切
        while len(line.encode("utf-8")) > limit:
            cut = line
            while len(cut.encode("utf-8")) > limit:
                cut = cut[:-1]
            out.append(cut); line = line[len(cut):]
        if len((buf + "\n" + line).encode("utf-8")) > limit:
            if buf: out.append(buf)
            buf = line
        else:
            buf = (buf + "\n" + line) if buf else line
    if buf: out.append(buf)
    return out or [""]

def send_text(touser: str, text: str):
    token = get_access_token()
    if not token:
        return
    url = f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}"
    for i, seg in enumerate(_split_text(text)):
        payload = {"touser": touser, "msgtype": "text",
                   "agentid": int(AGENTID), "text": {"content": seg}}
        try:
            r = requests.post(url, json=payload, timeout=30).json()
            if r.get("errcode") != 0:
                logger.error(f"⚠️ 发送应用消息失败 seg{i}: {r}")
        except Exception as e:
            logger.error(f"⚠️ 发送应用消息异常: {e}")

# ================= 授权白名单（热加载）=================
_wl = {"users": set(), "mtime": 0, "open": False}
_wl_lock = threading.Lock()

def _load_whitelist():
    with _wl_lock:
        try:
            if not os.path.exists(WHITELIST):
                _wl["users"], _wl["open"] = set(), False
                return
            mtime = os.path.getmtime(WHITELIST)
            if mtime == _wl["mtime"]:
                return
            with open(WHITELIST, encoding="utf-8") as f:
                data = json.load(f)
            _wl["users"] = set(data.get("authorized_userids", []))
            _wl["open"]  = bool(data.get("allow_all", False))   # 应急：临时全放行
            _wl["mtime"] = mtime
            logger.info(f"🔐 白名单已加载: {len(_wl['users'])} 人 allow_all={_wl['open']}")
        except Exception as e:
            logger.error(f"⚠️ 白名单加载失败，按全部拒绝处理: {e}")
            _wl["users"], _wl["open"] = set(), False

def is_authorized(userid: str) -> bool:
    _load_whitelist()
    with _wl_lock:
        return _wl["open"] or userid in _wl["users"]

# ================= 消息去重 =================
_seen = OrderedDict(); _seen_lock = threading.Lock()
def _dup(msg_id: str) -> bool:
    if not msg_id:
        return False
    with _seen_lock:
        if msg_id in _seen:
            return True
        _seen[msg_id] = time.time()
        while len(_seen) > 500:
            _seen.popitem(last=False)
        return False

# ================= 业务处理 =================
import re as _re
_CROP_RE = _re.compile(r'\[📷\s*[^\]]+\]')

def _handle(from_user: str, content: str):
    """后台线程：白名单 → answer_for → 推送。"""
    try:
        if not is_authorized(from_user):
            logger.warning(f"⛔ 未授权用户 {from_user} 提问被拒：{content[:40]}")
            send_text(from_user, "🔒 您暂无权限使用本知识库，请联系管理员开通。")
            return
        logger.info(f"💬 [{from_user}] {content[:60]}")
        text, _imgs = brain.answer_for(content)
        # 企业微信文本消息：去掉图片占位标记，Markdown 表格转纯文本
        text = brain._convert_md_tables(_CROP_RE.sub('', text)).strip()
        send_text(from_user, text or "（无内容）")
    except Exception:
        logger.exception("❌ wecom 处理异常")
        try:
            send_text(from_user, "❌ 系统处理异常，请稍后重试。")
        except Exception:
            pass

# ================= FastAPI 回调 =================
app = FastAPI(title="企业微信 EHS 知识库入口 (v12)")

@app.get("/wecom/callback")
async def verify(msg_signature: str = "", timestamp: str = "", nonce: str = "", echostr: str = ""):
    """URL 有效性验证：解密 echostr 原样返回明文。"""
    if not _CONFIGURED:
        return Response("not configured", status_code=503)
    if not _crypto.verify_sign(msg_signature, timestamp, nonce, echostr):
        logger.warning("⚠️ URL 验证签名不匹配")
        return Response("invalid signature", status_code=403)
    try:
        plain = _crypto.decrypt(echostr)
        logger.info("✅ 回调 URL 验证通过")
        return Response(plain)
    except Exception as e:
        logger.error(f"⚠️ echostr 解密失败: {e}")
        return Response("decrypt failed", status_code=403)

@app.post("/wecom/callback")
async def receive(request: Request, msg_signature: str = "", timestamp: str = "", nonce: str = ""):
    """接收消息：验签解密 → 秒回空串 ACK → 后台线程算答案主动推。"""
    if not _CONFIGURED:
        return Response("not configured", status_code=503)
    body = await request.body()
    try:
        encrypt = ET.fromstring(body).find("Encrypt").text
    except Exception as e:
        logger.error(f"⚠️ 回调体解析失败: {e}")
        return Response("")
    if not _crypto.verify_sign(msg_signature, timestamp, nonce, encrypt):
        logger.warning("⚠️ 消息签名不匹配")
        return Response("", status_code=403)
    try:
        xml = ET.fromstring(_crypto.decrypt(encrypt))
    except Exception as e:
        logger.error(f"⚠️ 消息解密失败: {e}")
        return Response("")

    msg_type  = (xml.findtext("MsgType") or "").strip()
    from_user = (xml.findtext("FromUserName") or "").strip()
    msg_id    = xml.findtext("MsgId") or ""

    if _dup(msg_id):
        return Response("")          # 重投，已处理
    if msg_type != "text":
        # 非文本（图片/事件等）暂不处理；ACK 即可
        return Response("")

    content = (xml.findtext("Content") or "").strip()
    clean = _re.sub(r'\[[^\]]*\]', '', content).strip()   # 滤纯表情
    if len(clean) < 3:
        return Response("")

    # 秒回 ACK，重活丢后台
    threading.Thread(target=_handle, args=(from_user, content), daemon=True).start()
    return Response("")

@app.get("/healthz")
async def healthz():
    return {"ok": True, "configured": _CONFIGURED, "whitelist_loaded": len(_wl["users"])}

if __name__ == "__main__":
    _load_whitelist()
    logger.info(f"🎉 企业微信入口启动 :{PORT}  configured={_CONFIGURED}  agentid={AGENTID or '(未配)'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
