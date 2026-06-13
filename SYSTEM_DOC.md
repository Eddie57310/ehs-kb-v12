# EHS 智慧知识库系统文档
**项目路径**: `~/doc_parser_v10`
**最后更新**: 2026-05-23

---

## 一、系统概述

本系统是一套面向工程项目 EHS 管理的私有知识库问答平台，核心能力：
- 将企业内部制度文件、法律法规、案例库向量化入库
- 通过飞书机器人提供自然语言问答
- 支持图文混排文档的图片关联输出

**技术栈**：ChromaDB（向量库）+ HuggingFace BGE-M3（嵌入模型）+ 火山引擎 doubao-seed-2.0-pro（LLM + Vision）+ 飞书 WebSocket

---

## 二、知识库目录结构

```
Local_KB/
├── EHS案例/          ← 图文混排，逐页PNG+Vision LLM，带image_path入库
│   ├── 负面案例/
│   ├── 正面案例/
│   └── 标准化/
├── 公司内部/         ← 文字为主，MinerU结构化提取，直接入库
│   ├── 总部制度文件/
│   ├── 浙江公司制度文件/
│   └── 杭曜置地中心项目部制度文件/
└── 国家规定/         ← 纯文字提取，MinerU，跳过OCR噪音校对
    ├── 1_法律
    ├── 2_行政法规
    ├── 3_部门规章
    ├── 4_地方性法规
    ├── 5_地方政府部门规章
    ├── 6_国家标准规范
    ├── 7_建筑行业标准
    ├── 8_浙江省地方标准规范
    └── 9_中央企业制度文件
```

**文件命名规范**：`YYYYMMDD_文件名.扩展名`，日期在文件名头部，用于时间过滤检索。
**domain字段**：每个chunk存储顶级文件夹名（EHS案例/公司内部/国家规定），供飞书机器人路由使用。

---

## 三、文件处理流水线

### 3.1 PDF 路由规则

```
PDF 文件
├── 路径以 "国家规定" 或 "公司内部" 开头
│   └── process_pdf()
│       ├── MinerU OCR → Markdown
│       ├── 内嵌图片 → Vision LLM描述 → [图示：xxx] 替换
│       ├── PAGE标记追踪页码 → 每chunk存page字段
│       ├── 国家规定跳过OCR噪音校对；公司内部正常检测
│       ├── 过滤目录页块（_is_toc_chunk）
│       ├── extract_pdf_structured_chunks() → 章节感知切块
│       └── 直接写入 ChromaDB（chunk_reviewer实时渲染源页）
│
└── EHS案例目录（图文混排）
    └── process_pdf_with_vision()
        ├── PyMuPDF 渲染每页为 PNG → slide_images/
        ├── PyMuPDF get_text() 提取文字
        ├── 字数 < 100 → 调用 Vision LLM 识图
        └── 直接写入 ChromaDB（每chunk带image_path）
```

### 3.2 PPTX 处理

```
process_pptx()
├── LibreOffice 转 PDF
├── PyMuPDF 渲染每页为 PNG → slide_images/（始终保存）
├── python-pptx 提取文字
├── 文字 < 100 字 → Vision LLM
└── 直接写入 ChromaDB（每chunk带image_path和page）
```

### 3.3 DOCX 处理

```
extract_docx_structured_chunks()
├── 按阅读顺序遍历body（段落+表格交错）
├── 按 Heading 样式检测标题边界
├── 表格 → Markdown格式（| col | col |）
├── 每节生成带路径前缀的 chunk
│   格式：【一级标题 > 二级标题】\n正文
├── 超长节进一步分割，保留前缀
└── 无标题则降级为 RecursiveCharacterTextSplitter
直接写入 ChromaDB
```

### 3.4 Excel 处理

```
_chunk_sheet()
├── 合并单元格感知（追溯到主格）
├── 自动检测表头行（序号/岗位/名称等）
├── 跳过合并表头延伸到数据区的行（防止重复"序号"块）
├── 按第一列岗位/类别分组
├── 纯数字序号自动并入标题
└── 直接写入 ChromaDB
```

---

## 四、章节感知切块（核心设计）

**问题根源**：MarkdownTextSplitter 按字数切，会在章节中间截断，导致"3.2 围护设计方案评审阶段"的标题和"3.2.1 介入时间"的内容被分到不同 chunk，问答检索时丢失上下文。

**解决方案**：`extract_pdf_structured_chunks()`
- 检测 `# ## ###` Markdown 标题 和 `3.2.1` 数字编号标题
- 以标题为边界切块，每块前缀完整章节路径
- 格式：`【第3章 围护工程 > 3.2 设计评审 > 3.2.1 介入时间】\n正文`
- 超长块再分割，但保留前缀
- 无标题时自动降级为普通切分

同样逻辑已用于 DOCX（`extract_docx_structured_chunks`）。

---

## 五、人工审核流程

非法律法规 PDF 和所有 PPTX 不直接入库，流程如下：

1. Sync 生成 `review_pending/*.yaml`（含每页图片路径 + 提取文字）
2. 打开 Web UI（端口 8081）→ Tab 3「图文内容审核」
3. 点「刷新文件列表」→ 选文件 → 点「加载」
4. 「批量通过文字页」一键通过 text 提取的页
5. 「跳转下一个Vision页」逐页检查 LLM 识别结果，编辑噪音
6. 点「完成审核并写入数据库」→ 写入 ChromaDB，YAML 状态变 `completed`

**注意**：跳过的页永久不入库；YAML 状态 `completed` 后下次 sync 不再重复处理。

---

## 六、飞书问答机制

### 6.1 检索流程

```
用户提问
├── 命令判断（/说明 /目录 /唤醒深度 等）
├── 目录查询意图检测（_detect_dir_query → _list_kb_dir，扫描文件系统）
├── 匹配精度解析（超精准0.5无兜底/精准0.65/普通0.8/宽松1.1/自定义阈值）
├── 时间过滤检测（近三年/近五年/具体年份）
├── domain路由（顶级类别关键词 → 按domain字段过滤）
├── 定向检索（具体文件名关键词 → 按source路径过滤）
├── 全局向量检索 k=20，得分过滤 < score_threshold，取前5
├── 合并上下文 → 发送 LLM
├── clean_for_feishu()：去掉 # 标题符 + Wingdings 字符
└── reply_feishu_with_images()：解析 [📷 path] 标签，文字图片交错输出
```

### 6.2 路由关键词

**domain级**（按顶级文件夹过滤）：负面案例/正面案例/案例库/警示手册 → EHS案例；公司内部/内部制度 → 公司内部；国家规定/法律法规/国家标准/行业标准 → 国家规定

**文件级定向**（按source路径匹配）：中央企业制度文件/总部制度文件/应知应会手册/管理十条/事故事件管理细则/考核实施细则/承包商安全管理细则 等

### 6.2b 匹配精度档位（技能4）

| 用户说 | 阈值 | 兜底 |
|---|---|---|
| 超精准匹配 | 0.5 | 无 |
| 精准匹配 | 0.65 | 有 |
| 普通匹配（默认） | 0.8 | 有 |
| 宽松匹配 | 1.1 | 有 |
| 阈值N | N | 有 |

### 6.3 隐藏指令

| 指令 | 效果 |
|------|------|
| /唤醒深度 | 切换至火山引擎 DeepSeek-V3.2 |
| /唤醒火山 | 切回默认 doubao-seed-2.0-pro |
| /唤醒深海 | 本地 DeepSeek-R1 32B |
| /唤醒千问 | 本地 Qwen2.5 32B |
| /唤醒极速 | 本地 Qwen2.5 14B |

---

## 七、增量同步机制

`sync_kb_v10.py` 每次运行：
1. 对比 ChromaDB 已有文件 vs Local_KB 物理文件
2. 已删除文件 → 清除向量 + batch_output 缓存
3. 新增文件 → 走完整处理流水线
4. 已存在文件 → 跳过（不重复处理）

**更新文件的正确方式**：删旧文件 → 放入新文件（文件名含新日期）→ 跑 sync → 只对新文件的 YAML 审核。

---

## 八、关键参数配置

| 参数 | 值 | 说明 |
|------|----|------|
| VISION_TEXT_THRESHOLD | 100 | 低于此字数调 Vision LLM |
| chunk_size | 800 | 向量块最大字数 |
| chunk_overlap | 150 | 块间重叠字数 |
| batch_size | 32 | ChromaDB 分批写入，防 OOM |
| embedding device | cpu | Sync 用 CPU，避免与飞书机器人争 GPU |
| feishu bot embedding | cuda | 飞书机器人用 GPU 加速检索 |
| 最大返回图片数 | 3 | 飞书回复最多附3张图 |
| OCR_BATCH_SIZE | 20 | OCR校对每批块数，批间暂停5s |
| OCR_BATCH_PAUSE | 5s | 批间暂停，避免API限流 |
| OCR LLM 超时 | 60s | 单次调用超时时间 |

---

## 九、已知问题与局限

| 问题 | 现状 | 建议 |
|------|------|------|
| 流程图/组织架构图 | 只能提取文字，空间关系丢失 | 依赖图片输出让用户直接看图 |
| PDF 制作质量缺陷（ExtGState等MuPDF警告） | 不影响提取，忽略即可 | 无需处理 |
| EHS案例库6个方框噪音标签 | 人工审核时手动删除 | 审核阶段过滤 |
| 法律法规 MinerU OCR 超长文件 | 超时3600s会失败，进 Failed_PDFs | 可拆分大文件后重试 |
| OCR噪音检测误判工程规范 | `_has_ocr_noise()` 对公式/标准号误判率高 | ✅ 已修复：国家规定目录跳过OCR校对 |

---

## 十、目录说明

| 目录 | 说明 |
|------|------|
| Local_KB/ | 知识库源文件（不入git） |
| nano_chroma_db/ | 向量数据库（不入git） |
| batch_output/ | MinerU解析缓存（不入git） |
| slide_images/ | 页面渲染PNG（不入git） |
| review_pending/ | 待审核YAML（不入git） |
| Failed_PDFs/ | 三引擎全部失败的PDF副本+error日志 |
| logs/ | 运行日志（不入git） |
| org_confirm/ | 组织架构确认YAML |
| SYSTEM_DOC.md | 本系统说明文档（根目录，随代码变更更新） |

---

## 十一、运营操作手册

### 启动服务
```bash
cd ~/doc_parser_v10
# 飞书机器人
nohup venv/bin/python feishu_ws_server_v10.py > logs/feishu_$(date +%Y%m%d).log 2>&1 &
# Web UI
nohup venv/bin/python web_ui.py > logs/webui_$(date +%Y%m%d).log 2>&1 &
# API 服务（Web UI依赖）
nohup venv/bin/python api_server_v10.py > logs/api_$(date +%Y%m%d).log 2>&1 &
```

### 同步知识库
```bash
cd ~/doc_parser_v10
nohup venv/bin/python sync_kb_v10.py > logs/sync_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/sync_*.log
```

### 删库重建
```bash
rm -rf nano_chroma_db/* batch_output/* slide_images/* review_pending/* Failed_PDFs/*
venv/bin/python sync_kb_v10.py
```

---

---

## 十二、待开发工具

### chunk_reviewer.py（数据质检工具）
- **用途**：人工逐块核对数据库内容与源文件，确保入库质量
- **布局**：双屏宽屏，左侧源文件（PDF页面图片/文本），右侧数据库块（可编辑）
- **功能**：按文件浏览chunk、编辑保存（自动重新嵌入）、删除块、审核进度记录
- **端口**：8082（独立于 web_ui 的 8081）
- **状态**：待开发（sync完成、数据稳定后实施）

---

---

## 十三、架构演进记录

### 2026-05-23 重大架构重构：MD优先 + 两轨并行

#### 为什么要重构

v1 方案（sync_kb_v10.py 直接入库）在实际使用中暴露出以下根本性问题：

**1. 切片质量差**
MinerU 生成的 .md 文件表格全部转成了图片（`![](images/xxx.jpg)`），文字内容虽然有，但表格语义完全丢失。工程规范、法规标准里大量表格无法被检索。同时 `extract_pdf_structured_chunks()` 的自定义切块逻辑复杂，在实际文件上切出来的块顺序乱、边界不自然。

**2. 页码追踪失效**
MinerU 从未在 .md 文件中写入 `--- PAGE N ---` 标记（sync 代码一直在找一个不存在的东西），导致所有 PDF chunk 的 `page` metadata 全部为 1，chunk_reviewer 永远只显示第1页，源文件与内容对不上，无法人工校正。

**3. 无法人工干预**
数据直接从源文件流向 ChromaDB，中间没有人工可编辑的中间层。图片标记只能靠 Vision LLM 自动识别，无法补录。OCR 噪音只能靠 LLM 校对，效果有限。

**4. 中间产物混乱**
batch_output、reviewed_md、slide_images、review_pending 各种临时目录累积大量无法追溯的中间文件，系统状态不清晰。

#### 新架构：两轨并行 + MD优先

**轨道 A：EHS案例**（不变）
- 逐页 PNG 渲染 + Vision LLM 识图
- 每页一个 chunk，带 image_path
- 图文天然关联，Vision LLM 效果好

**轨道 B：国家规定 / 公司内部**（全新）
```
Local_KB 源文件（唯一原始数据）
    ↓ rebuild_md.py
reviewed_md/（干净 Markdown，人工可编辑）
  - PDF：MinerU → content_list.json → 重建MD（表格为Markdown文本，含页码标记）
  - DOCX：python-docx → MD（标题层级+表格）
  - XLSX：openpyxl → MD（合并单元格感知）
    ↓ 人工审核（四栏工具，待开发）
    ↓ 从reviewed_md切片
chroma_db/（向量库）
```

**关键原则**：
- `reviewed_md/` 是唯一的数据加工层，人工可读可编辑
- `batch_output/` 仅作 MinerU 运行缓存，可随时删除重建
- 图片补录（`[📷 path]`）作为阶段二，不阻塞阶段一入库
- 向量库名从 `nano_chroma_db` 改为 `chroma_db`（历史遗留随意命名）

#### 同步完成的其他修复

- `chunk_reviewer.py`：检测"全部 page=1"旧数据，自动改用线性估算；支持渲染 DOCX 源文件
- `sync_kb_v10.py`：新增 `_inject_page_markers()`，从 content_list.json 注入页码标记；DOCX chunk 加顺序页码
- `pdf_table_extractor.py`：Rule 2 乱码检测加文字量条件，修复工程规范被误判为乱码
- 飞书机器人：移除所有兜底逻辑，加表情过滤，temperature=0，图片标记上下文保留
- Telegram 机器人：新建 telegram_bot_v10.py，复用飞书同一知识库

---

---

## 十四、客关部排雷库专项操作手册

> 最后更新：2026-05-29

### 14.1 背景与现状

排雷库目前有三个版本，均已完成格式清洗、配图补录，全部达到 V3.0 标准格式：

| 版本 | source 路径 | 案例块数 | 格式 | 配图 |
|------|-------------|---------|------|------|
| V1.0 | `客关部/排雷库/排雷库案例/25050501_杭州公司武器库之排雷库V1.0.pdf` | 60 | 全部整洁 | 60 张自动裁剪 |
| V2.0 | `客关部/排雷库/排雷库案例/20250926_杭州公司武器库之排雷库V2.0.pdf` | 26 | 全部整洁 | 13 人工截图 + 13 自动裁剪 |
| V3.0 | `客关部/排雷库/排雷库案例/20260401_浙江公司武器库之排雷库V3.0.pdf` | 28 | 全部整洁 | 28 人工截图 |

**V3.0 标准格式**（每块结构）：
```
编号：XX-XXXX
案例名称：…
排雷节点：…
问题分类：…
风险描述：…
风险原因：…
预防建议：…
涉及专业：…
涉及项目：…
产生后果：…
[📷 user_crops/文件名.png]
```

### 14.2 格式清洗原理（reformat_cases.py）

PDF 原始入库时块内字段顺序混乱、无标准分隔符。`reformat_cases.py` 对每块按字段关键词重新拼装：

- `clean`：已带 `[📷 真实图]` 或以 `编号：` 开头 → 保留不动
- `case`：以 `核心商密` 开头且含编号 → 按 V3.0 字段重排，末尾加 `[📷 待补]`
- `noncase`：清单分隔页 / PPT 尾页垃圾碎块 → 删除

### 14.3 自动截图原理（add_case_screenshots.py）

每页版式固定：左侧粉色字段表 + **右侧蓝/靛色边框框**（框内是配图+图注）。

检测逻辑：在图片右侧 40%+ 列中找**贯穿整框高 >55% 的连续蓝色竖线**（远强于图片内部零散蓝色如天空/窗户），定位框的左右上下四条边，向内缩 6px 裁剪。

### 14.4 新版本 V4.0 入库完整流程

#### 第一步：放文件 + 入库

```bash
# 把 PDF 放到指定目录（文件名必须含日期前缀）
# Local_KB/客关部/排雷库/排雷库案例/YYYYMMDD_xxx排雷库V4.0.pdf

cd ~/doc_parser_v11
# 同步入库（增量，不影响旧文件）
venv/bin/python sync_kb_v11.py
```

#### 第二步：格式清洗

```bash
# dry-run：看分类结果，确认待删的都是噪音（清单分隔页/PPT尾页），没有误删案例
venv/bin/python reformat_cases.py "V4.0"

# 确认无误后写库（删噪音 + 重排格式 + 重嵌入）
venv/bin/python reformat_cases.py "V4.0" --write
```

**典型输出**：
```
分类: 保留(已审)0  重排N  删除(非案例)M
   待删: 核心商密·长期 / 排雷库|机电配套（JD）
   待删: 谢谢大家！
```

#### 第三步：自动补截图

```bash
# dry-run：预览图存到 _crop_preview/，目视确认框有没有裁错
venv/bin/python add_case_screenshots.py "V4.0"

# 确认无误后写库（裁图存 user_crops/ + 替换 [📷 待补] + 重嵌入 + 更新 index.json + 重建 BM25）
venv/bin/python add_case_screenshots.py "V4.0" --write

# 清理预览目录
rm -rf _crop_preview
```

> **注意**：如果检测不到蓝框（报 ⚠️ 未检测到），说明该页版式有变化，在 8085 复核器手动框选即可。

#### 第四步：8085 复核器抽查

打开 `localhost:8085`，选 V4.0 的 source，逐块检查格式和配图是否正确。如发现少数配图裁歪，直接在复核器里重新框选覆盖。

#### 第五步：验证

```bash
venv/bin/python -c "
import chromadb, os, re
c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection(c.list_collections()[0].name)
r = col.get(where={'source': {'\\$contains': 'V4.0'}}, include=['documents'])
tag = re.compile(r'\[📷\s*([^\]]+?)\s*\]')
n = len(r['ids'])
todo = sum('待补' in d for d in r['documents'])
tags = [tag.search(d).group(1) for d in r['documents'] if tag.search(d)]
missing = [t for t in tags if t != '待补' and not os.path.exists(t)]
print(f'共{n}块 | 待补{todo} | 文件缺失{len(missing)}')
"
```

期望结果：`待补 0 | 文件缺失 0`

### 14.5 关键脚本说明

| 脚本 | 用途 | 重要参数 |
|------|------|---------|
| `reformat_cases.py <子串>` | 排雷库格式清洗 dry-run | 无 |
| `reformat_cases.py <子串> --write` | 实际写库 | — |
| `add_case_screenshots.py <子串>` | 自动截图 dry-run | `--preview-dir <目录>` |
| `add_case_screenshots.py <子串> --write` | 实际写库 | — |

两个脚本都有 dry-run 保护，先看结果再 `--write`，不会误操作。

### 14.6 常见问题

**Q：`add_case_screenshots.py` 某页报 "未检测到蓝框"？**  
A：该页版式可能有差异（如空白页或双图并列），在 8085 里手动框选即可。

**Q：飞书发不出图片（error 99991663）？**  
A：access token 过期，重启飞书 bot 即可：
```bash
cd ~/doc_parser_v11
kill $(pgrep -f feishu_ws_server_v11)
sleep 2
nohup venv/bin/python feishu_ws_server_v11.py >> logs/feishu_ws_server_v11_$(date +%Y-%m-%d).log 2>&1 &
```

**Q：`reformat_cases.py` 的字段提取效果不好？**  
A：检查 PDF 文本质量。V1/V2/V3 均已验证有效；新版本如果用了不同字段名，需更新 `FIELDS` 列表。

---

*本文档由 Claude Code 维护，代码有重要变更时同步更新。*
