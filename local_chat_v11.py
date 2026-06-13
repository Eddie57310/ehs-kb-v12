import os, requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 🌟 核心参数配置
RETRIEVAL_K = 20          
SCORE_THRESHOLD = 1.2   
MAX_CHUNKS = 10 # 最多喂给大模型的段落数

print("==========================================")
print(" 🤖 V5 本地终端知识大脑 (高精度+全息透视版) 🤖 ")
print("==========================================\n")

DB_DIR = os.path.expanduser("~/doc_parser_v12/chroma_db")
OLLAMA_API = "http://localhost:11434/api/generate"

print("🧠 正在唤醒 BGE-M3...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'cuda'})
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def ask_qwen(question, context):
    prompt = f"你是一个专业的建筑工程AI助手。请严格基于以下带有生效时间的【参考资料】回答用户的【问题】。\n要求：\n1. 仔细核准资料的生效时间，排除不相关的文件。\n2. 如果资料中没有，请明确回答“资料中未找到相关规定”。\n3. 在回答的最末尾，请务必单独列出你【实际真正引用了哪些文件】作为解答依据。\n\n【参考资料】:\n{context}\n\n【问题】:{question}"
    try:
        res = requests.post(OLLAMA_API, json={"model": "qwen2.5:32b", "prompt": prompt, "stream": False, "options": {"temperature": 0.1}})
        return res.json().get('response', '模型无响应')
    except Exception as e: return f"[大脑响应失败]: {e}"

while True:
    query = input("\n🧔 请输入问题 (建议用完整句子提问，精度更高): ")
    if query.lower() in ['quit', 'exit', 'q']: break
    if not query.strip(): continue
        
    print(f"\n🔍 扫描资料 (扩大捞网 k={RETRIEVAL_K}, 阈值 < {SCORE_THRESHOLD})...")
    docs_and_scores = db.similarity_search_with_score(query, k=RETRIEVAL_K)
    
    # ================= 🌟 透视分析模块 =================
    print("📊 [RAG] 检索与提纯明细：")
    valid_docs = []
    
    for i, (doc, score) in enumerate(docs_and_scores):
        source = doc.metadata.get('source', '未知')
        
        if score < SCORE_THRESHOLD:
            if len(valid_docs) < MAX_CHUNKS:
                print(f"  ✅ [提纯采纳] 得分: {score:.4f} | 来源: {source}")
                valid_docs.append(doc)
            else:
                print(f"  ✂️ [超出名额] 得分: {score:.4f} | 来源: {source} (为了防撑爆已截断)")
        else:
            # 为了防止满屏废话，只打印前几个被淘汰的底层数据，让你心里有数
            if i < MAX_CHUNKS + 2: 
                print(f"  🗑️ [得分过低] 得分: {score:.4f} | 来源: {source} (被阈值淘汰)")
    # ==================================================
    
    if not valid_docs:
        print("\n🤖 大脑：知识库中未找到足够相关的内容。")
        continue
        
    contexts = [f"【文件来源: {d.metadata.get('source', '未知')} | 生效时间: {d.metadata.get('date_str', '未知')}】\n{d.page_content}" for d in valid_docs]
    context_text = "\n---\n".join(contexts)
    
    print(f"\n🧠 最终提纯出 {len(valid_docs)} 段核心资料，正在请求本地大模型总结...")
    answer = ask_qwen(query, context_text)
    
    print("\n================ 答 案 ================")
    print(answer)
    print("=======================================")

