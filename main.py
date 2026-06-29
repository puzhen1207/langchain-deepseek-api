import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 初始化 FastAPI 应用
app = FastAPI(title="LangChain DeepSeek 编排服务")

# 定义请求参数结构
class ChatRequest(BaseModel):
    query: str

# 从环境变量读取密钥（部署时配置，不硬编码）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 初始化 LangChain 编排链路（LCEL 语法）
# 你可以在这里替换成任意复杂的 LangChain / LangGraph 逻辑
def build_chain():
    prompt = ChatPromptTemplate.from_template(
        "你是专业的内容总结助手，请用简洁清晰的语言回答用户问题：\n{query}"
    )
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3
    )
    output_parser = StrOutputParser()
    # LCEL 链式编排
    return prompt | llm | output_parser

chain = build_chain()

# 定义对外接口
@app.post("/chat", summary="调用 LangChain 编排链路")
async def chat_endpoint(request: ChatRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="未配置 DEEPSEEK_API_KEY 环境变量")
    
    try:
        # 执行 LangChain 编排流程
        result = chain.invoke({"query": request.query})
        return {
            "code": 0,
            "final_answer": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "code": -1,
            "final_answer": "",
            "status": f"执行失败: {str(e)}"
        }

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "ok"}