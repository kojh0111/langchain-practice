from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os
from typing import Dict, List, Any

app = FastAPI()


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List] = {}

    def get_chat_history(self, session_id: str) -> List:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_message(self, session_id: str, message: Any):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)


session_manager = SessionManager()

llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="llama3.2-vision",
    temperature=0.7,
    system="당신은 친절한 한국어 AI 어시스턴트입니다. 이전 대화 내용을 바탕으로 답변하세요.",  # 시스템 프롬프트 약간 수정
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 친절한 한국어 AI 어시스턴트입니다. 이전 대화 내용을 바탕으로 답변하세요.",
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # 이전 대화 기록을 위한 Placeholder
        ("human", "{question}"),
    ]
)

api_key_header = APIKeyHeader(name="X-API-KEY")


async def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_SECRET", "default_secret_key"):
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/chat/{session_id}")
async def chat_endpoint(
    session_id: str, question: str = Form(...), api_key: str = Depends(validate_api_key)
):
    # 대화 기록 가져오기
    chat_history = session_manager.get_chat_history(session_id)

    # 현재 질문을 대화 기록에 추가 (LLM 호출 전)
    session_manager.add_message(session_id, HumanMessage(content=question))

    # LCEL 방식의 체인 생성 (변경 없음)
    chain = prompt | llm | StrOutputParser()

    async def generate_response():
        response_text = ""
        # chain.astream 호출 시 chat_history 전달
        async for chunk in chain.astream(
            {
                "question": question,
                "chat_history": chat_history,  # Placeholder 이름과 동일한 키로 전달
            }
        ):
            if chunk:
                response_text += chunk
                yield f"data: {chunk}\n\n"

        # 응답 완료 후 대화 기록에 AI 응답 추가 (변경 없음)
        session_manager.add_message(session_id, AIMessage(content=response_text))

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
async def clear_memory(session_id: str, api_key: str = Depends(validate_api_key)):
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"status": "Memory cleared"}
    raise HTTPException(status_code=404, detail="Session not found")


# 서버 상태 확인 (변경 없음)
@app.get("/")
async def health_check():
    return {"status": "active"}


if __name__ == "__main__":
    import uvicorn

    # main:app 을 현재 파일 이름(예: app.py -> app:app)으로 변경 필요
    # uvicorn.run("app:app", host="localhost", port=8001, reload=True)
    # 또는 파일 이름을 main.py로 유지
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)
