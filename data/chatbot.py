import os
import sys
import threading
import webbrowser
import gradio as gr
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.cache import InMemoryCache
import langchain

# ====== Cấu hình cache ======
langchain.llm_cache = InMemoryCache()

# ====== Cấu hình chatbot ======
DATA_PATH = os.path.join("kien_thuc_giao_duc.txt")
CHROMA_DIR = "data/chroma_db"
OLLAMA_BASE = "http://127.0.0.1:11434"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

# ====== 1) Load dữ liệu ======
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_PATH}")

loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()

# ====== 2) Chia nhỏ văn bản ======
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# ====== 3) Embedding + Chroma vectorstore ======
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)

if not os.path.exists(CHROMA_DIR):
    vectorstore = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
else:
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# ====== 4) LLM ======
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE)

# ====== 5) Prompt ======
EDU_PROMPT = """
Bạn là trợ lý học vụ của Trường Đại học Cần Thơ (CTU), nhiệm vụ:

- Trả lời CHÍNH XÁC và NGẮN GỌN dựa trên dữ liệu trong [CONTEXT].
- KHÔNG được bịa thông tin.
- Nếu câu hỏi ngoài phạm vi học vụ, sinh viên, CTU → trả lời:
  "Xin lỗi, tôi chỉ hỗ trợ thông tin liên quan đến học vụ và sinh viên Trường Đại học Cần Thơ."
- Nếu không có dữ liệu phù hợp trong context → trả lời:
  "Tôi chưa có dữ liệu về nội dung này."

Dưới đây là dữ liệu bạn được phép sử dụng:

[CONTEXT]
{context}

Câu hỏi: {question}

Trả lời:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=EDU_PROMPT
)

# ====== 6) Hàm trả lời ======
def chatbot_ctu(user_input, chat_history):
    # Tìm đoạn văn phù hợp
    results = vectorstore.similarity_search(user_input, k=4)
    context = "\n\n".join([doc.page_content for doc in results]) if results else ""

    # Không có dữ liệu RAG
    if not context:
        answer = "Tôi chưa có dữ liệu về nội dung này."
    else:
        final_prompt = prompt.format(context=context, question=user_input)
        answer = llm.invoke(final_prompt)   # Gemma không dùng stream tốt

    chat_history.append((user_input, answer))
    return chat_history, chat_history

# ====== 7) Giao diện Gradio ======
with gr.Blocks(title="🎓 Chatbot Sinh viên CTU") as demo:
    gr.Markdown("## 🎓 Chatbot Sinh viên CTU\nHỗ trợ học vụ, đăng ký học phần, học phí, quy định, tuyển sinh...")

    chat_history = gr.Chatbot(label="Trợ lý CTU")
    user_input = gr.Textbox(placeholder="Nhập câu hỏi của bạn...", label="Bạn:")
    submit_btn = gr.Button("Gửi")

    submit_btn.click(
        fn=chatbot_ctu,
        inputs=[user_input, chat_history],
        outputs=[chat_history, chat_history]
    )
# ====== 8) Tự động mở trang web khi chạy ======
webbrowser.open("http://127.0.0.1:7860")
# ====== 9) Nhấn 'q' trong TERMINAL để thoát ======
def listen_for_exit():
    print("Nhấn 'q' trong terminal để dừng chương trình.")
    while True:
        key = sys.stdin.readline().strip().lower()
        if key == "q":
            print("Đã nhận lệnh thoát. Rất vui được hỗ trợ bạn!")
            os._exit(0)

listener_thread = threading.Thread(target=listen_for_exit, daemon=True)
listener_thread.start()
demo.launch()