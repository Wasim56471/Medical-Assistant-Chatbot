
import os
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── Dataset ───────────────────────────────────────────────────
print("⏳ Loading dataset...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
dataset = dataset.select(range(20000))

docs = []
for i, row in enumerate(dataset):
    if not row.get("input") or not row.get("output"):
        continue
    if len(row["output"].strip()) < 30:
        continue
    docs.append(Document(
        page_content=f"Patient: {row['input'].strip()}\nDoctor: {row['output'].strip()}",
        metadata={"source": "HealthCareMagic-100k", "index": i}
    ))
print(f"✅ {len(docs)} documents loaded.")

# ── Embeddings + Vector store ─────────────────────────────────
print("⏳ Building vector store...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever   = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
print("✅ Vector store ready.")

# ── LLM ───────────────────────────────────────────────────────
client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

def call_llm(prompt_text):
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are MedAssist, a professional medical information assistant. "
                    "CRITICAL RULES:\n"
                    "1. Respond in English only — never switch to any other language.\n"
                    "2. Be empathetic, clear, and concise.\n"
                    "3. Never assume the user has symptoms unless they stated them.\n"
                    "4. Always recommend consulting a qualified doctor for personal advice.\n"
                    "5. Do not repeat the disclaimer — it is added automatically."
                )
            },
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content

# ── Prompt (matches Cell 10 exactly) ─────────────────────────
PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are MedAssist, a professional medical information assistant.

STRICT RULES:
- The doctor-patient examples below are REFERENCE MATERIAL ONLY — they are NOT about the current user.
- NEVER assume the current user has any symptoms, readings, or conditions mentioned in the reference examples.
- NEVER use specific numbers, readings, or personal details from the reference examples in your response.
- Answer ONLY what the user actually asked — do not invent context.
- If the user sends a greeting (hi, hello, hey), respond warmly and briefly — no medical content.
- If the user asks about anything unrelated to health or medicine (e.g. geography, politics, history, science, sports, population, countries, technology), respond ONLY with: "I'm only able to assist with medical and health-related questions. Please ask me about symptoms, conditions, medications, or general health advice." — do not answer the question.
- NEVER start your response with "I understand" — vary your opening phrases naturally.
- If the user asked for your well being (how are you, how are you doing), respond warmly, and ask how you can help them today.
- If the user message does not contain any greeting word, NEVER start your response with "Hello", "Hi", "Hey" or any greeting word — get straight to the answer.
- NEVER give the exact same response twice — always vary the wording, structure, and detail level.
- For general medical questions, explain the topic clearly and objectively.
- Always recommend consulting a qualified doctor for personal medical advice.
- Do not repeat the disclaimer — it is added automatically.

--- Reference Examples (NOT the current user's history) ---
{context}
-----------------------------------------------------------

Conversation so far:
{chat_history}

User: {question}
MedAssist:"""
)

# ── Response function (matches Cell 12 exactly) ───────────────
chat_history_store = []

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ Medical Disclaimer: For general informational purposes only. "
    "Please consult a qualified healthcare professional for personalised guidance."
)

def ask_doctor(user_query, history):
    global chat_history_store

    if not user_query.strip():
        return "Please type a message."

    # Format last 4 turns of history
    history_text = ""
    for human, ai in chat_history_store[-4:]:
        history_text += f"User: {human}\nMedAssist: {ai}\n\n"

    # Retrieve doctor consultation context
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    # Fill prompt and call LLM
    filled_prompt = PROMPT.format(
        context=context,
        chat_history=history_text.strip(),
        question=user_query
    )

    try:
        answer = call_llm(filled_prompt)
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

    chat_history_store.append((user_query, answer))
    return answer + DISCLAIMER

# ── HTML (matches Cell 14 exactly) ───────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>MedAssist</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --primary: #d97706;        /* amber-orange — header, buttons, avatars */
      --primary-dark: #b45309;   /* darker orange for hover states */
      --bg: #f5f5f4;             /* warm off-white background */
      --border: #d6d3d1;         /* stone-300 border */
      --text: #1c1917;           /* stone-900 text */
      --muted: #78716c;          /* stone-500 muted text */
      --radius: 14px;
      --emergency: #dc2626;      /* red — kept for emergency messages */
    }
    body { font-family: \'Segoe UI\', system-ui, sans-serif; background: var(--bg);
           height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    header { background: var(--primary); color: white; padding: 12px 20px;
             display: flex; align-items: center; justify-content: space-between;
             flex-shrink: 0; box-shadow: 0 2px 10px rgba(14,116,144,0.3); }
    .header-left { display: flex; align-items: center; gap: 12px; }
    .logo-icon    { font-size: 26px; }
    .header-title { font-size: 20px; font-weight: 700; }
    .header-sub   { font-size: 11px; opacity: 0.8; margin-top: 1px; }
    .header-stats { display: flex; gap: 12px; }
    .stat-badge { background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
                  border-radius: 20px; padding: 4px 12px; font-size: 12px; color: white; }
    .disclaimer { background: #fff7ed; border-bottom: 2px solid #fed7aa; color: #9a3412;
                  text-align: center; padding: 6px 16px; font-size: 12px;
                  font-weight: 500; flex-shrink: 0;
                   }
    .main { flex: 1; display: flex; flex-direction: column; max-width: 800px;
            width: 100%; margin: 0 auto; padding: 14px 14px 0; overflow: hidden; }
    .chips { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 10px; flex-shrink: 0; }
    .chip  { background: white; border: 1.5px solid var(--border); border-radius: 20px;
             padding: 5px 13px; font-size: 12px; color: var(--primary);
             cursor: pointer; font-family: inherit; transition: all 0.15s; }
    .chip:hover { background: var(--primary); color: white; border-color: var(--primary); }
    #chat-window { flex: 1; background: white; border: 1px solid var(--border);
                   border-radius: var(--radius) var(--radius) 0 0; overflow-y: auto;
                   padding: 18px; display: flex; flex-direction: column;
                   gap: 16px; scroll-behavior: smooth; }
    .msg { display: flex; gap: 10px; max-width: 88%; animation: fadeUp 0.22s ease; }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.bot  { align-self: flex-start; }
    .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center;
              justify-content: center; font-size: 17px; flex-shrink: 0; margin-top: 2px; }
    .msg.bot  .avatar { background: var(--primary); }
    .msg.user .avatar { background: #fed7aa; }   /* light orange */
    .bubble { padding: 12px 15px; border-radius: 16px; font-size: 13.5px;
              line-height: 1.65; white-space: pre-wrap; word-break: break-word; }
    .msg.bot  .bubble { background: #f8fdff; border: 1px solid var(--border);
                        border-top-left-radius: 4px; color: var(--text); }
    .msg.user .bubble { background: var(--primary); color: white; border-top-right-radius: 4px; }
    .bubble.emergency { background: #fff1f2; border: 2px solid var(--emergency);
                        border-top-left-radius: 4px; color: #7f1d1d; }
    .typing-dots { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
    .typing-dots span { width: 7px; height: 7px; background: var(--primary);
                        border-radius: 50%; animation: bounce 1.2s infinite; opacity: 0.5; }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
                        30% { transform: translateY(-5px); opacity: 1; } }
    .input-area { background: white; border: 1px solid var(--border); border-top: none;
                  border-radius: 0 0 var(--radius) var(--radius); padding: 12px;
                  display: flex; gap: 8px; align-items: center; flex-shrink: 0; margin-bottom: 14px; }
    #user-input { flex: 1; border: 1.5px solid var(--border); border-radius: 24px;
                  padding: 10px 16px; font-size: 13.5px; outline: none;
                  font-family: inherit; color: var(--text); transition: border-color 0.2s; }
    #user-input:focus { border-color: var(--primary); }
    #send-btn { background: var(--primary); color: white; border: none; border-radius: 50%;
                width: 40px; height: 40px; font-size: 17px; cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                flex-shrink: 0; transition: background 0.2s; }
    #send-btn:hover    { background: var(--primary-dark); }
    #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .reset-btn { background: transparent; border: 1.5px solid var(--border); border-radius: 20px;
                 padding: 8px 12px; font-size: 12px; color: var(--muted); cursor: pointer;
                 font-family: inherit; white-space: nowrap; transition: all 0.15s; }
    .reset-btn:hover { border-color: var(--primary); color: var(--primary); }
    #chat-window::-webkit-scrollbar { width: 5px; }
    #chat-window::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="logo-icon">&#x1FA7A;</div>
      <div>
        <div class="header-title">MedAssist</div>
        <div class="header-sub">AI Medical Assistant &mdash; Powered by RAG + ChatDoctor Dataset</div>
      </div>
    </div>
    <div class="header-stats">
      <div class="stat-badge">Queries: <span id="q-count">0</span></div>
      <div class="stat-badge" id="status-badge">&#9679; Ready</div>
    </div>
  </header>
  <div class="disclaimer">
    &#9888;&nbsp; For informational purposes only &mdash; always consult a qualified doctor for personal medical advice.
  </div>
  <div class="main">
    <div class="chips">
      <button class="chip" onclick="sendChip(\'I have had a headache and fever for two days\')">Headache &amp; fever</button>
      <button class="chip" onclick="sendChip(\'What are the symptoms of diabetes?\')">Diabetes</button>
      <button class="chip" onclick="sendChip(\'I have been feeling very tired all the time\')">Fatigue</button>
      <button class="chip" onclick="sendChip(\'I have chest pain when I exercise\')">Chest pain</button>
      <button class="chip" onclick="sendChip(\'What causes high blood pressure?\')">Blood pressure</button>
      <button class="chip" onclick="sendChip(\'I have had a sore throat for a week\')">Sore throat</button>
      <button class="chip" onclick="sendChip(\'What are the side effects of ibuprofen?\')">Ibuprofen</button>
      <button class="chip" onclick="sendChip(\'I feel dizzy when I stand up quickly\')">Dizziness</button>
    </div>
    <div id="chat-window"></div>
    <div class="input-area">
      <button class="reset-btn" onclick="resetChat()">&#8634; New session</button>
      <input id="user-input" type="text" name="query"
             placeholder="Describe your symptoms or ask a health question..."
             autocomplete="off"/>
      <button id="send-btn" onclick="sendMessage()" title="Send">&#10148;</button>
    </div>
  </div>
  <script>
    const chatWindow  = document.getElementById(\'chat-window\');
    const userInput   = document.getElementById(\'user-input\');
    const qCountEl    = document.getElementById(\'q-count\');
    const statusBadge = document.getElementById(\'status-badge\');
    let   queryCount  = 0;
    const scrollToBottom = () => { chatWindow.scrollTop = chatWindow.scrollHeight; };

    function appendMessage(role, text) {
      const wrap = document.createElement(\'div\');
      wrap.className = `msg ${role}`;
      const avatar = document.createElement(\'div\');
      avatar.className = \'avatar\';
      avatar.textContent = role === \'bot\' ? \'🩺\' : \'👤\';
      const bubble = document.createElement(\'div\');
      bubble.className  = `bubble${text.includes(\'🚨\') ? \' emergency\' : \'\'}`;
      bubble.textContent = text;
      wrap.append(avatar, bubble);
      chatWindow.appendChild(wrap);
      scrollToBottom();
    }

    function showTyping() {
      const wrap = document.createElement(\'div\');
      wrap.className = \'msg bot\'; wrap.id = \'typing\';
      const avatar = document.createElement(\'div\');
      avatar.className = \'avatar\'; avatar.textContent = \'🩺\';
      const bubble = document.createElement(\'div\');
      bubble.className = \'bubble\';
      bubble.innerHTML = \'<div class="typing-dots"><span></span><span></span><span></span></div>\';
      wrap.append(avatar, bubble);
      chatWindow.appendChild(wrap);
      scrollToBottom();
    }

    function removeTyping() { document.getElementById(\'typing\')?.remove(); }

    async function sendMessage() {
      const text = userInput.value.trim();
      if (!text) return;
      userInput.value = \'\';
      document.getElementById(\'send-btn\').disabled = true;
      appendMessage(\'user\', text);
      showTyping();
      statusBadge.innerHTML = \'&#9679; Thinking...\';
      try {
        const res  = await fetch(\'/chat\', {
          method: \'POST\',
          headers: { \'Content-Type\': \'application/json\' },
          body: JSON.stringify({ message: text })
        });
        const data = await res.json();
        removeTyping();
        appendMessage(\'bot\', data.response || \'Sorry, I could not process that.\');
        queryCount++;
        qCountEl.textContent  = queryCount;
        statusBadge.innerHTML = \'&#9679; Ready\';
      } catch (err) {
        removeTyping();
        appendMessage(\'bot\', \'Connection error. Please try again.\');
        statusBadge.innerHTML = \'&#9679; Error\';
      } finally {
        document.getElementById(\'send-btn\').disabled = false;
        userInput.focus();
      }
    }

    async function resetChat() {
      try {
        const res  = await fetch(\'/reset\', { method: \'POST\' });
        const data = await res.json();
        chatWindow.innerHTML = \'\';
        queryCount = 0; qCountEl.textContent = \'0\';
        appendMessage(\'bot\', data.response);
        statusBadge.innerHTML = \'&#9679; Ready\';
      } catch { appendMessage(\'bot\', \'Could not reset. Please refresh.\'); }
    }

    function sendChip(text) { userInput.value = text; sendMessage(); }

    window.addEventListener(\'load\', () => {
      appendMessage(\'bot\',
        \'Hello! I am MedAssist, your AI medical information assistant.\\n\\n\' +
        \'I am trained on real doctor-patient consultations and can help with \' +
        \'questions about symptoms, conditions, medications, and general health advice.\\n\\n\' +
        \'How can I help you today?\\n\\n\' +
        \'Please note: I provide general information only. Always consult a real doctor for personal advice.\'
      );
      userInput.focus();
    });

    userInput.addEventListener(\'keydown\', (e) => {
      if (e.key === \'Enter\' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  </script>
</body>
</html>"""

# ── Flask ─────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
query_counter = 0

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    global query_counter
    try:
        data    = request.get_json(force=True, silent=True) or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "Please enter a message."})
        response       = ask_doctor(message, [])
        clean_response = response.split("---")[0].strip()
        query_counter += 1
        return jsonify({"response": clean_response, "queries": query_counter})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

@app.route("/reset", methods=["POST", "GET"])
def reset():
    global query_counter, chat_history_store
    query_counter = 0
    chat_history_store = []
    return jsonify({"response": "Session reset! Hello again — I am MedAssist.\n\nHow can I help you today?"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
