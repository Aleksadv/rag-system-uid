import os
import json
import random
import psycopg2
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List

from app.schemas.users import UserIn, UserOut

DATABASE_URL = os.getenv("DATABASE_URL")
TOP_K_SHORT = 5
TOP_K_LONG = 5

app = FastAPI()

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="/models/cache"
)

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def embed_query(text: str):
    q = f"query: {text}"
    emb = model.encode(
        [q],
        normalize_embeddings=True,
    )[0]
    return emb.tolist()

def search_short(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_short
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def search_long(query, top_k):
    emb = embed_query(query)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               embedding <=> %s::vector AS distance
        FROM documents_long
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (emb, emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

@app.post("/api/question", response_model=UserOut)
def search(q: UserIn):
    short_rows = search_short(q.question, TOP_K_SHORT)
    long_rows = search_long(q.question, TOP_K_LONG)
    short = [r[0] for r in short_rows]
    long = [r[0] for r in long_rows]
    return UserOut(short=short, long=long)

# ---------- Модели данных для тренажёра ----------
class TrainerQuestionResponse(BaseModel):
    id: int
    question: str

class TrainerCheckRequest(BaseModel):
    question_id: int
    answer: str

class TrainerCheckResponse(BaseModel):
    status: str          # "Верно", "Неверно", "Верно частично"
    explanation: str

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-8)

def _to_list(emb):
    """Преобразует эмбеддинг из разных форматов в список float"""
    if hasattr(emb, 'tolist'):
        return emb.tolist()
    if isinstance(emb, str):
        return json.loads(emb)
    return list(emb)

@app.get("/api/trainer/question", response_model=TrainerQuestionResponse)
def get_random_question():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM questions")
    total = cur.fetchone()[0]
    if total == 0:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Нет вопросов. Запустите bootstrapper с REBUILD_EMBEDDINGS=true")
    random_id = random.randint(1, total)
    cur.execute("SELECT id, question_text FROM questions WHERE id = %s", (random_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Вопрос не найден")
    return TrainerQuestionResponse(id=row[0], question=row[1])

@app.post("/api/trainer/check", response_model=TrainerCheckResponse)
def check_answer(req: TrainerCheckRequest):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT answer_text, embedding FROM questions WHERE id = %s", (req.question_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Вопрос не найден")
    answer_text, embedding_ref = row
    # Вычисляем эмбеддинг ответа пользователя
    user_emb = embed_query(req.answer)
    # Приводим эталонный эмбеддинг к списку float
    ref_emb = _to_list(embedding_ref)
    similarity = cosine_similarity(user_emb, ref_emb)

    # Настроенные пороги для реалистичной оценки
    if similarity >= 0.9:
        status = "Верно"
        explanation = ""
    elif similarity >= 0.8:
        status = "Верно частично"
        explanation = answer_text
    else:
        status = "Неверно"
        explanation = answer_text

    return TrainerCheckResponse(status=status, explanation=explanation)