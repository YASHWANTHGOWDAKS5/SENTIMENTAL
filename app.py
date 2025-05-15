from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_API_TOKEN = "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"  # Replace with your actual API key

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

class ChatRequest(BaseModel):
    responses: list[str]

class ChatResponse(BaseModel):
    average_scores: dict[str, float]
    top_emotions: list[str]
    final_emotion: str

def call_hf_api(text: str) -> list[dict]:
    payload = {"inputs": text}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"External API request failed: {e}")

@app.post("/analyze", response_model=ChatResponse)
def analyze_emotions(chat_request: ChatRequest):
    responses = chat_request.responses
    if len(responses) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 responses required")

    scores_matrix = []
    top_emotions = []
    labels = None  # To store emotion labels from the first response

    for response_text in responses:
        results = call_hf_api(response_text)
        # results is a list of dicts: [{"label": "joy", "score": 0.9}, ...]
        if not results or not isinstance(results, list):
            raise HTTPException(status_code=502, detail="Invalid response from emotion API")

        if labels is None:
            labels = [item['label'] for item in results]

        scores = [item['score'] for item in results]
        max_idx = scores.index(max(scores))
        top_emotions.append(labels[max_idx])
        scores_matrix.append(scores)

    scores_array = np.array(scores_matrix)
    avg_scores = np.mean(scores_array, axis=0)
    avg_scores_dict = {label: round(score, 3) for label, score in zip(labels, avg_scores)}

    # Final emotion is the one that appeared most frequently as the top emotion
    final_emotion = max(set(top_emotions), key=top_emotions.count)

    return ChatResponse(
        average_scores=avg_scores_dict,
        top_emotions=top_emotions,
        final_emotion=final_emotion
    )
