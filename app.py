from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_API_TOKEN = "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"  # Replace with your actual API key

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

class ChatRequest(BaseModel):
    responses: list[str]

class ChatResponse(BaseModel):
    average_scores: dict
    top_emotions: list
    final_emotion: str

def call_hf_api(text):
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

@app.post("/analyze", response_model=ChatResponse)
def analyze_emotions(chat_request: ChatRequest):
    responses = chat_request.responses
    if len(responses) != 6:
        return {"error": "Exactly 6 responses required"}

    scores_matrix = []
    top_emotions = []

    for response_text in responses:
        results = call_hf_api(response_text)
        # results is a list of dicts: [{"label": "joy", "score": 0.9}, ...]
        scores = [item['score'] for item in results]
        labels = [item['label'] for item in results]

        # Top emotion with highest score
        max_idx = scores.index(max(scores))
        top_emotions.append(labels[max_idx])
        scores_matrix.append(scores)

    # Average the scores across all responses
    import numpy as np
    scores_array = np.array(scores_matrix)
    avg_scores = np.mean(scores_array, axis=0)
    emotion_labels = [item['label'] for item in results]

    avg_scores_dict = {label: round(score, 3) for label, score in zip(emotion_labels, avg_scores)}

    # Simple logic for final emotion: pick the top emotion that appears most in top_emotions
    final_emotion = max(set(top_emotions), key=top_emotions.count)

    return ChatResponse(
        average_scores=avg_scores_dict,
        top_emotions=top_emotions,
        final_emotion=final_emotion
    )
