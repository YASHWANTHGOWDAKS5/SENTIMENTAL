from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import numpy as np

app = FastAPI(title="Emotion Chatbot API")

# Load model once at startup
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

class ChatRequest(BaseModel):
    responses: list[str]

class ChatResponse(BaseModel):
    average_scores: dict[str, float]
    top_emotions: list[str]
    final_emotion: str

def determine_dominant_emotion(average_scores, top_emotions):
    # Customize order of preference for emotions
    emotion_order = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
    emotion_counts = {}
    for emo in top_emotions:
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    combined_score = {}
    for emo in average_scores:
        freq = emotion_counts.get(emo, 0)
        avg = average_scores[emo]
        if emo == "neutral":
            boost = avg * 0.7 + 0.05 * freq  # penalize neutral slightly
        else:
            boost = avg + 0.1 * freq
        combined_score[emo] = boost

    sorted_emotions = sorted(combined_score.items(), key=lambda x: (-x[1], emotion_order.index(x[0])))
    return sorted_emotions[0][0]

@app.get("/")
def root():
    return {"message": "Emotion Chatbot API is running."}

@app.post("/analyze", response_model=ChatResponse)
def analyze_emotions(chat_request: ChatRequest):
    responses = chat_request.responses
    if len(responses) != 6:
        raise HTTPException(status_code=400, detail="Exactly 6 responses required.")

    scores_matrix = []
    top_emotions = []

    for response in responses:
        result = classifier(response)[0]
        scores = [score['score'] for score in result]
        top_emotion = result[np.argmax(scores)]['label']
        top_emotions.append(top_emotion)
        scores_matrix.append(scores)

    scores_array = np.array(scores_matrix)
    avg_scores = np.mean(scores_array, axis=0)
    avg_scores_dict = {label: round(score, 3) for label, score in zip(emotion_labels, avg_scores)}

    final_emotion = determine_dominant_emotion(avg_scores_dict, top_emotions)

    return ChatResponse(
        average_scores=avg_scores_dict,
        top_emotions=top_emotions,
        final_emotion=final_emotion
    )
