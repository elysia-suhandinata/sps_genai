from typing import Union
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()
image_classifier = ImageClassifier()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = embedding_model.get_embedding(request.word)
    return {"word": request.word, "embedding": embedding}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = embedding_model.get_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": score}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class = image_classifier.predict(image_bytes)
    return {"class": predicted_class}