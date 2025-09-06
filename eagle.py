import itertools
import json
import numpy as np
import os

from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from typing import cast, List, Dict, Literal, Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class ModelScore:
    model_name: str
    score: float

def get_api_embedding(text: str, model_name: Literal["text-embedding-3-large"] = "text-embedding-3-large"):
    response = client.embeddings.create( # type: ignore
        model=model_name,
        input=text
    )
    embedding_vector = response.data[0].embedding # type: ignore
    return cast(List[float], embedding_vector)


def update_elo(rating_a: float, rating_b: float, score_a: float, score_b: float, k: float = 32.0) -> Tuple[float, float]:
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    
    if score_a > score_b:
        actual_a, actual_b = 1.0, 0.0
    elif score_a < score_b:
        actual_a, actual_b = 0.0, 1.0
    else:
        actual_a, actual_b = 0.5, 0.5
    
    new_rating_a = rating_a + k * (actual_a - expected_a)
    new_rating_b = rating_b + k * (actual_b - expected_b)
    
    return new_rating_a, new_rating_b

class EagleRanker:
    def __init__(self, prompts: List[str], model_scores: Dict[str, List[float]], embedding_filepath: str, k: float):
        
        self.prompts = prompts
        self.model_names = list(model_scores.keys())
        self.k = k

        # Get global scores
        self.model_global_scores = {model_name: 100.0 for model_name in model_scores}
        self.model_local_scores = model_scores
        for model_name in model_scores:
            if len(model_scores[model_name]) != len(prompts):
                raise ValueError(f"Model {model_name} has {len(model_scores[model_name])} scores, but there are {len(prompts)} prompts.")
        self.compute_global_scores()

        # Load embeddings
        self.embedding_filepath = embedding_filepath
        if os.path.exists(embedding_filepath):
            self.embeddings = json.load(open(embedding_filepath, 'r'))
            self.embeddings = cast(Dict[str, List[float]], self.embeddings)
        else:
            self.embeddings = dict[str, List[float]]()
            json.dump(self.embeddings, open(embedding_filepath, 'w'))
        self.store_embeddings()
        
        # Turn embeddings into matrix
        self.embedding_matrix = np.array([self.embeddings[prompt] for prompt in self.prompts])
        self.embedding_matrix = self.embedding_matrix / np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)

    def compute_global_scores(self):
        for i in range(len(self.prompts)):
            for model_a, model_b in itertools.combinations(self.model_names, 2):
                score_a = self.model_local_scores[model_a][i]
                score_b = self.model_local_scores[model_b][i]
                updated_score_a, updated_score_b = update_elo(
                    self.model_global_scores[model_a],
                    self.model_global_scores[model_b],
                    score_a,
                    score_b,
                    self.k
                )
                self.model_global_scores[model_a] = updated_score_a
                self.model_global_scores[model_b] = updated_score_b
    
    def store_embeddings(self):
        for prompt in self.prompts:
            if prompt not in self.embeddings:
                self.embeddings[prompt] = self.compute_embedding(prompt)

    def compute_embedding(self, text: str):
        # Get input text embedding
        if text in self.embeddings:
            text_embedding = self.embeddings[text]
        else:
            text_embedding = get_api_embedding(text)
            self.embeddings[text] = text_embedding
            json.dump(self.embeddings, open(self.embedding_filepath, 'w'))
        return text_embedding
    

    def rank(self, text: str, n: int, p: float):
        # Get idxs of top n closest prompts
        text_embedding = np.array(self.compute_embedding(text))
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        cosine_similarities = text_embedding @ self.embedding_matrix.T
        closest_idxs = [int(x) for x in np.argsort(cosine_similarities)[-n:]]

        instance_level_scores = self.model_global_scores.copy()
        for idx in closest_idxs:
            for model_a, model_b in itertools.combinations(self.model_names, 2):
                score_a = self.model_local_scores[model_a][idx]
                score_b = self.model_local_scores[model_b][idx]
                updated_score_a, updated_score_b = update_elo(
                    instance_level_scores[model_a],
                    instance_level_scores[model_b],
                    score_a,
                    score_b,
                    self.k
                )
                instance_level_scores[model_a] = updated_score_a
                instance_level_scores[model_b] = updated_score_b
        # Compute final scores
        final_scores = [
            ModelScore(
                model_name=model,
                score=p*self.model_global_scores[model] + (1-p)*instance_level_scores[model]
            )
            for model in self.model_names
        ]
        return sorted(final_scores, key=lambda x: x.score, reverse=True)

