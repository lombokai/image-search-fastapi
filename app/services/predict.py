import os

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_NAME, MODEL_PATH
from loguru import logger


class MachineLearningModelHandlerScore(object):
    model = None

    # Similarity = (A.B) / (||A||.||B||)
    

    @classmethod
    def predict(cls, input, load_wrapper=None, method="predict"):
        clf = cls.get_model(load_wrapper)

        def cosine(vec_a, vec_b):
            dot = sum(a*b for a, b in zip(vec_a, vec_b))
            norm_a = sum(a*a for a in vec_a) ** 0.5
            norm_b = sum(b*b for b in vec_b) ** 0.5

            # Cosine similarity
            cos_sim = dot / (norm_a * norm_b)
            return cos_sim
        
        result = []
        input = input.flatten()
        for dct in clf:
            result.append({list(dct.keys())[0]: cosine(input, list(dct.values())[0])})
        ids = []
        sorted_res = sorted(result, key=lambda item: list(item.values())[0])[:10]
        return sorted_res
    
    @classmethod
    def get_model(cls, load_wrapper):
        if cls.model is None and load_wrapper:
            cls.model = cls.load(load_wrapper)
        return cls.model

    @staticmethod
    def load(load_wrapper):
        model = None
        if MODEL_PATH.endswith("/"):
            path = f"{MODEL_PATH}{MODEL_NAME}"
        else:
            path = f"{MODEL_PATH}/{MODEL_NAME}"
        if not os.path.exists(path):
            message = f"Machine learning model at {path} not exists!"
            logger.error(message)
            raise FileNotFoundError(message)
        with open(path, 'rb') as handle:
            model = load_wrapper(handle)
        if not model:
            message = f"Model {model} could not load!"
            logger.error(message)
            raise ModelLoadException(message)
        return model
