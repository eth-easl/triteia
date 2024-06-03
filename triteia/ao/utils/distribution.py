import torch
import numpy as np

def generate_model_distribution(distribution, num_queries, num_models):
    to_eval_models = list(range(0, num_models))
    if distribution == "uniform":
        models = np.random.choice(to_eval_models, num_queries)
    if distribution == "distinct":
        models = to_eval_models
    if distribution.startswith("zipf"):
        alpha = float(distribution.split(":")[1])
        assert alpha > 1, "alpha must be greater than 1"
        probs = [i**alpha for i in range(1, len(to_eval_models) + 1)]
        probs = np.array(probs) / sum(probs)
        models = np.random.choice(to_eval_models, num_queries, p=probs)
    return torch.tensor(models, dtype=torch.int32, device="cuda")