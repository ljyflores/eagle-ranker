from eagle import ModelScore
from typing import List, Dict

def compute_accuracy(
    model_ground_truth_scores: Dict[str, List[float]],
    ranker_scores: List[List[ModelScore]],
    print_flags: bool = False,
    ):
    correct = 0
    for i in range(len(ranker_scores)):
        # Get ground truth scores for this prompt
        gt_scores = {model: model_ground_truth_scores[model][i] for model in model_ground_truth_scores}
        # If the ground truth scores are all the same, evaluate if the ranker also ranked them the same
        if len(set(gt_scores.values())) == 1:
            if len(set([item.score for item in ranker_scores[i]])) == 1:
                correct += 1
                if print_flags: print("Same scores and same rank")
            else:
                if print_flags: print("Same scores but different rank")
        else:
        # Otherwise, evaluate if the ranker got the top model correct
        # Get ranked models
            best_ranker_model = sorted(ranker_scores[i], key=lambda x: x.score, reverse=True)[0]
            best_ground_truth_model = sorted(gt_scores, key=gt_scores.get, reverse=True)[0]
            if print_flags: print(best_ranker_model.model_name, best_ground_truth_model)
            if best_ranker_model.model_name == best_ground_truth_model:
                correct += 1
    return correct / len(ranker_scores)