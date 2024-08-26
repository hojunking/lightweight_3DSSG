import numpy as np
import pandas as pd

def calculate_performance_loss(baseline_acc, current_acc):
    """Calculate normalized performance loss."""
    #print('check: baseline_acc: ', baseline_acc, 'current_acc: ', current_acc)
    return (current_acc - baseline_acc) / baseline_acc if baseline_acc != 0 else 0

def calculate_differential_losses(pruning_results, baseline):
    """
    Calculate the differential of performance losses across pruning ratios.
    Arguments:
    pruning_results -- list of tuples, where each tuple contains (pruning_ratio, obj_acc, rel_acc, triplet_acc)
    baseline -- tuple of (baseline_obj_acc, baseline_rel_acc, baseline_triplet_acc)
    """
    # Sorting by pruning ratios for accurate differential calculation
    sorted_results = sorted(pruning_results, key=lambda x: x[0])
    ratios = [x[0] for x in sorted_results]
    losses = [[calculate_performance_loss(baseline[i], x[i + 1]) for i in range(3)] for x in sorted_results]
    
    # Calculating differentials of losses
    diffs = -np.diff(np.array(losses), axis=0)
    diffs = np.vstack([diffs, diffs[-1]])  # Duplicate last differential as an approximation for the last point
    return ratios, losses, diffs

def calculate_model_score(losses, diffs, weights=(1, 1, 5), mu=5, pruning_ratio=0, detailed=False):
    """Calculate scores incorporating the absolute values of losses and their differentials, adjusted by lambda."""
    adjusted_losses = losses
    adjusted_diffs = diffs
    #print(f'pruning ratio: {pruning_ratio}, adjusted losses: {adjusted_losses}, adjusted diffs: {adjusted_diffs}')
    weighted_diff_loss = sum(weights[i] * (adjusted_diffs[i] + adjusted_losses[i]) for i in range(3))
    score = weighted_diff_loss + mu * pruning_ratio
    
    if detailed:
        return score, weighted_diff_loss
    return score

def calculate_average_loss(pruning_results, baseline):
    losses = np.array([
        [calculate_performance_loss(baseline[i], x[i + 1]) for i in range(3)]
        for x in pruning_results
    ])
    average_losses = np.mean(np.abs(losses), axis=0)
    return average_losses

base_dir = ''

file_path1 = base_dir + ''
file_path2 = base_dir +''
file_path3 = base_dir +''
file_paht4 = base_dir + ''

# CSV 파일 로드
df1 = pd.read_csv(file_path1).sort_values(by='Pruning Ratio')
df2 = pd.read_csv(file_path2).sort_values(by='Pruning Ratio')
df3 = pd.read_csv(file_path3).sort_values(by='Pruning Ratio')
df4 = pd.read_csv(file_paht4).sort_values(by='Pruning Ratio')


# 필요한 데이터를 추출하여 리스트로 변환
sgfn_pruning_results = [
    (row['Pruning Ratio'], row['3d obj Acc@1'], row['3d rel Acc@1'], row['3d triplet Acc@50'])
    for index, row in df1.iterrows()
]
sgfn_baseline = (
    df1['3d obj Acc@1'].iloc[0],
    df1['3d rel Acc@1'].iloc[0],
    df1['3d triplet Acc@50'].iloc[0]
)
attn_sgfn_pruning_results = [
    (row['Pruning Ratio'], row['3d obj Acc@1'], row['3d rel Acc@1'], row['3d triplet Acc@50'])
    for index, row in df2.iterrows()
]
attn_sgfn_baseline = (
    df2['3d obj Acc@1'].iloc[0],
    df2['3d rel Acc@1'].iloc[0],
    df2['3d triplet Acc@50'].iloc[0]
)

sgpn_pruning_results = [
    (row['Pruning Ratio'], row['3d obj Acc@1'], row['3d rel Acc@1'], row['3d triplet Acc@50'])
    for index, row in df3.iterrows()
]
sgpn_baseline = (
    df3['3d obj Acc@1'].iloc[0],
    df3['3d rel Acc@1'].iloc[0],
    df3['3d triplet Acc@50'].iloc[0]
)

vlsat_pruning_results = [
    (row['Pruning Ratio'], row['3d obj Acc@1'], row['3d rel Acc@1'], row['3d triplet Acc@50'])
    for index, row in df4.iterrows()
]
vlsat_baseline = (
    df4['3d obj Acc@1'].iloc[0],
    df4['3d rel Acc@1'].iloc[0],
    df4['3d triplet Acc@50'].iloc[0]
)
print("SGFN Pruning Results:", sgfn_pruning_results)
print("SGFN Baseline:", sgfn_baseline)
print("Attention SGFN Pruning Results:", attn_sgfn_pruning_results)
print("Attention SGFN Baseline:", attn_sgfn_baseline)
print("SGPN Pruning Results:", sgpn_pruning_results)
print("SGPN Baseline:", sgpn_baseline)
print("VLSAT Pruning Results:", vlsat_pruning_results)
print("VLSAT Baseline:", vlsat_baseline)

models_data = {
    "SGPN": {
        "baseline": sgpn_baseline,
        "pruning_results": sgpn_pruning_results
    },
    "SGFN": {
        "baseline": sgfn_baseline,
        "pruning_results": sgfn_pruning_results
    },
    "Attn + SGFN": {
        "baseline": attn_sgfn_baseline,
        "pruning_results": attn_sgfn_pruning_results
    },
    "VLSAT": {
        "baseline": vlsat_baseline,
        "pruning_results": vlsat_pruning_results
    }
    
}

def display_scores_limited_ratio(models_data, base_weights=(1, 1, 1), mu=0.2, max_ratio=0.7):
    results = {}
    for model_name, data in models_data.items():
        baseline = data["baseline"]
        pruning_results = data["pruning_results"]
        
        # Pruning ratio 0.7 이하로 필터링
        pruning_results = [result for result in pruning_results if result[0] <= max_ratio]
        
        if not pruning_results or len(pruning_results) < 2:
            print(f"Not enough data to calculate scores for {model_name}.")
            continue
        
        # 가중치 계산 (자동 계산된 가중치 * 임의의 가중치)
        average_losses = calculate_average_loss(pruning_results, baseline)
        auto_weights = 1 / average_losses
        auto_weights = auto_weights / np.sum(auto_weights)  # 가중치 정규화
        
        # 임의의 가중치와 자동 계산된 가중치를 곱함
        final_weights = auto_weights * np.array(base_weights)
        final_weights = final_weights / np.sum(final_weights)  # 최종 가중치 정규화
        
        ratios, losses, diffs = calculate_differential_losses(pruning_results, baseline)
        
        # Store detailed score components for each pruning ratio
        detailed_scores = []
        scores = []
        for idx in range(len(ratios)):
            score, components = calculate_model_score(losses[idx], diffs[idx], final_weights, mu, ratios[idx], detailed=True)
            scores.append(score)
            detailed_scores.append({
                "ratio": ratios[idx],
                "weighted_diff_loss": components,
                "weighted_ratio": mu * ratios[idx],
                "total_score": score
            })
        
        max_score_index = np.argmax(scores)
        best_ratio = ratios[max_score_index]
        best_score = scores[max_score_index]

        results[model_name] = {
            "best_ratio": best_ratio,
            "best_score": best_score,
            "all_scores": scores,
            "detailed_scores": detailed_scores
        }

        print(f"Scores for {model_name}:")
        print(f"Calculated Weights: {final_weights}")
        for entry in detailed_scores:
            print(f"  Pruning Ratio {entry['ratio']}, Weighted Diff-Loss: {entry['weighted_diff_loss']}, "
                  f"Weighted Ratio: {entry['weighted_ratio']:.3f}, Total Score: {entry['total_score']:.3f}")
        print(f"Best Pruning Ratio: {best_ratio}, Highest Score: {best_score:.3f}\n")

    return results

# 예제 실행 (max_ratio를 0.7로 설정하여 pruning ratio 0.7 이하의 데이터만 사용)
results = display_scores_limited_ratio(models_data, max_ratio=0.75)
