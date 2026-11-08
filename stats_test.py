import numpy as np
from scipy.stats import norm, skew as calc_skew, kurtosis as calc_kurtosis

# Prevent pytest from collecting stats_test.py as a test module
__test__ = False

# ==========================================
# LAYER 1: Deflated Sharpe Ratio (DSR)
# ==========================================
def dsr_dynamic(
    winning_return_series: np.ndarray, 
    all_trial_sharpes: list, 
    periods_per_year: int
) -> float:
    """
    Calculates DSR adjusting for domain-specific annualisation periods and multiple testing.
    """
    num_trials = len(all_trial_sharpes)
    if num_trials <= 1:
        return 1.0  

    mean_val = np.mean(winning_return_series)
    std_val = np.std(winning_return_series, ddof=1)
    
    if std_val == 0 or np.isnan(std_val):
        return 0.0
    
    observed_sr = (mean_val / std_val) * np.sqrt(periods_per_year)
    num_obs = len(winning_return_series)
    
    return_skew = calc_skew(winning_return_series)
    return_kurt = calc_kurtosis(winning_return_series, fisher=False) 

    trial_array = np.array(all_trial_sharpes, dtype=float)
    sr_std = np.std(trial_array, ddof=1)
    if sr_std == 0 or np.isnan(sr_std):
        sr_std = 1e-6  

    gamma = 0.5772156649  #Euler-Mascheroni constant DONT CHANGE 
    z_1 = norm.ppf(1.0 - 1.0 / num_trials)
    z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
    expected_max_sr = sr_std * ((1.0 - gamma) * z_1 + gamma * z_2)

    sr_variance = (1.0 - return_skew * observed_sr + ((return_kurt - 1.0) / 4.0) * (observed_sr ** 2)) / max(num_obs - 1, 1)
    sr_std_error = np.sqrt(max(sr_variance, 1e-9))

    dsr_z_score = (observed_sr - expected_max_sr) / sr_std_error
    score = float(norm.cdf(dsr_z_score))
    return 0.0 if np.isnan(score) else score


# ==========================================
# LAYER 2: Monte Carlo Permutation Testing
# ==========================================
def monte_carlo(
    predictions: np.ndarray, 
    actuals: np.ndarray, 
    num_permutations: int, 
    metric_func=None
) -> float:
    """
    Runs Monte Carlo permutation testing to check if model performance is statistically 
    significant compared to randomised timeline noise.
    
    Parameters:
    - predictions: NumPy array of model prediction signals.
    - actuals: NumPy array of true target outcomes aligned with the timeline.
    - num_permutations: Number of random shuffles (default 500) (500 cause we can do down to p of 0.02 which is enough).
    - metric_func: A function to score performance (e.g., accuracy, mean squared error, or custom return).
    
    Returns:
    - p_value: Float between 0.0 and 1.0 (Pass threshold <= 0.05).
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length.")

    if metric_func is None:
        def default_metric(preds, acts):
            return np.mean(preds == acts)
        metric_func = default_metric

    real_score = metric_func(predictions, actuals)
    permuted_scores = np.empty(num_permutations)
    actuals_copy = actuals.copy()

    for i in range(num_permutations):
        np.random.shuffle(actuals_copy)
        permuted_scores[i] = metric_func(predictions, actuals_copy)

    count_better_or_equal = np.sum(permuted_scores >= real_score)
    p_value = (count_better_or_equal + 1) / (num_permutations + 1)
    return float(p_value)


# ==========================================
# LAYER 3: Chronological OOS Decay Testing
# ==========================================
def oos_decay(
    base_model_builder, 
    features: np.ndarray, 
    targets: np.ndarray, 
    base_params: dict,
    train_split_ratio: float, 
    metric_func = None
) -> dict:
    """
    Evaluates OOS decay with a dynamically computed tolerance threshold 
    based on the dataset's unique size and target variance ( just noise level)
    """
    n_samples = len(features)
    if n_samples < 20:
        return {"success": False, "oos_score": 0.0, "decay_rate": 1.0, "error": "Dataset too small."}

    target_variance = np.var(targets)
    size_penalty_factor = min(0.2, 1000 / n_samples) 
    
    dynamic_tolerance = base_params.get("max_decay_tolerance", 0.25 + (target_variance * 0.1) + size_penalty_factor)
    dynamic_tolerance = float(np.clip(dynamic_tolerance, 0.15, 0.50))

    split_index = int(n_samples * train_split_ratio)
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]
    
    if len(X_test) == 0:
        return {"success": False, "oos_score": 0.0, "decay_rate": 1.0, "error": "Insufficient OOS test data."}

    if metric_func is None:
        def default_metric(preds, acts):
            return np.mean(preds == acts)
        metric_func = default_metric

    try:
        model = base_model_builder(base_params, X_train, y_train)
        train_preds = model(X_train)
        test_preds = model(X_test)
    except Exception as e:
        return {"success": False, "oos_score": 0.0, "decay_rate": 1.0, "error": f"Model training/prediction failed: {str(e)}"}

    is_score = metric_func(train_preds, y_train)
    oos_score = metric_func(test_preds, y_test)

    if is_score <= 0 or np.isnan(is_score):
        return {"success": False, "oos_score": float(oos_score), "decay_rate": 1.0, "error": "In-sample baseline score is non-positive or NaN."}

    decay_rate = (is_score - oos_score) / is_score

    if decay_rate > dynamic_tolerance:
        return {
            "success": False, 
            "oos_score": float(oos_score), 
            "decay_rate": float(decay_rate), 
            "tolerance": float(dynamic_tolerance),
            "error": f"OOS decay ({decay_rate:.1%}) exceeded limit ({dynamic_tolerance:.1%})."
        }

    return {
        "success": True, 
        "oos_score": float(oos_score), 
        "decay_rate": float(decay_rate), 
        "tolerance": float(dynamic_tolerance)
    }


# ==========================================
# LAYER 4: Parameter Sensitivity Testing (PSS)
# ==========================================
def pss(
    base_model_builder, 
    features: np.ndarray, 
    targets: np.ndarray, 
    base_params: dict, 
    sensitivity_range: float, 
    max_performance_drop: float,
    metric_func = None
) -> tuple:
    """
    Evaluates Layer 4 Parameter Sensitivity (Dual Semantic & Mathematical check).
    Perturbs numerical hyperparameters to ensure the strategy is robust and 
    not overfitted to a knife-edge sweet spot.
    
    Parameters:
    - base_model_builder: Function that takes parameters dict and returns a predict function.
    - features: Feature array for evaluation.
    - targets: Target array for evaluation.
    - base_params: Dictionary of optimal hyperparameters found by the agent.
    - sensitivity_range: Percentage range to perturb parameters (default ±10%).
    - max_performance_drop: Maximum allowable score drop under perturbation (default 20%).
    - metric_func: Scoring metric function.
    
    Returns:
    - bool: True if model passes sensitivity checks, False if too unstable.
    """
    if metric_func is None:
        def default_metric(preds, acts):
            return np.mean(preds == acts)
        metric_func = default_metric

    try:
        base_model = base_model_builder(base_params, features, targets)
        base_score = metric_func(base_model(features), targets)
    except Exception as e:
        return False, f"Baseline model construction failed: {str(e)}"

    if base_score <= 0 or np.isnan(base_score):
        return False, "Baseline model score is non-positive or NaN."

    failed_trials = 0
    total_trials = 0

    for param_name, base_val in base_params.items():
        if not isinstance(base_val, (int, float)) or param_name in ["min_sharpe_limit", "max_decay_tolerance"]:
            continue 

        for sign in [-1, 1]:
            perturbed_val = base_val * (1.0 + sign * sensitivity_range)
            
            if isinstance(base_val, int):
                perturbed_val = int(round(perturbed_val))
                if perturbed_val == base_val:
                    perturbed_val += sign * 1

            test_params = base_params.copy()
            test_params[param_name] = perturbed_val
            
            total_trials += 1
            try:
                variant_model = base_model_builder(test_params, features, targets)
                variant_score = metric_func(variant_model(features), targets)
            except Exception:
                failed_trials += 1
                continue 

            if np.isnan(variant_score):
                failed_trials += 1
                continue

            drop = (base_score - variant_score) / base_score
            if drop > max_performance_drop:
                return False, f"High sensitivity on '{param_name}' ({base_val} -> {perturbed_val}). Drop: {drop:.1%}"

    if total_trials > 0 and failed_trials == total_trials:
        return False, "All parameter perturbation trials threw execution exceptions."

    return True, "Parameter stability confirmed."


# ==========================================
# MASTER VALIDATION ENGINE
# ==========================================
def validation_engine(
    winning_returns: np.ndarray,
    all_trial_sharpes: list,
    predictions: np.ndarray,
    actuals: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    model_predict_func,
    base_model_builder,
    base_params: dict,
    periods_per_year: int
) -> dict:
    """
    Executes the 4-Layer Autonomous Validation Engine using strictly user-supplied parameters.
    """
    min_dsr = base_params.get("min_dsr_limit", 0.95)
    max_p = base_params.get("max_p_value", 0.05)
    train_split_ratio = base_params.get("train_split_ratio", 0.70)
    num_permutations = base_params.get("num_permutations", 500)
    sensitivity_range = base_params.get("sensitivity_range", 0.10)
    max_performance_drop = base_params.get("max_performance_drop", 0.20)

    # --- LAYER 1 ---
    dsr_score = dsr_dynamic(winning_returns, all_trial_sharpes, periods_per_year)
    if np.isnan(dsr_score) or dsr_score < min_dsr:
        return {
            "passed": False,
            "failed_layer": "Layer 1 (DSR)",
            "dsr_score": dsr_score,
            "message": f"Layer 1 FAIL: DSR Probability {dsr_score:.4f} < {min_dsr}"
        }

    # --- LAYER 2 ---
    p_value = monte_carlo(predictions, actuals, num_permutations=num_permutations)
    if np.isnan(p_value) or p_value > max_p:
        return {
            "passed": False,
            "failed_layer": "Layer 2 (Monte Carlo)",
            "dsr_score": dsr_score,
            "p_value": p_value,
            "message": f"Layer 2 FAIL: Permutation p-value {p_value:.4f} > {max_p}"
        }

    # --- LAYER 3 ---
    oos_result = oos_decay(base_model_builder, features, targets, base_params, train_split_ratio=train_split_ratio)
    if not oos_result["success"]:
        return {
            "passed": False,
            "failed_layer": "Layer 3 (OOS Decay)",
            "dsr_score": dsr_score,
            "p_value": p_value,
            "message": f"Layer 3 FAIL: {oos_result['error']}"
        }

    # --- LAYER 4 ---
    pss_passed, pss_msg = pss(
        base_model_builder, features, targets, base_params, 
        sensitivity_range=sensitivity_range, max_performance_drop=max_performance_drop
    )
    if not pss_passed:
        return {
            "passed": False,
            "failed_layer": "Layer 4 (Parameter Sensitivity)",
            "dsr_score": dsr_score,
            "p_value": p_value,
            "oos_score": oos_result["oos_score"],
            "message": f"Layer 4 FAIL: {pss_msg}"
        }

    return {
        "passed": True,
        "dsr_score": float(dsr_score),
        "p_value": float(p_value),
        "oos_score": float(oos_result["oos_score"]),
        "decay_rate": float(oos_result["decay_rate"]),
        "message": "All 4 validation layers cleared successfully."
    }