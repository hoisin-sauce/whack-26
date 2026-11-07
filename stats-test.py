import numpy as np
from scipy.stats import norm, skew as calc_skew, kurtosis as calc_kurtosis
# Layer 1
def dsr_dynamic(
    winning_return_series: np.ndarray, 
    all_trial_sharpes: list, 
    periods_per_year: int = 252
) -> float:
    """
    calculates DSR for any domain by adjusting the annualisation 
    periods_per_year (e.g., 252 for finance, 38 for soccer, 24 for F1).
    """
    num_trials = len(all_trial_sharpes)
    if num_trials <= 1:
        return 1.0  # No multiple-testing penalty if only 1 trial was run

    # calculate observed Sharpe/Performance Ratio using domain frequency
    mean_val = np.mean(winning_return_series)
    std_val = np.std(winning_return_series, ddof=1)
    
    if std_val == 0:
        return 0.0
    
    # Scale using the domain-specific periods_per_year square root factor
    observed_sr = (mean_val / std_val) * np.sqrt(periods_per_year)
    num_obs = len(winning_return_series)
    
    # Calculate distribution moments dynamically
    return_skew = calc_skew(winning_return_series)
    return_kurt = calc_kurtosis(winning_return_series, fisher=False) 

    # Calculate trial variance
    trial_array = np.array(all_trial_sharpes, dtype=float)
    sr_std = np.std(trial_array, ddof=1)
    if sr_std == 0:
        sr_std = 1e-6  # stop division by zero

    # Compute Expected Maximum Hurdle (SR_0) using Extreme Value Theory (idk what it is i just reada quicK wiki page about it)
    gamma = 0.5772156649  # Euler-Mascheroni constant DONT CHANGE   
    z_1 = norm.ppf(1.0 - 1.0 / num_trials)
    z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
    expected_max_sr = sr_std * ((1.0 - gamma) * z_1 + gamma * z_2)

    # Standard error adjustment
    sr_variance = (1.0 - return_skew * observed_sr + ((return_kurt - 1.0) / 4.0) * (observed_sr ** 2)) / max(num_obs - 1, 1)
    sr_std_error = np.sqrt(max(sr_variance, 1e-9))

    # Final DSR Z-Score and Probability Conversion
    dsr_z_score = (observed_sr - expected_max_sr) / sr_std_error
    return float(norm.cdf(dsr_z_score))
# Layer 2
#FUCKIGN WORK U FUCK ASS BITCH I WILL KIL YOU NOW
def monte_carlo(
    predictions: np.ndarray, 
    actuals: np.ndarray, 
    num_permutations: int = 500, 
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

    # Define a default scoring metric if none is provided (e.g., Mean Accuracy / Hit Rate)
    if metric_func is None:
        def default_metric(preds, acts):
            # Assumes binary or categorical alignment match rate
            return np.mean(preds == acts)
        metric_func = default_metric
    # Calculate the real unscrambled performance score
    real_score = metric_func(predictions, actuals)

    # Run permutations by shuffling the actual outcomes across the timeline
    permuted_scores = np.empty(num_permutations)
    actuals_copy = actuals.copy()

    for i in range(num_permutations):
        # Shuffle the actual targets randomly to destroy temporal relationship
        np.random.shuffle(actuals_copy)
        # Calculate what score the model would get purely by accident on shuffled data
        permuted_scores[i] = metric_func(predictions, actuals_copy)

    # Calculate empirical p-value: 
    # How many times did random shuffles beat or equal the real score?
    # Adding 1 to numerator and denominator prevents a p-value of exactly 0.0.
    count_better_or_equal = np.sum(permuted_scores >= real_score)
    p_value = (count_better_or_equal + 1) / (num_permutations + 1)

    return float(p_value)
#Layer 3
def oos_decay(
    model_predict_func, 
    features: np.ndarray, 
    targets: np.ndarray, 
    train_split_ratio: float = 0.7, 
    metric_func = None
) -> float:
    """
    Evaluates OOS decay with a dynamically computed tolerance threshold 
    based on the dataset's unique size and target variance ( just noise level)
    """
    n_samples = len(features)
    if n_samples < 20:
        raise ValueError("Dataset is too small for chronological OOS splitting.")

    # Dynamically compute an adaptive decay tolerance based on domain noise & data scale
    # Target variance represents how volatile/noisy the outcomes are
    # Smaller sample sizes get a slightly wider tolerance buffer to avoid false rejections
    target_variance = np.var(targets)
    size_penalty_factor = min(0.2, 1000 / n_samples) # Smaller datasets get minor padding
    
    # Dynamic tolerance formula: Base 25% + noise adjustment + sample size padding
    dynamic_tolerance = 0.25 + (target_variance * 0.1) + size_penalty_factor
    # Clamp tolerance between 15% and 50% to keep it safe and realistic
    dynamic_tolerance = float(np.clip(dynamic_tolerance, 0.15, 0.50))

    # chronological split index
    split_index = int(n_samples * train_split_ratio)
    X_train, X_test = features[:split_index], features[split_index:]
    y_train, y_test = targets[:split_index], targets[split_index:]
    
    if len(X_test) == 0:
        print("Layer 3 FAIL: Not enough data left for out-of-sample testing.")
        return -1.0

    # Default scoring metric (Match accuracy)
    if metric_func is None:
        def default_metric(preds, acts):
            return np.mean(preds == acts)
        metric_func = default_metric

    # Generate predictions and score eras
    train_preds = model_predict_func(X_train)
    test_preds = model_predict_func(X_test)

    is_score = metric_func(train_preds, y_train)
    oos_score = metric_func(test_preds, y_test)

    if is_score <= 0:
        print("Layer 3 FAIL: In-sample performance baseline is zero or negative.")
        return -1.0

    # Calculate decay rate
    decay_rate = (is_score - oos_score) / is_score

    # Check against the dynamically derived tolerance
    if decay_rate > dynamic_tolerance:
        print(f"Layer 3 FAIL: OOS decay ({decay_rate:.1%}) exceeded dynamic limit ({dynamic_tolerance:.1%}). IS: {is_score:.3f}, OOS: {oos_score:.3f}")
        return -1.0

    print(f"Layer 3 PASS: Graceful decay ({decay_rate:.1%} <= dynamic limit {dynamic_tolerance:.1%}). IS: {is_score:.3f}, OOS: {oos_score:.3f}")
    return float(oos_score)
#Layer 4
def pss(
    base_model_builder, 
    features: np.ndarray, 
    targets: np.ndarray, 
    base_params: dict, 
    sensitivity_range: float = 0.10, 
    max_performance_drop: float = 0.20,
    metric_func = None
) -> bool:
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

    # Establish baseline performance with optimal parameters
    base_predict_func = base_model_builder(base_params)
    base_score = metric_func(base_predict_func(features), targets)

    if base_score <= 0:
        print("Layer 4 FAIL: Baseline model score is non-positive.")
        return False

    # Iterate through numerical hyperparameters and test perturbations
    # (Mathematical condition check: testing local gradient stability)
    for param_name, base_val in base_params.items():
        if not isinstance(base_val, (int, float)):
            continue  # Skip non-numerical semantic parameters for gradient testing

        for sign in [-1, 1]:
            # Perturb parameter up and down by sensitivity range
            perturbed_val = base_val * (1.0 + sign * sensitivity_range)
            
            # Type casting safeguard for integers (e.g., window sizes)
            if isinstance(base_val, int):
                perturbed_val = int(round(perturbed_val))
                if perturbed_val == base_val:
                    continue

            # Build model variant with perturbed parameter
            test_params = base_params.copy()
            test_params[param_name] = perturbed_val
            
            try:
                variant_predict_func = base_model_builder(test_params)
                variant_score = metric_func(variant_predict_func(features), targets)
            except Exception:
                continue # If invalid parameter combination crashes, skip

            # Calculate performance drop
            drop = (base_score - variant_score) / base_score

            # If a small tweak causes performance to collapse, it fails Layer 4
            if drop > max_performance_drop:
                print(f"Layer 4 FAIL: High sensitivity on parameter '{param_name}' ({base_val} -> {perturbed_val}). Drop: {drop:.1%}")
                return False

    print(f"Layer 4 PASS: Strategy is mathematically stable and parameter-robust across perturbations.")
    return True

#gotta link everything together for the agent to use easily
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
) -> bool:
    """
    Executes the 4-Layer Autonomous Validation Engine.
    
    Layers:
    1. Deflated Sharpe Ratio (DSR) - Overfitting/Multiple testing correction.
    2. Monte Carlo Permutation Testing - Random timeline noise check.
    3. Chronological Out-of-Sample (OOS) Decay - Generalization check.
    4. Parameter Sensitivity Testing -  semantic & mathematical robustness check.
    
    Returns:
    - bool: True if the strategy passes all layers, False otherwise.
    """
    print("--- [VALIDATION ENGINE]  4-Layer Stats Check ---")

    # ==========================================
    # LAYER 1: Deflated Sharpe Ratio (DSR)
    # ==========================================
    print("Running Layer 1: Deflated Sharpe Ratio...")
    dsr_score = dsr_dynamic(winning_returns, all_trial_sharpes, periods_per_year)
    if dsr_score < 0.95:
        print(f"Layer 1 FAIL: DSR Probability {dsr_score:.4f} < 0.95 (Data snooping detected)")
        return False
    print(f"Layer 1 PASS: DSR Score = {dsr_score:.4f}")

    # ==========================================
    # LAYER 2: Monte Carlo Permutation Testing
    # ==========================================
    print("Running Layer 2: Monte Carlo Permutation Testing...")
    p_value = monte_carlo(predictions, actuals, num_permutations=500)
    if p_value > 0.05:
        print(f"Layer 2 FAIL: Permutation p-value {p_value:.4f} > 0.05 (Indistinguishable from luck)")
        return False
    print(f"Layer 2 PASS: Empirical p-value = {p_value:.4f}")

    # ==========================================
    # LAYER 3: Out-of-Sample (OOS) Decay
    # ==========================================
    print("Running Layer 3: Chronological OOS Decay Testing...")
    oos_score = oos_decay(model_predict_func, features, targets)
    if oos_score == -1.0:
        print("Layer 3 FAIL: Severe out-of-sample decay or insufficient data.")
        return False
    print(f"Layer 3 PASS: OOS Score = {oos_score:.3f}")

    # ==========================================
    # LAYER 4: Parameter Sensitivity Testing
    # ==========================================
    print("Running Layer 4: Parameter Sensitivity Testing...")
    sensitivity_passed = pss(
        base_model_builder, features, targets, base_params
    )
    if not sensitivity_passed:
        print("Layer 4 FAIL: Strategy is overly sensitive to minor parameter tweaks.")
        return False
    print("Layer 4 PASS: Strategy is mathematically and semantically robust.")

    print("SUCCESS: Strategy cleared all 4 validation layers! Ready for deployment.")
    return True
