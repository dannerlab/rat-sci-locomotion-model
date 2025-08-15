from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import yaml
import xgboost as xgb
import shap
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import io
import contextlib
from copy import copy


class CustomCDumper(yaml.CDumper):
    """ Custom YAML CDumper that forces sequences (lists) to be in flow style []. """
# Function to check if all_ elements in a list are primitive (str, int, float, bool, None)
def is_primitive_list(lst):
    return all(isinstance(item, (str, int, float, bool, type(None))) for item in lst)
# Custom representer for lists
def represent_list(dumper, data):
    flow_style = is_primitive_list(data)  # Use `[]` only for primitive lists
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)
# Register the custom representer for lists
CustomCDumper.add_representer(list, represent_list)

def permutation_importance(model, X, y, n_repeats=10, random_state=0):
    baseline = r2_score(y, model.predict(X))
    importances = np.zeros(X.shape[1])

    rng = np.random.RandomState(random_state)
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            scores.append(r2_score(y, model.predict(X_perm)))
        importances[col] = baseline - np.mean(scores)
    return importances

def shap_importance(model, X):
    explainer = shap.Explainer(model)
    shap_vals = explainer(X)
    var_expl = np.mean(shap_vals.values**2, axis=0)
    percent_var = 100 * var_expl / var_expl.sum()
    return percent_var

def shap_interaction_importance(model, X,y, names):
    explainer = shap.TreeExplainer(model)
    shap_int = explainer.shap_interaction_values(X)   # if multi-output, take shap_int[k]

    # 2) drop self-interactions
    N, d, _ = shap_int.shape
    upper = np.triu_indices(d, k=1)

    # 3) variance of surrogate target across the same N runs
    var_y = np.var(y, ddof=1)

    # 4) global 2nd-order index S_ij  (Sobol‐style, normalised)
    S2 = np.zeros((d, d))
    for i, j in zip(*upper):
        S2[i, j] = 2 * np.mean(np.abs(shap_int[:, i, j])) / var_y
        S2[j, i] = S2[i, j]            # for symmetry

    # 5) nice list of strongest interactions
    pairs  = [(i, j, S2[i, j]) for i, j in zip(*upper)]
    top10  = sorted(pairs, key=lambda p: p[2], reverse=True)[:10]
    for i, j, s in top10:
        print(f"{names[i]} × {names[j]}  :  2nd-order Sobol ≈ {s:5.3f}")

def shap_first_second(model, X,y):
    # explainer on *tree_path_dependent* mode
    expl = shap.TreeExplainer(model)                # no feature_dependence="independent"
    sh_int = expl.shap_interaction_values(X)  # shape (N, d, d)

    N, d, _   = sh_int.shape
    var_y_tot = np.var(y, ddof=1)                 # total surrogate variance

    # 1. FIRST-ORDER  S_i = Var[ main effect ] / Var[Y]
    main = sh_int[:, range(d), range(d)]                # N × d
    S1   = np.var(main, axis=0, ddof=1) / var_y_tot
    first_order_total = S1.sum()

    # 2. SECOND-ORDER  S_ij = 2·Var[ interaction(i,j) ] / Var[Y]
    S2_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            inter_var      = np.var(sh_int[:, i, j], ddof=1)
            S2_matrix[i,j] = S2_matrix[j,i] = 2 * inter_var / var_y_tot
    second_order_total = S2_matrix[np.triu_indices(d, k=1)].sum()

    print(f"First-order total  : {first_order_total:6.3f}")
    print(f"Second-order total : {second_order_total:6.3f}")
    print(f"Residual (≥3rd)    : {1 - first_order_total - second_order_total:6.3f}")
    #import seaborn as sns
    #upper = np.triu_indices(d, k=1)
    #thr = np.percentile(S2_matrix[upper], 90)          # top 10 %
    #mask = S2_matrix < thr
    #sns.heatmap(S2_matrix, mask=mask, cmap="Reds", annot=False,
    #            fmt=".2f", cbar_kws=dict(label="S₂"))
    #plt.title("Second-order Sobol indices (top 10 %)")
    #plt.show()
    #print(f"Fraction of variance from first order      : {first_order_total: .3f}")
    #print(f"Fraction of variance from second order     : {second_order_total: .3f}")
    #print(f"Unexplained (≥third-order interactions)    : {1 - first_order_total - second_order_total: .3f}")

def rank(vars, importances):
    importances = np.array(importances)
    importances = importances / np.sum(importances)  # Normalize importances
    importances = importances * 100  # Convert to percentage
    ranking = sorted(zip(vars, importances), key=lambda p: p[1], reverse=True)
    return ranking

def partial_dependence_plot(model, X, var_index, n_points=100,names=None):
    pdp = partial_dependence(model, X, features=var_index, grid_resolution=n_points)
    fig = plt.figure(figsize=(8, 6))
    XX, YY = np.meshgrid(pdp["grid_values"][0], pdp["grid_values"][1])
    Z = pdp.average[0].T
    ax = fig.add_subplot(projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor="k")
    if names is not None:
        ax.set_xlabel(names[var_index])
        ax.set_ylabel('divergence')

    # pretty init view
    ax.view_init(elev=22, azim=122)
    clb = plt.colorbar(surf, pad=0.08, shrink=0.6, aspect=10)
    clb.ax.set_title("Partial\ndependence")
    plt.show()

def partial_dependence_plot2(model, X,var_indices, n_points=100,names=None):
    common_params = {
        "subsample": 50,
        "n_jobs": 2,
        "grid_resolution": n_points,
        "random_state": 0,
    }
    features_info = {
        "features": var_indices,
        #"feature_names": [names[i] if isinstance(i, int) else [names[j]] for j in i for i in var_indices],
        "kind": "average",
    }
    print(features_info)
    _, ax = plt.subplots(ncols=3, figsize=(10, 4), constrained_layout=True)
    
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        **features_info,
        ax=ax,
        **common_params,
    )
    
    _ = display.figure_.suptitle(
        "1-way vs 2-way of numerical PDP using gradient boosting", fontsize=16
    )
    plt.show()

def plot_pair_marginal(kde, dims, limits, n_grid=25, n_mc=100):
    d1, d2         = dims
    (a1, b1), (a2, b2) = limits          # axis limits for the grid
    x1  = np.linspace(a1, b1, n_grid)
    x2  = np.linspace(a2, b2, n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    # Monte-Carlo marginalisation over the remaining dimensions
    mc = kde.sample(n_mc, random_state=1)                # (n_mc, 4)
    logdens = np.empty((n_grid, n_grid))

    for i in range(n_grid):
        print(i)
        for j in range(n_grid):
            probe      = mc.copy()
            probe[:,d1] = X1[i,j]
            probe[:,d2] = X2[i,j]
            # log-sum-exp average converts to marginal log-density
            logdens[i,j] = np.logaddexp.reduce(
                kde.score_samples(probe)
            ) - np.log(n_mc)

    plt.figure(figsize=(4,3.5))
    cs = plt.contourf(X1, X2, np.exp(logdens), 40)
    plt.colorbar(cs, fraction=0.046)      # probability density
    plt.xlabel(f"dim {d1}")
    plt.ylabel(f"dim {d2}")
    plt.title("Monte-Carlo marginal KDE")
    # Improved visualization
    cs = plt.contour(X1, X2, np.exp(logdens), colors='k', linewidths=0.7)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    csf = plt.contourf(X1, X2, np.exp(logdens), 40, cmap="viridis")
    plt.colorbar(csf, fraction=0.046, label="Probability density")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--sconfig", dest="s_config_fn",default=None)
    parser.add_option("-r", "--run", dest="i_run",default=-1, type="int")
    options, args = parser.parse_args()
    if options.s_config_fn is None:
        print("Please provide a sensitivity configuration file with -s or --sconfig")
        exit(1)


    with open(options.s_config_fn, 'r') as file:
        s_config = yaml.safe_load(file)
    data_file_fn = s_config['run'][options.i_run]['out_filename']+'.npz'
    data = np.load(data_file_fn, allow_pickle=True)

    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    variable_groups = s_config["variable_groups"]
    par_names = data['par_names']

    #variables = [ v for group in variable_groups.values() for v in group ]
    #group_lens = [len(group) for group in variable_groups.values()]
    #update_method = s_config['update_method']

    #if update_method == 'variable_groups':
    #    N = len(variable_groups)
    #    par_names = [f"{group_name}" for group_name, group in variable_groups.items() ]
    #elif update_method == 'variables':
    #    N = len(variables)
    #    par_names = variables
    #else:
    #    raise ValueError(f"Unknown update method: {update_method}")
    #import IPython; IPython.embed()
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)

    if False: # test to check sensitivity when values were only decreased
        y_train = y_train[~np.any(X_train>1.,axis=1),:]
        X_train = X_train[~np.any(X_train>1.,axis=1),:]
        
        y_val = y_val[~np.any(X_val>1.,axis=1),:]
        X_val = X_val[~np.any(X_val>1.,axis=1),:]
    scoring_fns = copy(s_config['scoring_fn'])
    if not isinstance(scoring_fns, list):
        scoring_fns = [scoring_fns]
    #import IPython; IPython.embed()
    if y_train.shape[1] == 4 + len(scoring_fns):
        cov_train = np.std(y_train[:,-4:],axis=1)/np.mean(y_train[:,-4:],axis=1)
        cov_val = np.std(y_val[:,-4:],axis=1)/np.mean(y_val[:,-4:],axis=1)

        lr_ratio_train = y_train[:,1]/y_train[:,2]
        lr_ratio_val = y_val[:,1]/y_val[:,2]
        y_train = y_train[:,:-2]
        y_train[:, -2] = cov_train
        y_train[:, -1] = lr_ratio_train
        y_val = y_val[:,:-2]
        y_val[:, -2] = cov_val
        y_val[:, -1] = lr_ratio_val

        scoring_fns.append('cov')
        scoring_fns.append('lr_ratio')
    #import IPython; IPython.embed()
    
    for score, i in zip(scoring_fns, range(y_val.shape[1])):
        print(f"Analyzing output variable {i+1}/{y_val.shape[1]} {score}")
        y_train_i = y_train[:, i]
        y_val_i = y_val[:, i]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=10000,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=20,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=10.0,
            n_jobs=-1,
            early_stopping_rounds=100, # Stop if no improvement for 10 rounds
            eval_metric='rmse'
        )
        model.fit(
            X_train, y_train_i,
            eval_set=[(X_val, y_val_i)],
            verbose=False
        )
        
        r2_train = r2_score(y_train_i, model.predict(X_train))
        r2_val = r2_score(y_val_i, model.predict(X_val))

        output = io.StringIO()
        
        output.write(f"train R²: {r2_train}\n")
        output.write(f"val   R²: {r2_val}\n")

        ranked = rank(par_names, shap_importance(model, X_train))

        output.write("\n\n")
        output.write("Variable         | Importance (%)\n")
        output.write("-----------------|---------------\n")
        for name, imp in ranked:
            output.write(f"{name:<16} | {imp:6.2f}\n")
        output.write("\n\n")

        # Capture output of shap_first_second

        with contextlib.redirect_stdout(output):
            shap_first_second(model, X_train, y_train_i)

        result_str = output.getvalue()
        print(result_str)

        # Ensure the score key exists in the YAML structure
        if score not in s_config['run'][options.i_run]:
            s_config['run'][options.i_run][score] = {}
        s_config['run'][options.i_run][score]['xgboost_r2'] = {'train': float(r2_train), 'val': float(r2_val)}
        # Convert numpy types to native Python types for YAML serialization
        s_config['run'][options.i_run][score]['shap_importance'] = {str(k): float(v) for k, v in ranked}
        s_config['run'][options.i_run][score]['metadata'] = {'n_samples': X_train.shape[0], 
                                                    'n_samples_val': X_val.shape[0], 
                                                    'range':[float(np.min(X_train)), float(np.max(X_train))],
                                                    'y_range':[float(np.min(y_train_i)), float(np.max(y_train_i))]}
        #import IPython; IPython.embed()
    with open(options.s_config_fn, 'w') as file:
        yaml.dump(s_config, file,default_flow_style=False, sort_keys=False, Dumper=CustomCDumper)


    
