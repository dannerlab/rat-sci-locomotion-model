import pandas as pd
import numba
import numpy as np
import CPGNetworkSimulator.tools.py_simulator as nsim 
from sklearn.neighbors import KernelDensity
from optparse import OptionParser
from yaml_sim import update_sim_from_yaml,yamlload
from scipy.special import rel_entr
import os
from datetime import datetime
from scoop import futures 
from tqdm import tqdm
import yaml 
import ot
import time

@numba.njit()
def c_dist2(x,y): 
    n=6
    return np.sum(np.abs(np.angle(np.exp(x[:n]*1j)/np.exp(y[:n]*1j))))+np.abs(x[n:]-y[n:])

@numba.njit()
def c_dist3(x,y): 
    n=6
    tp=np.abs(x[:n]-y[:n])
    tp[tp>np.pi]=tp[tp>np.pi]-2*np.pi
    return np.sum(np.abs(tp))+np.sum(np.abs(x[n:]-y[n:]))


dist_metric = c_dist3

class CustomCDumper(yaml.CDumper):
    """ Custom YAML CDumper that forces sequences (lists) to be in flow style []. """
def is_primitive_list(lst):
    return all(isinstance(item, (str, int, float, bool, type(None))) for item in lst)
def represent_list(dumper, data):
    flow_style = is_primitive_list(data)  # Use `[]` only for primitive lists
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow_style)
CustomCDumper.add_representer(list, represent_list)

def clean_times(times,thr=0.02):
    times_ = []
    for limb in range(4):
        times_l = times[times[:, 1] == limb,:]
        in_rem = np.where(np.diff(times_l[:,0]) < thr)[0]
        in_rem = np.delete(in_rem, np.where(np.diff(in_rem) == 1)[0] + 1)
        times_l[in_rem, 3] = -1
        times_l[in_rem+1, 3] = -1
        times_l = times_l[times_l[:,3] != -1]
        times_l = times_l[times_l[:,0].argsort()]
        times_l[times_l[:,3]==0,2] = np.arange(len(times_l[times_l[:,3]==0,2]))
        times_l[times_l[:,3]==1,2] = np.arange(len(times_l[times_l[:,3]==1,2]))
        times_.append(times_l)
    times_out = np.concatenate(times_)
    times_out = times_out[times_out[:,0].argsort()]
    return times_out

def create_times_df(times):
    df_times = pd.DataFrame({})
    for i in range(4):
        onsets = times[np.logical_and(times[:, 1] == i, times[:, 3] == 1)][:, 0]
        offsets = times[np.logical_and(times[:, 1] == i, times[:, 3] == 0)][:, 0]

        while onsets[0]>offsets[0]:
            offsets = offsets[1:]
        while len(offsets) > len(onsets):
            offsets = offsets[:-1]
        if len(onsets) > len(offsets): 
            onsets = onsets[:-1]
        df_ = pd.DataFrame({'leg':np.ones((len(onsets),),dtype=np.int32)*i,
                            'onset': onsets,
                            'offset': offsets,
                            'stance_dur': (offsets-onsets),
                            'swing_dur': np.concatenate(((onsets[1:]-offsets[:-1]),[np.nan]))
                            })
        
        df_times = pd.concat([df_times,df_],ignore_index=True)
    df_times['midstance']=(df_times.onset+df_times.offset)*0.5
    df_times = df_times.sort_values(by='onset')
    df_times.reset_index(drop=True,inplace=True)
    return df_times

def calc_sup(df):
    now = 0
    res = list()
    stack = list()
    for i, (index,row) in enumerate(df.iterrows()):
        new = row
        while(True):
            if (len(stack)>0) and (stack[0][0].offset < new.onset):
                res.append((stack[0][0].offset,now,stack[0][0].leg,stack[0][1],'off'))
                now -=1
                stack.pop(0)
                continue
            else:
                res.append((new.onset,now,new.leg,index,'on'))
                now += 1
                break
        stack.append((new,index))
        stack = sorted(stack,key=lambda x:x[0].offset)
    res.append((new.offset,0,new.leg,len(df)-1,'off'))
    return pd.DataFrame(res,columns=['t', 'n', 'leg', 'ind','type'])

def calc_phase_df(df,ref_leg=0,crit = 'midstance',phase_diffs = [(0,1),(0,2),(0,3)],phase_names = ['LR_h','hl','diag'],limb_names = ['lh','rh','lf','rf']):
    df_out = pd.DataFrame({})
    if np.sum(np.diff(df.iloc[:4][crit])) < 0.01:
        df=df[4:]
    ref=df[df.leg==ref_leg].iloc[:-1]                                   
    phase_dur = ref.swing_dur+ref.stance_dur
    M=np.ones((len(ref),4))*np.nan
    for i, (index,row) in enumerate(ref.iterrows()):
        M[i][ref_leg] = index
        for l in np.setdiff1d([0,1,2,3],ref_leg):
            for k in range(15):
                if not np.isin(k+index+1, df.index):
                    #print(i,index,l,k,k+index+1,'b')
                    break
                nr = df.loc[index+k+1]
                if int(nr.leg) == l:
                    M[i][l] = k+index+1
                    break

    for i,(x,y) in enumerate(phase_diffs):
        with np.errstate(invalid='ignore'):
            df_out[phase_names[i]] =((np.array(df.reindex(M[:,y])[crit])-np.array(df.reindex(M[:,x])[crit]))/np.array(phase_dur)) % 1.0
    for i in range(len(limb_names)):
        df_out['swing_dur_'+limb_names[i]] = np.array(df.reindex(M[:,i]).swing_dur)
        df_out['stance_dur_'+limb_names[i]] = np.array(df.reindex(M[:,i]).stance_dur)
        df_out['onset_'+limb_names[i]] = np.array(df.reindex(M[:,i]).onset)
        df_out['offset_'+limb_names[i]] = np.array(df.reindex(M[:,i]).offset)
        df_out['midstance_'+limb_names[i]] = np.array(df.reindex(M[:,i]).midstance)
    df_out['phase_dur'] = np.array(phase_dur)
    res = calc_sup(df)
    ins = res[(res.leg==ref_leg)&(res.type=='on')].index 
    dsup=np.zeros((len(df_out),5))
    for i in range(len(ins[:-1])):
        dsup_=np.zeros((5,))
        rs = res[ins[i]:ins[i+1]+1].values
        for j in range(len(rs)-1):
            dsup_[rs[j+1][1]] += rs[j+1][0]-rs[j][0]
        dsup[i]=dsup_/(rs[-1][0]-rs[0][0])
    dsup=pd.DataFrame(dsup,columns=['nolimb','onelimb','twolimb','threelimb','fourlimb'])
    df_out=pd.concat([df_out,dsup],axis=1)
    df_out['frequency']=1.0/df_out.phase_dur
    df_out['LR_f']=(df_out.diag-df_out.hl) %1.0
    df_out['hl_r']=(df_out.diag-df_out.LR_h) %1.0
    df_out['diag_2']=(df_out.hl-df_out.LR_h) %1.0
    for l in ['lh', 'rh', 'lf', 'rf']:
        df_out['duty_factor_' + l] = df_out['stance_dur_' + l] / (
            df_out['stance_dur_' + l] + df_out['swing_dur_' + l])

    df_out['duty_factor'] = df_out[[
        'duty_factor_lh', 'duty_factor_rh', 'duty_factor_lf', 'duty_factor_rf'
        ]].mean(axis=1,skipna=True)
    df_out['duty_factor_h']=(df_out.duty_factor_lh+df_out.duty_factor_rh)*0.5
    df_out['duty_factor_f']=(df_out.duty_factor_lf+df_out.duty_factor_rf)*0.5

    df_out['swing_dur_h']=(df_out.swing_dur_lh+df_out.swing_dur_rh)*0.5
    df_out['swing_dur_f']=(df_out.swing_dur_lf+df_out.swing_dur_rf)*0.5


    df_out['stance_dur_h']=(df_out.stance_dur_lh+df_out.stance_dur_rh)*0.5
    df_out['stance_dur_f']=(df_out.stance_dur_lf+df_out.stance_dur_rf)*0.5     
    df_out['ro'] = df_out.onset_lh                                                                                        
    return df_out


parser = OptionParser()
parser.add_option("-s", "--sconfig", dest="s_config_fn",default=None)
options, args = parser.parse_args()

neurons = ["RGF_NaP_hind_L", "RGF_NaP_hind_R",      # neurons to be read every time step 
               "RGF_NaP_fore_L", "RGF_NaP_fore_R",
               "RGE_NaP_hind_L", "RGE_NaP_hind_R",      # neurons to be read every time step 
               "RGE_NaP_fore_L", "RGE_NaP_fore_R"]

s_config = yamlload(options.s_config_fn)


config = yamlload(s_config['model_yaml'])
modelname = config['model_file_name']

filename = "./models/" + modelname

cpg_sim = nsim.simulator(neurons=neurons, filename=filename,dt=0.001)
cpg_sim.initialize_simulator()


dur = 50. # duration of the ramp up/down
N_rep = 2 
alpha_range_reduction = 0.0 

    
do_sample = False

do_updown = False
do_upholddown = False
do_upupdown = False
config_sim = config['simulation']
if config_sim['type'] in ['up_down','up_hold_down','up_up_down']:
    do_updown = True
    
    dur = config_sim['duration'] # duration of the ramp up/down
    N_rep = config_sim['N_rep'] # number of repetitions of the ramp up and down
    if 'N_rep_mult' in s_config:
        N_rep = int(N_rep * s_config['N_rep_mult'])
        print("N_rep set to:", N_rep)
    alpha_range_reduction = config_sim['alpha_range_reduction'] # reduction of the lower extreme of alpha during the up/down ramp (except first and last ramp)
    hold_dur = dur
    if config_sim['type'] == 'up_hold_down':
        do_upholddown = True
        if 'hold_duration' in config_sim:
            hold_dur = config_sim['hold_duration']
    elif config_sim['type'] == 'up_up_down':
        do_upupdown = True

sigma = config['sigma']
alpha_range = config['alpha_range']
case = config['case']

update_sim_from_yaml(config,cpg_sim)
cpg_sim.sim.updateParameter('sigmaNoise',sigma)

time_vec = np.arange(0.0,dur,cpg_sim.dt)
alphas = np.concatenate(
            (np.linspace(alpha_range[0],alpha_range[1],len(time_vec)),
            np.linspace(alpha_range[1],alpha_range[1],len(time_vec)*2),
            np.linspace(alpha_range[1],alpha_range[0],len(time_vec))))
    
time_vec = np.arange(0.0,dur*4,cpg_sim.dt)

if do_updown:
    alpha_range2 = alpha_range[0]+(alpha_range[1]-alpha_range[0])*alpha_range_reduction
    
    time_vec = np.arange(0.0,dur,cpg_sim.dt)
    
    if do_upholddown:
        tv_hold = np.arange(0.0,hold_dur,cpg_sim.dt)
        alphas_ = np.concatenate(
            (np.linspace(alpha_range[1],alpha_range[1],len(tv_hold)),
            np.linspace(alpha_range[1],alpha_range2,len(time_vec)),
            np.linspace(alpha_range2,alpha_range[1],len(time_vec))))
    elif do_upupdown:
        n_up1 = int(len(time_vec)*alpha_range_reduction)
        n_up2 = len(time_vec)-n_up1
        alpha_mid = (alpha_range[0]+alpha_range[1])*0.5
        alphas_ = np.concatenate(
            (np.linspace(alpha_range[0],alpha_mid,n_up1),
            np.linspace(alpha_mid,alpha_range[1],n_up2),
            np.linspace(alpha_range[1],alpha_range[0],len(time_vec))))
    else:
        alphas_ = np.concatenate(
                (np.linspace(alpha_range[1],alpha_range2,len(time_vec)),
                np.linspace(alpha_range2,alpha_range[1],len(time_vec))))
    
    alphas = np.concatenate(
        (
            np.linspace(alpha_range[0],alpha_range[1],len(time_vec)),
            np.matlib.repmat(alphas_, 1, N_rep-1)[0],
            np.linspace(alpha_range[1],alpha_range[0],len(time_vec))
        )
        )
    time_vec = np.arange(0.0,len(alphas)*cpg_sim.dt,cpg_sim.dt)

variable_groups = s_config["variable_groups"]
variables = [ v for group in variable_groups.values() for v in group ]
group_lens = [len(group) for group in variable_groups.values()]

IV=np.array(cpg_sim.sim.setupVariableVector(variables))


update_method = s_config['update_method']

if update_method == 'variable_groups':
    N = len(variable_groups)

    def calculate_var_vec(ind):
        ind_ = np.zeros(sum(group_lens))
        idx = 0
        for i, l in enumerate(group_lens):
            ind_[idx:idx+l] = ind[i]
            idx += l
        return IV*ind_
    par_names = [f"{group_name}" for group_name, group in variable_groups.items() ]
elif update_method == 'variables':
    N = len(variables)

    def calculate_var_vec(ind):
        return IV*np.array(ind)
    par_names = variables
else:
    raise ValueError("update_method must be either 'variable_groups' or 'variables'")
    

def run_sim(ind,cpg_sim_=cpg_sim):
    alpha_range = (0.0,1.0)
    if ind is not None:
        uv = calculate_var_vec(ind)
        cpg_sim_.sim.updateVariableVector(uv)

    cpg_sim_.sim.setAlpha(alpha_range[0])
    for t in np.arange(0.0,10.0,cpg_sim_.dt):
        cpg_sim_.run_step()

    out = np.zeros((len(time_vec),len(cpg_sim_.neurons)))

    for ind_t,alpha in enumerate(alphas):
        cpg_sim_.sim.setAlpha(alpha)
        act = cpg_sim_.run_step()
        out[ind_t,:]=act

    times = nsim.simulator.calc_on_offsets(time_vec,out)
    len_times = len(times)
    
    for i in range(10):
        times = clean_times(times)
        
        if len_times > len(times):
            len_times = len(times)
            continue
        else:
            break
    
    df_times = create_times_df(times)
    unique_counts = df_times.groupby('leg').apply(lambda x: x.shape[0],include_groups=False)
    lrh_ratio = unique_counts[0] / unique_counts[1]
    ref_limb = 0 if lrh_ratio > 1 else 1
    df = calc_phase_df(df_times,ref_limb)
    return out,df,unique_counts



def js_mc(kde_p, kde_q, n=10000, rng=None):
    samp_p = kde_p.sample(n)
    samp_q = kde_q.sample(n)

    log_p = kde_p.score_samples(samp_p)
    log_q = kde_q.score_samples(samp_p)
    log_m = np.logaddexp(log_p, log_q) - np.log(2)
    kl_p = np.mean(rel_entr(np.exp(log_p), np.exp(log_m)))

    log_p2 = kde_p.score_samples(samp_q)
    log_q2 = kde_q.score_samples(samp_q)
    log_m2 = np.logaddexp(log_p2, log_q2) - np.log(2)
    kl_q = np.mean(rel_entr(np.exp(log_q2), np.exp(log_m2)))
    
    return 0.5 * (kl_p + kl_q)

def hellinger_kde_distance(kde_p, kde_q, n=10000, rng=None):
    sp = kde_p.sample(n)
    log_p, log_q = kde_p.score_samples(sp), kde_q.score_samples(sp)
    inner_p = np.mean(np.exp(0.5 * (log_q - log_p)))   # E_p[√(q/p)]

    # sample from q
    sq = kde_q.sample(n)
    log_q2, log_p2 = kde_q.score_samples(sq), kde_p.score_samples(sq)
    inner_q = np.mean(np.exp(0.5 * (log_p2 - log_q2))) # E_q[√(p/q)]

    inner = 0.5 * (inner_p + inner_q)
    return float(np.sqrt(max(0.0, 1.0 - inner)))


def sliced_wasserstein(kde_p, kde_q, d, n_samples=1000, n_projections=20, rng=None):
    """
    Compute the sliced Wasserstein distance between two KernelDensity objects.
    """
    from scipy.stats import wasserstein_distance
    if rng is None:
        rng = np.random.default_rng()

    # ---------- 1-D: direct -----------------------------------------------
    if d == 1 or n_projections is None:
        samp_p = kde_p.sample(n_samples).ravel()
        samp_q = kde_q.sample(n_samples).ravel()
        return wasserstein_distance(samp_p, samp_q)

    # ---------- d ≥ 2: sliced Wasserstein ----------------------------------
    distances = []
    for _ in range(n_projections):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)                 # random unit vector

        samp_p = kde_p.sample(n_samples // 2) @ v
        samp_q = kde_q.sample(n_samples // 2) @ v
        distances.append(wasserstein_distance(samp_p, samp_q))

    return float(np.mean(distances))

def sliced_wasserstein2(X, Y, n_proj=300, seed=0):
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    dist = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        dist += wasserstein_distance(X @ v, Y @ v)
    return dist / n_proj

def emd_distance(P,Q):
    n, m = P.shape[0], Q.shape[0]
    weights_P = np.full(n, 1.0 / n, dtype=float)
    weights_Q = np.full(m, 1.0 / m, dtype=float)    
    C = ot.dist(P, Q, metric=dist_metric)
    emd = ot.emd2(weights_P, weights_Q, C)
    return emd  

def sinkhorn_distance(P,Q):
    n, m = P.shape[0], Q.shape[0]
    weights_P = np.full(n, 1.0 / n, dtype=float)
    weights_Q = np.full(m, 1.0 / m, dtype=float)    
    C = ot.dist(P, Q, metric=dist_metric)
    reg = 1e-3  # Regularization parameter
    sinkhorn_dist = ot.sinkhorn2(weights_P, weights_Q, C, reg)
    return sinkhorn_dist

def evaluate(ind,cpg_sim_=cpg_sim):
    _,df,unique_steps = run_sim(ind,cpg_sim_)
    Xeval=df[['LR_h','hl','diag','hl_r','LR_f','diag_2','frequency']].dropna().values
    Xeval[:,:6] = Xeval[:,:6]*2.0*np.pi
    Xeval[:,-1] = Xeval[:,-1]
    return Xeval,unique_steps

def evaluate_and_score(ind,scoring_fns=['emd'],Xeval_bl=None,kde_bl=None):
    Xeval_,unique_steps = evaluate(ind)
    if 'hellinger' in scoring_fns or 'js' in scoring_fns:
        kde = KernelDensity(bandwidth=np.pi/20., metric="pyfunc", metric_params={"func": dist_metric}, algorithm='ball_tree')
        kde.fit(Xeval_)
    
    results = []
    dist_fns = {
        'hellinger': lambda: hellinger_kde_distance(kde_bl, kde, n=2000),
        'js': lambda: js_mc(kde_bl, kde, n=10000),
        'sliced_wasserstein': lambda: sliced_wasserstein2(Xeval_bl, Xeval_, n_proj=100),
        'emd': lambda: emd_distance(Xeval_bl, Xeval_),
        'sinkhorn': lambda: sinkhorn_distance(Xeval_bl, Xeval_)
    }
    for fn in scoring_fns:
        if fn in dist_fns:
            t0 = time.time()
            dist = dist_fns[fn]()
            t1 = time.time()
            print(f"{fn} distance: {dist} (time: {t1-t0:.3f}s)")
            results.append(dist)
    for i in range(4):
        results.append(unique_steps[i])

    return results

def sobol_sample(type, N_vars=5, m_samples=10,seed=42):
    from scipy.stats import qmc

    if type == 'mult':
        param_bounds = np.array([[-1,  1]]* N_vars)
    elif type == 'zero_one':
        param_bounds = np.array([[0, 1]]* N_vars)
    
    d = N_vars

    engine = qmc.Sobol(d, scramble=True, seed=seed)
    u = engine.random_base2(m=m_samples)          # shape (2**m, d)

    lower, upper = param_bounds[:,0], param_bounds[:,1]

    return lower + u * (upper - lower) 



if __name__ == "__main__":
    out_dir = './sensitivity'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_fn = options.s_config_fn.split('/')[-1].split('.')[0]+"_"+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    out_filename = os.path.join(out_dir,out_fn)
    print("Output will be saved to:", out_filename)
    
    # either simulate intact model (not updating variables from yaml) or simulate model with standard parameters (sensitivity)
    do_bl = False
    if 'comp_intact' in s_config:
        if s_config['comp_intact']:
            cpg_sim2 = nsim.simulator(neurons=neurons, filename=filename,dt=0.001)
            cpg_sim2.initialize_simulator()
            cpg_sim2.sim.updateParameter('sigmaNoise',sigma)
            Xeval,unique_steps = evaluate(None,cpg_sim2)
        else:
            do_bl = True
    else:
        do_bl = True
    if do_bl:
        ind = np.ones((N,))
        Xeval,unique_steps = evaluate(ind)

    kde_bl = KernelDensity(bandwidth=np.pi/20., metric="pyfunc",metric_params={"func":dist_metric}, algorithm='ball_tree')
    kde_bl.fit(Xeval)

    if 'range' in s_config:
        if isinstance(s_config['range'], list):
            range_ = np.array(s_config['range'])
            sample = sobol_sample('zero_one',N_vars=N, m_samples=s_config['m_samples'],seed=42)
            sample_val = sobol_sample('zero_one',N_vars=N, m_samples=s_config['m_samples_val'],seed=43)
            X_train = sample * (range_[1] - range_[0]) + range_[0]
            X_val = sample_val * (range_[1] - range_[0]) + range_[0]
        else:
            print("range specified but not a list")
            exit(1)
    elif 'scale_exp' in s_config:
        # Sample for train and validation sets
        sample = sobol_sample('mult',N_vars=N, m_samples=s_config['m_samples'],seed=42)
        
        sample_val = sobol_sample('mult',N_vars=N, m_samples=s_config['m_samples_val'],seed=43)

        scale = np.log(s_config['scale_exp'])
        X_train = np.exp(sample * scale)
        X_val = np.exp(sample_val * scale)
    print("Sample shape:", sample.shape)

    # Print the minimum and maximum values in the training sample for inspection.
    print("Min of X_train:", np.min(X_train))
    print("Max of X_train:", np.max(X_train))
    X = np.concatenate((X_train, X_val), axis=0)
    
    scoring_fns = s_config['scoring_fn'] if isinstance(s_config['scoring_fn'], list) else [s_config['scoring_fn']]
    my_function = lambda x: evaluate_and_score(x, scoring_fns=scoring_fns, Xeval_bl=Xeval, kde_bl=kde_bl)

    #import IPython;IPython.embed()
    y = list(tqdm(
        futures.map(
            my_function,
            X
        ),
        total=len(X),
        desc="Evaluating samples"
    ))


    y = np.array(y)
    y_train = y[:X_train.shape[0]]
    y_val = y[X_train.shape[0]:]

    # Save results
    np.savez(out_filename, X=X, y=y, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,par_names=par_names,variables=variables,variable_groups=variable_groups,s_config_fn=options.s_config_fn)
   
    # Check if 'run' exists in s_config, if not create it
    if 'run' not in s_config or not isinstance(s_config['run'], list):
        s_config['run'] = []

    # Determine entry number
    entry_number = len(s_config['run'])

    # Create new entry
    run_entry = {
        'entry': entry_number,
        'out_filename': out_filename,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if 'N_rep_mult' in s_config:
        run_entry['N_rep_mult'] = s_config['N_rep_mult']
    # Append entry
    s_config['run'].append(run_entry)

    # Save updated s_config back to the YAML file
    with open(options.s_config_fn, 'w') as file:
        import yaml
        yaml.dump(s_config, file,default_flow_style=False, sort_keys=False, Dumper=CustomCDumper)
    

    

    
    
    
    