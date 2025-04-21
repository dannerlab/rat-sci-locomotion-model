from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import circmean, circstd
import os
from classify_gait import classify_gait
import lmfit
import yaml
from datetime import datetime
from yaml_sim import update_sim_from_yaml,yamlload
import CPGNetworkSimulator.tools.py_simulator as nsim 

def circ_r(a,low=0,high=1):
    a=(a-low)*2*np.pi/(high-low)
    return np.abs(np.sum(np.exp(1j*a)))/len(a)


fc = ['#000000','#F7941D','#00A875','#0072BC','#DA70D6']


def plot_avg_gait(df,label):
    lr = circmean(df.LR_h,low=0,high=1)
    hl = circmean(df.hl,low=0,high=1)
    diag = circmean(df.diag,low=0,high=1)
    duty_factor =  [np.mean(df.duty_factor)]*4
    frequency = np.mean(df.frequency)
    
    phases=[0.0,lr,hl,diag]
    period = 1/frequency
    width = 0.2
    fc = ['black','red','blue','green']
    
    fig,ax=plt.subplots(1,1)
    fig.set_size_inches(8., 2, forward=True)
    for j in range(4):
        xranges = [(on,period*duty_factor[j]) for on in np.arange(-0.2,1.2,period) + phases[j]*period]
        yranges = (1.0+j*width*1.5,width)
        ax.broken_barh(xranges,yranges,facecolors=fc[j])
    ax.set_xlim((0,0.5))
    ax.set_yticks([1.0+j*width*1.5+width*0.5 for j in range(4)])
    ax.set_yticklabels(['lh','rh','lf','rf'])
    ax.set_title(str(label) +' '+ str(len(df)))




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

def calc_phase(time_vec,out,phase_diffs,times):
    ref_onsets = times[np.logical_and(times[:,1]==0,times[:,3]==0)][:,0]
    phase_dur=np.append(ref_onsets[1:]-ref_onsets[:-1],np.nan)
    p = times[times[:,1]==0]
    indices = np.where(np.diff(p[:,3])==1)
    
    fl_phase_dur = np.zeros((len(ref_onsets)))
    fl_phase_dur[:] = np.nan
    fl_phase_dur[p[indices,2].astype(int)] = p[[ind+1 for ind in indices],0] - p[indices,0]
    ex_phase_dur = phase_dur-fl_phase_dur
    
    M = np.zeros((len(ref_onsets),out.shape[1]))
    M[:] = np.nan
    M[:,0]=ref_onsets
    
    for i in range(1,out.shape[1]):
        p = times[np.logical_and((times[:,1]==0) | (times[:,1]==i),times[:,3]==0)]
        indices = np.where(np.diff(p[:,1])==i)
        M[p[indices,2].astype(int),i] = p[[ind+1 for ind in indices],0]
        
    phases=np.zeros((len(ref_onsets),len(phase_diffs)))
    for i,(x,y) in enumerate(phase_diffs):
        phases[:,i] = ((M[:,y]-M[:,x])/phase_dur)  % 1.0
    if phases.shape[0]!=0:
        no_nan = ~np.isnan(np.concatenate(
                    (np.stack((phase_dur,fl_phase_dur,ex_phase_dur),1),phases),1
                    )).any(axis=1)
        return (phase_dur[no_nan],fl_phase_dur[no_nan],ex_phase_dur[no_nan],phases[no_nan],ref_onsets[no_nan])
    else:
        return (phase_dur,fl_phase_dur,ex_phase_dur,phases,ref_onsets[:-1])   
    
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
    #if not np.all(np.isin(np.setdiff1d([0,1,2,3],ref_leg),df[ref.tail(1).index.item():].leg)):
    #    ref=ref.iloc[:-1]                                         
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
    
    phases=np.zeros((len(ref),len(phase_diffs)))
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

cp=sns.color_palette("Paired", 20)
cp[0]=cp[11]
cp.pop(7)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-y", "--yaml", dest="yaml_fn",default='config_test.yaml')
    options, args = parser.parse_args()

    neurons = ["RGF_NaP_hind_L", "RGF_NaP_hind_R",      # neurons to be read every time step 
               "RGF_NaP_fore_L", "RGF_NaP_fore_R",
               "RGE_NaP_hind_L", "RGE_NaP_hind_R",      # neurons to be read every time step 
               "RGE_NaP_fore_L", "RGE_NaP_fore_R"]
    config = yamlload(options.yaml_fn)
    modelname = config['model_file_name']

    

    filename = "./models/" + modelname

    fn = os.path.join(os.path.dirname(__file__),".",filename)
    print(fn)

    cpg_sim = nsim.simulator(neurons=neurons, filename=fn,dt=0.001)
    cpg_sim.initialize_simulator()

    
    dur = 50. # duration of the ramp up/down
    N_rep = 2 
    alpha_range_reduction = 0.0 

    sigma0 = 0.1
    exp_color = '0.8'
    min_fr = 0
    max_fr = 7.5
    max_phase = 0.3
    s_sci = 8
    s_sci1 = 5
    s_sim = 1.5
    s_sim1 = 1.5
    t_size = 5
    
    do_sample = False

    do_updown = False
    do_upholddown = False
    do_upupdown = False
    config_sim = config['simulation']
    if config_sim['type'] in ['up_down','up_hold_down','up_up_down']:
        do_updown = True
        
        dur = config_sim['duration'] # duration of the ramp up/down
        N_rep = config_sim['N_rep'] # number of repetitions of the ramp up and down
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
    
    config['update_history'] = update_sim_from_yaml(config,cpg_sim)

    out_dir = 'runs'
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
    out_fn = modelname.split('.')[0]+'_case'+case+'_'+datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    out_filename = os.path.join(out_dir,out_fn)
    with open(out_filename+'.yaml','w') as f:
        yaml.dump(config,f,Dumper=yaml.CDumper,default_flow_style=False, sort_keys=False)
   

    cpg_sim.sim.updateParameter('sigmaNoise',sigma)
    cpg_sim.initialize_simulator()
    
    
    time_vec = np.arange(0.0,dur,cpg_sim.dt)
    alphas = np.concatenate(
            (np.linspace(alpha_range[0],alpha_range[1],len(time_vec)),
            np.linspace(alpha_range[1],alpha_range[1],len(time_vec)*2),
            np.linspace(alpha_range[1],alpha_range[0],len(time_vec))))
    
    time_vec = np.arange(0.0,dur*4,cpg_sim.dt)

    if config_sim['type'] in ['simulate_bouts']:
        alpha_values = config_sim['alpha_values']
        c = config_sim['multiplier']
        times = [c*t for t in config_sim['times']]
        
        N_rep = config_sim['N_rep']
        time_vec = np.arange(0.0,np.sum(times[-1]),cpg_sim.dt)
        alphas_interp = np.interp(time_vec, times, alpha_values)
        
        alphas = np.matlib.repmat(alphas_interp, 1, N_rep)
        alphas = alphas.reshape((alphas.shape[-1],))
        time_vec = np.arange(0.0,len(alphas)*cpg_sim.dt,cpg_sim.dt)

    if config_sim['type'] in ['simulate_bouts_noisy']:
        alpha_values = config_sim['alpha_values']
        alpha_stds = config_sim['alpha_stds']
        N_rep = config_sim['N_rep']
        interval = config_sim['interval']

        
        av = alpha_values * N_rep + [alpha_values[0]]
        
        astd = alpha_stds * N_rep + [alpha_stds[0]]
        avs = [a+np.clip(np.random.normal(0,s),-2*s,2*s) for a,s in zip(av,astd)]
        avs = np.clip(avs,0.1,1.1)
        avs = [val for val in avs for _ in range(2)]
        time_vec = np.arange(0.0,interval*(len(avs))+cpg_sim.dt,cpg_sim.dt)
        alphas = np.interp(time_vec, np.arange(len(avs)) * interval, avs)



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

    out = np.zeros((len(time_vec),len(cpg_sim.neurons)))
    
    # simulation loop
    for ind_t,alpha in enumerate(alphas):
        cpg_sim.sim.setAlpha(alpha)
        act = cpg_sim.run_step()
        out[ind_t,:]=act
        if time_vec[ind_t]%100.0 == 0.0:
            print('t = ' + str(time_vec[ind_t]))

    # process output data
    burst_durs = nsim.simulator.calc_burst_durations(time_vec,out)
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
    df = calc_phase_df(df_times)
    df=classify_gait(df.copy())

    df_phases = pd.read_hdf("./data/df_phases.h5",'df_phases')

    # calculate hindlimb stance/swing durations and duty factor
    df_phases['stance_dur_h']=(df_phases.stance_dur_lh+df_phases.stance_dur_rh)*0.5
    df_phases['swing_dur_h']=(df_phases.swing_dur_lh+df_phases.swing_dur_rh)*0.5
    df_phases['duty_factor']=df_phases['stance_dur_h']/df_phases['phase_dur']

    #  experimental data 
    if 'contusion' in case:
        df_sci = df_phases.loc[df_phases.SCI == 'contusion']
    elif 'hemi' in case:
        df_sci = df_phases.loc[df_phases.SCI == 'hemisection']
    else:
        df_sci = df_phases.loc[df_phases.SCI == 'intact']
    df_sci=classify_gait(df_sci.copy())

    if do_sample:
        N_samples = len(df_sci)
        df=df.sample(n=N_samples)
        #df=df.iloc[:N_samples]
        s_sim = s_sci
        s_sim1 = s_sci1

    # print mean and std of the phase differences
    for phase in ['LR_h','hl','diag','LR_f']:
        print(phase + " {:4.3f} ({:4.3f})".format(circmean(df[phase],low=0,high=1),circstd(df[phase],low=0,high=1)))

    col = [cp[k] for k in df.gaits.cat.codes.values]

    # plot burst durations and phase differences vs time
    fig, ax = plt.subplots(8, 5, figsize=(10, 10),gridspec_kw={'width_ratios':[2,2,2,1,1],'height_ratios': [1,1,1, 1, 1 , 1, 1, 1]})

    # read experimental data
    c_factor = 'duty_factor'    

    phd_full ={'LR_h':'left-right hind (rh-lh)','LR_f':'left-right fore (rf-lf)','hl':'homolateral left (lf-lh)','hl_r':'homolateral right (rf-rh)','diag':'diagonal 1 (rf-lh)','diag_2':'diagonal 2 (lf-rh)'}
    phd_full = {k:v[:-8]+'\nphase difference' for k,v in phd_full.items()}
    # plot exp data: phase differences vs frequency 
    clm = 0
    plot_xys = [('frequency', 'LR_h'),('frequency', 'LR_f'),('frequency', 'hl'),('frequency', 'hl_r'),('frequency', 'diag'),('frequency', 'diag_2'),]
    for (j, phasep) in enumerate(plot_xys):
        ax[j,clm].scatter(df_sci[phasep[0]],df_sci[phasep[1]],c=[cp[k] for k in df_sci.gaits.cat.codes.values],alpha=0.25,s=s_sci,lw=0)
        ax[j,clm].plot([min_fr, max_fr],[0.5,0.5],'k',lw=0.5,alpha=0.5)
        ax[j,clm].set_xlim((min_fr, max_fr))
        ax[j,clm].set_ylim([-0.05, 1.05])
        ax[j,clm].tick_params(axis='y', labelsize=8)
        ax[j,clm].tick_params(labelbottom=False)    
        ax[j,clm].set_ylabel(phd_full[phasep[1]],fontsize=8)
    ax[0,clm].set_title('Experiment-'+case,fontsize=t_size,fontweight='bold')
    
    

    if sigma < sigma0: # plot data: phase differences vs frequency in back groud (for intact case only)
        exp_color = '0.8'       
        clm = 1
        for (j, phasep) in enumerate(plot_xys):
            ax[j,clm].scatter(df_sci[phasep[0]],df_sci[phasep[1]],c=exp_color,alpha=0.25,s=4,lw=0)
            ax[j,clm].plot([0,1],[0.5,0.5],'k',lw=1,alpha=0.5)
            ax[j,clm].plot([0.5,0.5],[0,1],'k',lw=1,alpha=0.5)
            ax[j,clm].set_xlim((min_fr, max_fr))
            ax[j,clm].set_ylim([-0.05, 1.05])
        clm = 3
        # plot phase differences against other phase differences in back groud (for intact case only)
        for i,(xpd,ypd) in enumerate([['LR_h','LR_f'],['LR_h','hl'],['LR_h','diag'],['hl','diag']]):
            ax[i,clm].scatter(df_sci[xpd],df_sci[ypd],c=exp_color,alpha=0.25,s=4,lw=0)

    # plot phase differences vs frequency 
    clm = 1
    for i, (xname,yname) in enumerate(plot_xys):
        ax[i,clm].scatter(df[xname],df[yname],c=col,s=s_sim,alpha=0.25,lw=0)
        ax[i,clm].plot([min_fr, max_fr],[0.5,0.5],'k',lw=0.5,alpha=0.5)
        ax[i,clm].set_ylim([-0.05, 1.05])
        ax[i,clm].set_xlim([min_fr, max_fr])
        ax[i,clm].tick_params(labelbottom=False,labelleft=False)
    ax[0,clm].set_title('Model-'+case,fontsize=t_size,fontweight='bold') 

    row = 6    
    ax[row,clm].scatter(df.frequency,df.stance_dur_h,s=s_sim,label='- E')   
    ax[row,clm].scatter(df.frequency,df.swing_dur_h,s=s_sim,label='- F')    
    ax[row,clm].set_xlabel('frequency',fontsize=8)
    ax[row,clm].tick_params(axis='both', labelbottom=True,labelleft=False,labelsize=8)
    ax[row,clm].set_xlim([min_fr, max_fr])
    ax[row,clm].set_ylim((0, max_phase)) 
    ax[row,clm].legend(loc="best",fontsize=8)

    # plot  phase differences, frequency and fl phase duration vs alpha   
    clm = 2
    for i,pd_ in enumerate([y for x,y in plot_xys]):
        ax[i,clm].scatter(df.ro,df[pd_],c=col,s=1.5,alpha=0.25,lw=0)
        ax[i,clm].set_ylim([-0.05, 1.05])
        ax[i,clm].tick_params(labelbottom=False,labelleft=False)
    ax[0,2].set_title(modelname+', sigma='+str(sigma)+', alpaha=['+str(min(alpha_range))+','+str(max(alpha_range))+']',fontsize=t_size,fontweight='bold')

    row = 6
    ax[row,clm].plot(df.ro,df.stance_dur_h,label='- E')
    ax[row,clm].plot(df.ro,df.swing_dur_h,label='- F')    
    ax[row,clm].tick_params(axis='both',labelsize=8,labelleft=False)
    ax[row,clm].set_ylim(0, max_phase)   
    ax[row,clm].legend(loc="best",fontsize=8)

    row = 7
    ax[row,clm].plot(df.ro,1.0/df.phase_dur,label='frequency')
    ax[row,clm].set_ylabel('frequency',fontsize=8)
    ax[row,clm].set_ylim((0, 8)) 
    ax[row,clm].tick_params(axis='both', labelsize=8)
    ax[row,clm].set_xlabel('alpha',fontsize=8)
    
    # plot phase differences against other phase differences
    clm = 3
    for i in range(2):
        for j in range(2):
            ax[2*j+i, clm].plot([-0.05,1.05],[0.5,0.5],'k',lw=0.5,alpha=0.5)
            ax[2*j+i, clm].plot([0.5,0.5],[-0.05,1.05],'k',lw=0.5,alpha=0.5)

            ax[2*j+i, clm].plot([-0.05,1.05],[1.05,-0.05],'k',lw=0.5,alpha=0.5)
            ax[2*j+i, clm].plot([-0.05,1.05],[-0.05,1.05],'k',lw=0.5,alpha=0.5)
    row = 0
    ax[row,clm].scatter(df.LR_h,df.LR_f,c=col,s=s_sim1,alpha=0.25,lw=0)
    ax[row,clm].set_ylim([-0.05, 1.05])
    ax[row,clm].set_xlim([-0.05, 1.05])
    ax[row,clm].set_ylabel('LR_f',fontsize=8)
    ax[row,clm].set_title('LR_h=' + str(round(circmean(df.LR_h,1),2)) + ' +-'+ str(round(circstd(df.LR_h,1),2)),fontsize=8)
    ax[row,clm].set_aspect('equal')
    ax[row,clm].tick_params(labelsize=8,labelbottom=False)
    row = 1
    ax[row,clm].scatter(df.LR_h,df.hl,c=col,s=s_sim1,alpha=0.25,lw=0)
    ax[row,clm].set_ylim([-0.05, 1.05])
    ax[row,clm].set_xlim([-0.05, 1.05])
    ax[row,clm].set_ylabel('hl',fontsize=8)
    ax[row,clm].set_title('hl=' + str(round(circmean(df.hl,1),2)) + ' +-'+ str(round(circstd(df.hl,1),2)),fontsize=8)
    ax[row,clm].set_aspect('equal')
    ax[row,clm].tick_params(labelsize=8,labelbottom=False)
    row = 2
    ax[row,clm].scatter(df.LR_h,df.diag,c=col,s=s_sim1,alpha=0.25,lw=0)
    ax[row,clm].set_xlabel('LR_h',fontsize=8)
    ax[row,clm].set_ylim([-0.05, 1.05])
    ax[row,clm].set_xlim([-0.05, 1.05])
    ax[row,clm].set_ylabel('diag',fontsize=8)
    ax[row,clm].set_title('diag=' + str(round(circmean(df.diag,1),2)) + ' +-'+ str(round(circstd(df.diag,1),2)),fontsize=8)
    ax[row,clm].set_aspect('equal')
    ax[row,clm].tick_params(labelsize=8,labelbottom=False)
    row = 3
    ax[row,clm].scatter(df.hl,df.diag,c=col,s=s_sim1,alpha=0.25,lw=0)
    ax[row,clm].set_ylim([-0.05, 1.05])
    ax[row,clm].set_xlim([-0.05, 1.05])
    ax[row,clm].set_xlabel('hl',fontsize=8)
    ax[row,clm].set_ylabel('diag',fontsize=8)
    ax[row,clm].set_aspect('equal')
    ax[row,clm].tick_params(labelsize=8,labelbottom=False)

    # plot exp phase differences against other phase differences
    clm = 4
    for i,(xpd,ypd) in enumerate([['LR_h','LR_f'],['LR_h','hl'],['LR_h','diag'],['hl','diag']]):
        ax[i,clm].scatter(df_sci[xpd],df_sci[ypd],c=[cp[k] for k in df_sci.gaits.cat.codes.values],alpha=0.25,s=s_sci1,lw=0)
        for x,y in [([-0.05,1.05],[-0.05,1.05]),
                    ([-0.05,1.05],[1.05,-0.05]),
                    ([-0.05,1.05],[0.5,0.5]),
                    ([0.5,0.5],[-0.05,1.05])]:
            ax[i,clm].plot(x,y,c='k',lw=0.5)
        ax[i,clm].set_ylim([-0.05, 1.05])
        ax[i,clm].set_xlim([-0.05, 1.05])
        ax[i,clm].set_aspect('equal')
        ax[i,clm].tick_params(labelsize=8,labelbottom=False,labelleft=False)
    ax[0,clm].set_title('Experiment-'+case,fontsize=t_size,fontweight='bold') 
    ax[2,clm].set_xlabel('LR_h',fontsize=8)   
    ax[3,clm].set_xlabel('hl',fontsize=8) 
    # linear regression of swing/stance duration vs period
    df_sci=df_sci[df_sci.phase_dur<0.4]
    df_sci=df_sci[df_sci.stance_dur_h<0.2]
    df_sci=df_sci[df_sci.swing_dur_h<0.2]
    df_sci_=df_sci[['stance_dur_h','swing_dur_h','frequency','phase_dur']].dropna()
    # import IPython;IPython.embed()
    f_lin = lambda x, a, b: x * a + b
    f_ex = lambda x, a, b: np.exp(-x * a) + b
    x_lim = [0.1,0.4]
    x = np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 100.0)
    pars = lmfit.Parameters()
    pars.add_many(('a', 0.1, True,-1e5, 1e5, None),
                ('b', 0.1, True, -1e5, 1e5, None))

    x = np.arange(x_lim[0], x_lim[1], (x_lim[1] - x_lim[0]) / 100.0)

    out_st=lmfit.Model(f_lin).fit(df_sci_['stance_dur_h'],x=df_sci_['phase_dur'],params=pars)

    best_st = out_st.eval(x=x)
    dely_st = out_st.eval_uncertainty(x=x, sigma=5)
    clm = 1
    row = 7

    # plot flexor and extensor phases vs period
    ax[row,clm].scatter(df.phase_dur,df.stance_dur_h,s=3,lw=0,label='- E')
    ax[row,clm].scatter(df.phase_dur,df.swing_dur_h,s=3,lw=0,label='- F')
    ax[row,clm].set_xlabel('period',fontsize=8)
    ax[row,clm].set_ylim((0, 0.25)) 
    ax[row,clm].tick_params(axis='both',labelsize=8,labelleft=False)
    ax[row,clm].legend(loc="best",fontsize=8)

    if sigma < sigma0:
        ax[row,clm].scatter(df_sci_['phase_dur'],df_sci_['stance_dur_h'],c=exp_color,s=5,lw=0,alpha=0.2)
    ax[row,clm].plot(x, best_st)

    ax[row,clm].fill_between(x,best_st - dely_st,best_st + dely_st,alpha=0.5)
    ax[row,clm].set_ylim((0, max_phase)) 
    ax[row,clm].set_xlim(x_lim) 

    out_sw=lmfit.Model(f_lin).fit(df_sci_['swing_dur_h'],x=df_sci_['phase_dur'],params=pars)

    best_sw = out_sw.eval(x=x)
    dely_sw = out_sw.eval_uncertainty(x=x, sigma=5)

    if sigma < sigma0:
        ax[row,clm].scatter(df_sci_['phase_dur'],df_sci_['swing_dur_h'],c=exp_color,s=5,lw=0,alpha=0.2)
    ax[row,clm].plot(x, best_sw)

    ax[row,clm].fill_between(x,best_sw - dely_sw,best_sw + dely_sw,alpha=0.5)
    ax[row,clm].set_xlim(x_lim) 
    ax[row,clm].set_ylim((0, 0.25))      
    
    # plot stance and swing  vs frequency
    clm = 0
    row = 6
    ax[row,clm].scatter(1/df_sci['phase_dur'],df_sci['stance_dur_h'],s=3,lw=0,label='- stance')
    ax[row,clm].scatter(1/df_sci['phase_dur'],df_sci['swing_dur_h'],s=3,lw=0,label='- swing') 
    ax[row,clm].set_ylabel('F-E phases',fontsize=8)
    ax[row,clm].set_xlabel('frequency',fontsize=8)
    ax[row,clm].set_ylim((0, max_phase)) 
    ax[row,clm].tick_params(axis='both', labelsize=8)
    ax[row,clm].set_xlim([min_fr, max_fr])
    ax[row,clm].legend(loc="best",fontsize=8)
    row = 7
    # plot tance and swing  vs frequencys vs period
    ax[row,clm].scatter(df_sci['phase_dur'],df_sci['stance_dur_h'],s=3,lw=0,label='- stance')
    ax[row,clm].scatter(df_sci['phase_dur'],df_sci['swing_dur_h'],s=3,lw=0,label='- stance')
    ax[row,clm].set_xlabel('period',fontsize=8)
    ax[row,clm].set_ylabel('F-E phases vs period',fontsize=8)
    ax[row,clm].set_ylim((0, max_phase)) 
    ax[row,clm].tick_params(axis='both', labelsize=8)
    ax[row,clm].legend(loc="best",fontsize=8)
    ax[row,clm].plot(x, best_sw)
    ax[row,clm].fill_between(x,best_sw - dely_sw,best_sw + dely_sw,alpha=0.5)
    ax[row,clm].set_xlim(x_lim) 
    ax[row,clm].set_ylim((0, 0.25)) 

    ax[row,clm].plot(x, best_st)
    ax[row,clm].fill_between(x,best_st - dely_st,best_st + dely_st,alpha=0.5)
    ax[row,clm].set_ylim((0, 0.25)) 
    ax[row,clm].set_xlim(x_lim) 

    plt.subplots_adjust(left=0.03,
                bottom=0.05, 
                right=0.98, 
                top=0.98, 
                wspace=0.17, 
                hspace=0.26)

    clm = 3
    gs = ax[3, clm].get_gridspec()
  
    for ax_ in ax[4:, clm:]: 
        for ax2_ in ax_:
            ax2_.remove()      
    
    axbig = fig.add_subplot(gs[6:, clm])

    g=df.groupby('gaits',observed=False).apply(len,include_groups=False)/len(df)

    sns.barplot(x=g.index,y=g.values,hue=g.index,palette=cp,ax=axbig)
    plt.xticks(rotation=45,fontsize=6)
    plt.yticks(fontsize=8)
    plt.ylim([0, 0.6])

    clm = 4
    axbig = fig.add_subplot(gs[6:, clm])
    g=df_sci.groupby('gaits',observed=False).apply(len,include_groups=False)/len(df_sci)
    sns.barplot(x=g.index,y=g.values,hue=g.index,palette=cp,ax=axbig)
    plt.xticks(rotation=45,fontsize=6)
    plt.yticks(fontsize=8)
    plt.ylim([0, 0.6])

    sns.despine()
    df.to_hdf(out_filename+'.h5',key='df',format='table')
    df_times.to_hdf(out_filename+'.h5',key='df_raw',format='table',append=True)

    def ma_circ(x,fc,interval=1):
        """Calculates moving average of x (in radians) with normalized cutoff frequency fc"""
        alpha = (fc*2*np.pi*interval)/(fc*2*np.pi*interval+1)
        y=np.zeros(len(x))
        y[0]=x[0]
        for i in range(1,len(x)):
            y[i] = y[i-1]+alpha*(np.angle(np.exp(1j*x[i])/np.exp(1j*y[i-1])))
        y=y%(2*np.pi)
        return y

    def ma_dev(x,fc,interval=1):
        """Calculates mean deviation of x (in radians) from its moving average with normalized cutoff frequency fc"""
        ma_x = ma_circ(x,fc,interval)
        return np.mean(np.abs(np.angle(np.exp(1j*(x-ma_x)))))/(np.pi)

    def calc_mean_ma_dev(df_,fc):
        """Calculates mean deviation from the moving average for all six phase differences in dataframe df_ with normalized cutoff frequency fc"""
        ret={}
        for phase_diff in ['LR_h','hl','diag','LR_f','hl_r','diag_2']:
            ret['r_'+phase_diff]=ma_dev(df_[phase_diff].values*np.pi*2.0,fc)
        return ret


    df["bout"] = np.floor(np.array(df.index) / 10)
    
    def calc_madev(df_,groupby):
        res_=df_.groupby(groupby).apply(lambda df:calc_mean_ma_dev(df,0.125),include_groups=False).apply(pd.Series)
        res=res_.reset_index()
        df_madev=res.melt(id_vars=groupby, var_name='pdiff',value_name='madev')
        df_madev.pdiff=pd.Categorical(df_madev.pdiff,categories=['r_LR_f', 'r_LR_h',  'r_hl', 'r_hl_r','r_diag', 'r_diag_2'])

        df_madev.loc[df_madev.madev==0.,'madev']=0.0000001
        return df_madev
    
    for i,(df_,groupby_) in enumerate([(df,['bout']),(df_sci,['ID','RID','bout'])]):
        axbig = fig.add_subplot(gs[4:6, i+3])
        df_madev=calc_madev(df_,groupby_)
        sns.boxplot(x='pdiff',y='madev',data=df_madev,ax=axbig)
        axbig.set_ylim([0, 0.25])
        plt.xticks(rotation=45,fontsize=6)
        plt.yticks(fontsize=8)
        sns.despine()
        
    
    fig.savefig(out_filename+'_1'+'.pdf',bbox_inches='tight')
    # plt.show() 

    def plot_avg_gait_phase(df,label,ax):
        lr = circmean(df.LR_h,1,0)
        hl = circmean(df.hl,1,0)
        diag = circmean(df.diag,1,0)
        lr_fl = circmean(df.LR_f,1,0)
        
        phases=np.array([0.0,lr,hl,diag,lr_fl])
        phase_r=np.array([1.0,circ_r(df.LR_h),circ_r(df.hl),circ_r(df.diag),circ_r(df.LR_f)])
        
        
        for j in range(4):
            ax.plot([phases[j]*2*np.pi]*2,[0,phase_r[j]],fc[j])
        ax.scatter(phases*2*np.pi,phase_r,c=fc)
        ax.set_ylim((0,1.1))
        ax.set_yticks([1])
        ax.set_yticklabels([""])
            
        
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
            
        ax.set_title(str(label) )
        ax.set_thetagrids(
                    np.linspace(0.0, 360, 9)[:-1], ["0", "", ".25", "", ".5", "", ".75",""],
                    fontsize=6)

    def plot_bout(ax,times,dfg):
        width = 0.2

        xg = [(x1, x2 - x1) for x1, x2 in zip(dfg.ro.iloc[0:-1], dfg.ro.iloc[1:])]
        yg = (0.9,4*width*1.5+0.1)
        
        col = [cp[k] for k in dfg.gaits.cat.codes.values]
        ax.broken_barh(xg,yg,facecolors=col,alpha=0.5)
        for j in range(4):
            onsets = times[np.logical_and(times[:,1]==j,times[:,3]==1)][:,0]
            offsets = times[np.logical_and(times[:,1]==j,times[:,3]==0)][:,0]
            while offsets[0]<onsets[0]:
                offsets=offsets[1:]
            while onsets[-1]>offsets[-1]:
                onsets=onsets[:-1]

            xranges = [(on,off-on) for on,off in zip(onsets,offsets)]
            #xranges = [(row.onset,row.offset-row.onset) for (index,row) in df_bout[df_bout.leg==j].iterrows()]
            yranges = (1.0+j*width*1.5,width)
            ax.broken_barh(xranges,yranges,facecolors=fc[j],lw=0)
        
        ax.set_yticks([1.0+j*width*1.5+width*0.5 for j in range(4)])
        ax.set_yticklabels(['lh','rh','lf','rf'])

    def plot_circ_line(x,phi,ax,**kwargs):
        i_wrap=np.where(np.abs(np.diff(phi))>0.5)[0]
        i_wrap+=1
        i_wrap=np.append(i_wrap,[len(phi)])
        segs = []
        last=0
        if not (i_wrap.size>0):
            segs.append((x,phi))
        else:
            for iw in i_wrap:
                segs.append((x[last:iw],phi[last:iw]))
                phi_=phi[iw-1:iw+1]
                dphi=np.diff(phi_)
                x_=x[iw-1:iw+1]
                dx=np.diff(x_)
                if dphi.size > 0:
                    if dphi<0:
                        din=((1-phi_[0])/(dphi%1))*dx
                        if din.size==1:
                            din=din[0]
                        segs.append(([x_[0],x_[0]+din],[phi_[0],1]))
                        segs.append(([x_[0]+din,x_[1]],[0,phi_[1]]))
                    elif dphi>0:
                        din=((-phi_[0])/(dphi-1))*dx
                        if din.size==1:
                            din=din[0]
                        segs.append(([x_[0],x_[0]+din],[phi_[0],0]))
                        segs.append(([x_[0]+din,x_[1]],[1,phi_[1]]))
                last=iw
        for x_,phi_ in segs:
            ax.plot(x_,phi_,**kwargs)

    def onclick(event):
        PLOT_DUR = 4.
        NADD = 300
        print('%s click: button=%d, x=%f, y=%f' %
            (event.inaxes, event.button,
                event.xdata, event.ydata))
        
        #import IPython;IPython.embed()
        if len(np.where(np.diff(df.frequency > event.xdata))[0])<1:
            i_fq = np.where(np.diff(df.ro < event.xdata))[0][0]
        else:
            i_fq = np.where(np.diff(df.frequency > event.xdata))[0][0]
        #if (i_fq < NADD) or (i_fq > len(out)-NADD-int(PLOT_DUR/cpg_sim.dt)):
        #    return

        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(5, 4, hspace=0.0, wspace=0.2)
        ax1 = fig.add_subplot(grid[:-3,:-1])
        t_on = df.ro.iloc[i_fq]
        t_off = t_on + PLOT_DUR
        i_fq_off = np.where(np.diff(df.ro >= t_off))[0][0]+1

        i_on = np.where(np.diff(time_vec >= t_on))[0][0]
        i_off = int(i_on + cpg_sim.dt**-1 * PLOT_DUR)
        if i_off>= len(time_vec):
            i_off = len(time_vec)-1

        for i in range(4):
            ax1.plot(time_vec[i_on:i_off], out[i_on:i_off, i]+i,fc[i])
            ax1.plot(time_vec[i_on:i_off], out[i_on:i_off, i+4]+i,fc[i],linestyle='dashed',lw=0.5)
            # ax1.plot(time_vec[i_on:i_off], out[i_on:i_off, i+4]+i,fc[i],linestyle='dashed',lw=0.5)
            ax1.plot([time_vec[i_on],time_vec[i_off]],[i]*2,c='k',lw=0.5)
        ax1.set_ylim((-0.1,4.1))
        ax1.set_yticks(np.arange(0,4.5,0.5))
        ax1.set_yticklabels(['','lh','','rh','','lf','','rf',''])

        dfp =df.iloc[i_fq:i_fq_off+1]
        ax2 = fig.add_subplot(grid[-1,-1],projection='polar')
        plot_avg_gait_phase(dfp,"",ax2)
        
        ax4 = fig.add_subplot(grid[-3,:-1],sharex=ax1)
        x = dfp.ro+dfp.phase_dur*0.5
        for phd,color in zip([dfp.LR_h,dfp.hl,dfp.diag,dfp.LR_f],[fc[1],fc[2],fc[3],fc[4]]):
            plot_circ_line(x.values,phd.values,ax4,c=color,lw=1)
            if True:
                flp=0.125
                plot_circ_line(x.values,ma_circ(phd.values*2*np.pi,flp)/(2*np.pi),ax4,c=color,lw=1,alpha=0.5)
            for y_ in [0.0,0.5,1.0]:
                ax4.plot([time_vec[i_on],time_vec[i_off]],[y_]*2,c='k',lw=0.5,linestyle='dashed')
            ax4.scatter(x,phd,c=color,s=8)
        ax4.set_ylim([-0.05,1.05])
        ax3 = fig.add_subplot(grid[-2,:-1],sharex=ax1)
        
        times_ = times[np.logical_and(times[:,0]>=time_vec[i_on-NADD],times[:,0]<=time_vec[i_off+NADD])]
        plot_bout(ax3,times_,dfp)

        ax5 = fig.add_subplot(grid[-1,:-1],sharex=ax1)
        ax5.plot(time_vec[i_on:i_off],alphas[i_on:i_off],c='k')
        ax1.set_xlim([time_vec[i_on],time_vec[i_off]])

        plt.show()
        
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    
    plt.show()
    