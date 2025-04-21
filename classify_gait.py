import numpy as np
import pandas as pd

def classify_gait(df):
    gaits = {
            'pronk': np.array([0.0,0.0,0.0]),
        
            'trot': np.array([0.5,0.5,0.0]),
            'bound': np.array([0.0,0.5,0.5]),
            'bound_2': np.array([0.0,2/3,2/3]),
            'pace': np.array([0.5,1.0,0.5]),
        
            'hbound': np.array([0.0,1/3,2/3]),
            'hbound_2': np.array([0.0,2/3,1/3]),
        
            'canter': np.array([2/3,1/3,0]),
            'canter_2': np.array([1/3,1/3,2/3]),
            'canter_3': np.array([1/3,2/3,0]),
            'canter_4': np.array([2/3,2/3,1/3]),
        
            'rhbound': np.array([1/3,2/3,2/3]),
            'rhbound_2': np.array([2/3,1/3,1/3]),
        
            'lcanter': np.array([1/3,0,2/3]),
            'lcanter_2': np.array([2/3,0,1/3]),
            'lcanter_3': np.array([1/3,2/3,1/3]),
            'lcanter_4': np.array([2/3,1/3,2/3]),
            
            'rot_1': np.array([0.75,0.25,0.5]),
            'rot_2': np.array([0.25,0.75,0.5]),
            'lat_seq': np.array([0.5,0.25,0.75]),
            'diag_seq': np.array([0.5,0.75,0.25]),
            'trans_1': np.array([0.75,0.5,0.25]),
            'trans_2': np.array([0.25,0.5,0.75]),
            }
    

    D = df[['LR_h','hl','diag']]
    temp = pd.DataFrame()
    for gait, m_gait in gaits.items():
        tp=np.abs(m_gait-D)
        tp[tp>0.5]=tp[tp>0.5]-1
        temp[gait] = np.sum((tp)**2.0,axis=1)
    #for gait, m_gait in gaits.items():
    #    temp[gait] = np.sqrt(np.sum(np.angle(np.exp(1j * (m_gait - D) * 2 * np.pi)) ** 2, axis=1))/(np.pi*2)

    gaits=temp.idxmin(axis=1)
    gaits[np.any(pd.isna(D),axis=1)]=np.nan
    gaits[gaits=='bound_2'] = 'bound'

    gaits[(df.duty_factor>=0.5)&(gaits=='lat_seq')]='walk_lat'
    gaits[(df.duty_factor>=0.5)&(gaits=='diag_seq')]='walk_diag'
    gaits[(df.duty_factor<0.5)&((gaits=='lat_seq'))]='run_lat'
    gaits[(df.duty_factor<0.5)&((gaits=='diag_seq'))]='run_diag'
    
    
    gaits[(df.duty_factor>=0.5)&((gaits=='rot_1')|(gaits=='rot_2'))]='walk_rot'
    gaits[(df.duty_factor>=0.5)&((gaits=='trans_1')|(gaits=='trans_2'))]='walk_trans'
    df['gaits_all2'] = pd.Categorical(
        gaits)
    gaits[(df.duty_factor<0.5)&((gaits=='rot_1')|(gaits=='rot_2'))]='gallop_rot'
    gaits[(df.duty_factor<0.5)&((gaits=='trans_1')|(gaits=='trans_2'))]='gallop_trans'
    gaits[gaits=='hbound_2'] = 'hbound'
    gaits[gaits=='rhbound_2'] = 'rhbound'
    gaits[gaits=='canter_2'] = 'canter'
    gaits[gaits=='canter_3'] = 'rcanter'
    gaits[gaits=='canter_4'] = 'rcanter'
    gaits[gaits=='lcanter_2'] = 'lcanter'
    gaits[gaits=='lcanter_3'] = 'lcanter'
    gaits[gaits=='lcanter_4'] = 'lcanter'
    gaits[(df.duty_factor>=0.5)&(gaits=='bound')]='hop'
    gaits[(df.duty_factor>=0.5)&(gaits=='hbound')]='hop'
    
    df['gaits_all'] = pd.Categorical(
        gaits)
    df['gaits'] = pd.Categorical(
        ['nd'] * len(df), ['nd', 'walk', 'trot','run','pace','canter','gallop', 'hbound', 'bound'])

    df.loc[df.gaits_all=='walk_trans',"gaits"] = 'walk'
    df.loc[df.gaits_all=='walk_rot',"gaits"] = 'walk'
    df.loc[df.gaits_all=='walk_lat',"gaits"] = 'walk'
    df.loc[df.gaits_all=='walk_diag',"gaits"] = 'walk'
    df.loc[df.gaits_all=='trot',"gaits"] = 'trot'
    df.loc[df.gaits_all=='run_lat',"gaits"] = 'run'
    df.loc[df.gaits_all=='run_diag',"gaits"] = 'run'
    df.loc[df.gaits_all=='gallop_trans',"gaits"] = 'gallop'
    df.loc[df.gaits_all=='gallop_rot',"gaits"] = 'gallop'
    df.loc[df.gaits_all=='hbound',"gaits"] = 'hbound'
    df.loc[df.gaits_all=='bound',"gaits"] = 'bound'
    df.loc[df.gaits_all=='canter',"gaits"] = 'canter'
    df.loc[df.gaits_all=='pace',"gaits"] = 'pace'

    df['gaits2'] = pd.Categorical(
            ['nd'] * len(df), categories=['nd', 'lat_seq','trot','diag_seq','pace','canter','gallop', 'hbound', 'bound'])
    df.loc[df.gaits_all=='walk_lat','gaits2'] = 'lat_seq'
    df.loc[df.gaits_all=='walk_diag','gaits2'] = 'diag_seq'
    df.loc[df.gaits_all=='trot','gaits2'] = 'trot'
    df.loc[df.gaits_all=='run_lat','gaits2'] = 'lat_seq'
    df.loc[df.gaits_all=='run_diag','gaits2'] = 'diag_seq'
    df.loc[df.gaits_all=='gallop_trans','gaits2'] = 'gallop'
    df.loc[df.gaits_all=='gallop_rot','gaits2'] = 'gallop'
    df.loc[df.gaits_all=='hbound','gaits2'] = 'hbound'
    df.loc[df.gaits_all=='bound','gaits2'] = 'bound'
    df.loc[df.gaits_all=='canter','gaits2'] = 'canter'
    df.loc[df.gaits_all=='pace','gaits2'] = 'pace'
    return df