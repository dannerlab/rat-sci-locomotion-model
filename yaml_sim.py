# 
# This file is part of https://github.com/dannerlab/rat-sci-locomotion-model.
# Copyright (c) 2025 Simon M. Danner.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


import yaml
yamlload = lambda fn: yaml.load(open(fn),Loader=yaml.FullLoader)

def update_sim_from_yaml(config,cpg_sim):
    update_history = {}
    def update_variable(var,factor=None,value_new=None):
        value = cpg_sim.sim.getVariableValue(var)
        if (factor is not None) and (value_new is None):
            value_new = value * factor
        elif (factor is not None) and (value_new is not None):
            print('======== error both factor and value specified')
            exit()
        elif (factor is None) and (value_new is None):
            print('======== error neither factor nor value specified')
            exit()
        cpg_sim.sim.updateVariable(var,value_new)
        if var in update_history.keys():
            print('======== warning variable',var,'already updated')
            update_history[var] = update_history[var]+[value_new]
        else: 
            update_history[var] = [value,value_new]
        print('{:20} {:8.7f} -> {:8.7f}'.format(var,value,value_new))

    print('\nupdating variables')
    
    for ue in config['update']:
        if 'variable_group' in ue:
            #print('group update - name: {:8}, group: {:8}, postfix: {:8}, {:8}: {:8}'.format(ue['name'],ue['variable_group'],ue['postfix'],'factor' ,'a'))
            if 'postfix' in ue:
                if 'factor' in ue:
                    print('group update - name:',ue['name'],
                        ', group:',ue['variable_group'],
                        ', postfix:',ue['postfix'],
                        ', factor:',ue['factor'])
                elif 'value' in ue:
                    print('group update - name:',ue['name'],
                        ', group:',ue['variable_group'],
                        ', postfix:',ue['postfix'],
                        ', value:',ue['value'])
            else:
                if 'factor' in ue:
                    print('group update - name:',ue['name'],
                        ', group:',ue['variable_group'],
                        ', factor:',ue['factor'])
                elif 'value' in ue:
                    print('group update - name:',ue['name'],
                        ', group:',ue['variable_group'],
                        ', value:',ue['value'])
            print('{:20}  {:8} -> {:8}'.format('variable name','from','to'))
            for v in config['variable_groups'][ue['variable_group']]:
                if 'postfix' in ue:
                    for pf in ue['postfix']:
                        var = v+pf
                        if 'factor' in ue:
                            update_variable(var,factor=ue['factor'])
                        elif 'value' in ue:
                            update_variable(var,value_new=ue['value'])
                else:
                    if 'factor' in ue:
                        update_variable(v,factor=ue['factor'])
                    elif 'value' in ue:
                        update_variable(v,value_new=ue['value'])
            print('\n')
        elif 'variable' in ue:
            var = ue['variable']
            if 'factor' in ue:
                print('variable update:',var,
                      ', factor:',ue['factor'])
                update_variable(var,factor=ue['factor'])
            elif 'value' in ue:
                value = ue['value']
                update_variable(var,value_new=ue['value'])

    print('\n')
    return update_history
    