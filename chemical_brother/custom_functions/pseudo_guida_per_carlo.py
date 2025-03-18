
from modules_files.Classifiier_function import *
from modules_files.usefull_functions import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
# plt.ion()
mpl.use('macosx')
#%%    load data
python_session = 'X&Y_train_X&Y_test_RANDOMIZED.pkl'
# dill.load_module(python_session)
with open(python_session, 'rb') as file:
    loaded_variables = pickle.load(file)

# Access the variables from the loaded dictionary
X_train         = loaded_variables['X_train'       ]
y_train         = loaded_variables['y_train'       ]
X_test          = loaded_variables['X_test'        ]
y_test          = loaded_variables['y_test'        ]
substance_set   = loaded_variables['substance_set' ]
all_dataset     = loaded_variables['all_dataset'   ]
DATA_folder     = loaded_variables['DATA_folder'   ]
lab_folder     = loaded_variables['lab_folder'   ]
# %%    degradation simulation
t0=0
t1=30
t2=130
t3= 200

passo = 5

time=range(t0, t3+passo , passo)
# t1_v=[t1, t1, t1, t1, t1, t1+5, t1+10, t1+15, t1+20, t1+25]
t1_v=[t1, t1, t1, t1, t1, t1, t1, t1, t1, t1] # inizio degrado
t2_v=[t2, t2+10, t2, t2, t2, t2-12, t2, t2, t2, t2]# completamento degrado

mode_array =        ["log","lin","log","exp","exp","exp","log","exp","exp", "exp"]
t2_v                = [   t2,   t2,  t2,  t2,  t2,  t3,  t3,  t3,  t3,  t3]
final_value_array   = [ 1.2 ,   2 ,1.2 ,  2 ,1.2 ,   0,   0,   0,   0,   0]
# final_value_array = [10, 10,    0,  1.2,    0,  1.2,    0,  1.2,    0,   1.2]
# mode_array = ["log","log","log","log","log","log","log","log","log", "log"]
# final_value_array = [0,1.2,0,1.2,0,1.2,0,1.2,0,1.2]
# final_value_array = [0,0,0,0,0,0,0,0,0,0]


from usefull_functions_per_quel_cazzone_di_carlo import*

#%%    array_model generation

DEGRADATED_DB  = [None for _ in range(   len(time)   )]
DEGRADATED_DB_PCA  = [None for _ in range(   len(time)   )]

import copy

for i,t in enumerate(time):

    XX_test = copy.deepcopy(X_test)
    deg_X = degradate(XX_test, t1_v, t2_v, t, mode_array = mode_array ,  final_values_array = final_value_array)
    deg_X_pca = pca.transform(deg_X)
    deg_X = degradate(X, t1_v, t2_v, t, mode_array=mode_array,final_values_array=final_value_array)

    DEGRADATED_DB[i] = deg_X
    DEGRADATED_DB_PCA[i] = deg_X_pca

import dill
dill.dump_session('./DEGRADATED_DB_4_comparison.pkl')
