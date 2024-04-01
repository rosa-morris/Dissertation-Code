# %%
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd 
from rdkit.Chem import Descriptors
from rdkit.Chem import SaltRemover
import numpy as np

# %%
df = PandasTools.LoadSDF('/Users/RosaMorris/Desktop/datawarriorpka.sdf')

# %%
df.insert(8, 'pkafloat', df['pKa'].astype(float))

# %%
ROMols = df['ROMol']

# %%
SMILES = []
for x in ROMols:
    SMILE = Chem.MolToSmiles(x)
    SMILES.append(SMILE)

# %%
df.insert(12, 'SMILES', SMILES)

# %%
dftest = df[~df['SMILES'].str.contains('Si|Se|B[A-Z]|B[(]|B$')]

# %%
dftest = dftest[~dftest['Row-ID'].str.contains('^742$')]

# %%
dftest = dftest.query('basicOrAcidic == "acidic"')

# %%
dftest = dftest.query('pkafloat > 0 and pkafloat < 14')

# %%
dftest = dftest.reset_index(drop=True)

# %%
dftest

# %%
%matplotlib inline
import matplotlib.pyplot as plt

# %%
plt.hist(dftest['pkafloat'], bins=28, color='lightblue', edgecolor='grey', alpha=0.7)
plt.title('Distribution of Acidic pKa Values')
plt.xlabel('pKa')
plt.ylabel('Frequency')
plt.show()

# %%
ROMolstest = dftest['ROMol']

# %%
stripped_molecules = []

remover = SaltRemover.SaltRemover()

for mol in ROMolstest:
    if mol is not None:
        stripped_mol = remover.StripMol(mol)
        stripped_molecules.append(stripped_mol)

# %%
dftest.insert(14,'ROMolstripped', stripped_molecules)

# %%
def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:

        try:
            val = fn(mol)
        except:
        
            import traceback
            traceback.print_exc()

            val = missingVal
        res[nm] = val
    return res

# %%
allDescrs = [getMolDescriptors(x) for x in ROMolstest]

# %%
df_alldescrs = pd.DataFrame(allDescrs)

# %%
df_alldescrs.insert(0,'pKa', dftest['pKa'])

# %%
df_alldescrs2 = df_alldescrs.drop('Ipc', axis=1)
df_alldescrs2 = df_alldescrs2.dropna()

# %%
df_alldescrs2 = df_alldescrs2.reset_index(drop=True)

# %%
df_alldescrs2

# %%
pip install --user xgboost

# %%
pip install -U scikit-learn as sklearn

# %%
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np

# %%
Xdesc, Ydesc = df_alldescrs2.drop("pKa", axis=1), pd.DataFrame(df_alldescrs2[["pKa"]])

# %%
df_alldescrs2

# %%
Xdesc_train, Xdesc_test, Ydesc_train, Ydesc_test = train_test_split(Xdesc, Ydesc['pKa'], test_size=0.25, random_state=1)

# %%
modelxgb = XGBRegressor()
modelxgb.fit(Xdesc_train, Ydesc_train)

# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import absolute
from sklearn.model_selection import KFold

# %%
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
maescoresxgb = cross_val_score(modelxgb, Xdesc_train, Ydesc_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
rmsescoresxgb = cross_val_score(modelxgb, Xdesc_train, Ydesc_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
r2_scoresxgb = cross_val_score(modelxgb, Xdesc_train, Ydesc_train, scoring='r2', cv=cv, n_jobs=-1)

# %%
rmse_scoresxgb = np.sqrt(-rmsescoresxgb)
mae_scoresxgb = absolute(maescoresxgb)


# %%
print(f'MAE: {mae_scoresxgb.mean()},({mae_scoresxgb.std()})')
print(f'RMSE: {rmse_scoresxgb.mean()},({rmse_scoresxgb.std()})')
print(f'R-squared: {r2_scoresxgb.mean()}, ({r2_scoresxgb.std()})')

# %%
predictionsxgb = modelxgb.predict(Xdesc_test)

# %%
Ydesc_test

# %%
predictionsxgb

# %%
df_predictionsxgb = pd.DataFrame(predictionsxgb)
df_Ydesc_test = pd.DataFrame(Ydesc_test)

# %%
msexgb = mean_squared_error(Ydesc_test, predictionsxgb)
maexgb = mean_absolute_error(Ydesc_test, predictionsxgb)
rmsexgb = np.sqrt(msexgb)
r2xgb = r2_score(Ydesc_test, predictionsxgb)

# %%
print(f'MAE: {maexgb}')
print(f'RMSE: {rmsexgb}')
print(f'R-squared: {r2xgb}')

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
modelrf = RandomForestRegressor(n_estimators=1000,oob_score =True, bootstrap=True, random_state=42)
modelrf.fit(Xdesc_train, Ydesc_train)

# %%
maescoresrf = cross_val_score(modelrf, Xdesc_train, Ydesc_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresrf = cross_val_score(modelrf, Xdesc_train, Ydesc_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresrf = cross_val_score(modelrf, Xdesc_train, Ydesc_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresrf = np.sqrt(-rmsescoresrf)
mae_scoresrf = absolute(maescoresrf)


# %%
print(f'MAE: {mae_scoresrf.mean()},({mae_scoresrf.std()})')
print(f'RMSE: {rmse_scoresrf.mean()},({rmse_scoresrf.std()})')
print(f'R-squared: {r2_scoresrf.mean()}')

# %%
oob_predictionsrf = modelrf.oob_prediction_
oob_maerf = mean_absolute_error(Ydesc_train, oob_predictionsrf)
oob_rmserf = np.sqrt(mean_squared_error(Ydesc_train, oob_predictionsrf))

# %%
print(f'OOB MAE: {oob_maerf}')
print(f'OOB RMSE: {oob_rmserf}')
print(f'R-squared: {modelrf.oob_score_}')

# %%
predictionsrf = modelrf.predict(Xdesc_test)

# %%
mserf = mean_squared_error(Ydesc_test, predictionsrf)
maerf = mean_absolute_error(Ydesc_test, predictionsrf)
rmserf = np.sqrt(mserf)
r2rf = r2_score(Ydesc_test, predictionsrf)

# %%
print(f'MAE: {maerf}')
print(f'RMSE: {rmserf}')
print(f'R-squared: {r2rf}')

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid = {'max_depth': [None, 10, 20, 30, 40]}

# %%
grid_search = GridSearchCV(modelrf, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(Xdesc_train, Ydesc_train)

# %%
max_depth_valuesrf = param_grid['max_depth']
mean_test_scoresrf = np.sqrt(-grid_search.cv_results_['mean_test_score'])
std_test_scoresrf = grid_search.cv_results_['std_test_score']

# %%
max_depth_valuesrf_plot = max_depth_valuesrf[1:]
mean_test_scoresrf_plot = mean_test_scoresrf[1:]
std_test_scoresrf_plot = std_test_scoresrf[1:]

# %%
plt.errorbar(max_depth_valuesrf_plot, mean_test_scoresrf_plot, yerr=std_test_scoresrf_plot, marker='x', linestyle='-', color='black')
ymin, ymax = 1.55, 1.66
plt.ylim(ymin, ymax)
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.xticks(max_depth_valuesrf_plot)
plt.axhline(y=mean_test_scoresrf[0], color='red', linestyle='--', label=f'y = {mean_test_scores[0]}')
plt.grid(True)
plt.show()

# %%
param_grid2 =  {'max_depth': [ 26, 28, 30, 32, 34]}

# %%
grid_search2 = GridSearchCV(modelrf, param_grid2, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search2.fit(Xdesc_train, Ydesc_train)

# %%
max_depth_values2 = param_grid2['max_depth']
mean_test_scores2 = np.sqrt(-grid_search2.cv_results_['mean_test_score'])
std_test_scores2 = grid_search2.cv_results_['std_test_score']

# %%
plt.errorbar(max_depth_values2, mean_test_scores2, yerr=std_test_scores2, marker='x', linestyle='--', color='black')
ymin, ymax = 1.57, 1.58
plt.ylim(ymin, ymax)
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.xticks(max_depth_values2)
plt.axhline(y=mean_test_scores[0], color='red', linestyle='--', label=f'y = {mean_test_scores[0]}')
plt.grid(True)
plt.show()

# %%
param_gridMAE = {'max_depth': [ None, 26, 28, 30, 32, 34]}

# %%
grid_searchMAE = GridSearchCV(modelrf, param_gridMAE, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_searchMAE.fit(Xdesc_train, Ydesc_train)

# %%
max_depth_valuesMAE = param_gridMAE['max_depth']
mean_test_scoresMAE = (-grid_searchMAE.cv_results_['mean_test_score'])
std_test_scoresMAE = grid_searchMAE.cv_results_['std_test_score']

# %%
max_depth_valuesMAE_plot = max_depth_valuesMAE[1:]
mean_test_scoresMAE_plot = mean_test_scoresMAE[1:]
std_test_scoresMAE_plot = std_test_scoresMAE[1:]

# %%
plt.errorbar(max_depth_valuesMAE_plot, mean_test_scoresMAE_plot, yerr=std_test_scoresMAE_plot, marker='x', linestyle='--', color='black')
ymin, ymax = 1.02, 1.03
plt.ylim(ymin, ymax)
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('MAE')
plt.xticks(max_depth_valuesMAE_plot)
plt.axhline(y=mean_test_scoresMAE[0], color='red', linestyle='--', label=f'y = {mean_test_scores[0]}')
plt.grid(True)
plt.show()

# %%
param_gridr2 = {'max_depth': [ None, 26, 28, 30, 32, 34]}

# %%
grid_searchr2 = GridSearchCV(modelrf, param_gridr2, scoring='r2', cv=5, n_jobs=-1)
grid_searchr2.fit(Xdesc_train, Ydesc_train)

# %%
max_depth_valuesr2 = param_gridr2['max_depth']
mean_test_scoresr2 = -(-grid_searchr2.cv_results_['mean_test_score'])
std_test_scoresr2 = grid_searchr2.cv_results_['std_test_score']

# %%
max_depth_valuesr2_plot = max_depth_valuesr2[1:]
mean_test_scoresr2_plot = mean_test_scoresr2[1:]
std_test_scoresr2_plot = std_test_scoresr2[1:]

# %%
mean_test_scoresr2

# %%
plt.errorbar(max_depth_valuesr2_plot, mean_test_scoresr2_plot, yerr=std_test_scoresr2_plot, marker='x', linestyle='--', color='black')
ymin, ymax = 0.755, 0.765
plt.ylim(ymin, ymax)
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('R2')
plt.xticks(max_depth_valuesr2_plot)
plt.axhline(y=mean_test_scoresr2[0], color='red', linestyle='--', label=f'y = {mean_test_scores[0]}')
plt.grid(True)
plt.show()

# %%
modelrf2 = RandomForestRegressor(n_estimators=1000,oob_score =True, max_depth = 30, bootstrap=True, random_state=42)
modelrf2.fit(Xdesc_train, Ydesc_train)

# %%
maescoresrf2 = cross_val_score(modelrf2, Xdesc_train, Ydesc_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresrf2 = cross_val_score(modelrf2, Xdesc_train, Ydesc_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresrf2 = cross_val_score(modelrf2, Xdesc_train, Ydesc_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresrf2 = np.sqrt(-rmsescoresrf2)
mae_scoresrf2 = absolute(maescoresrf2)

# %%
print(f'MAE: {mae_scoresrf2.mean()},({mae_scoresrf2.std()})')
print(f'RMSE: {rmse_scoresrf2.mean()},({rmse_scoresrf2.std()})')
print(f'R-squared: {r2_scoresrf2.mean()}, ({r2_scoresrf2.std()})')

# %%
oob_predictionsrf2 = modelrf2.oob_prediction_
oob_maerf2 = mean_absolute_error(Ydesc_train, oob_predictionsrf2)
oob_rmserf2 = np.sqrt(mean_squared_error(Ydesc_train, oob_predictionsrf2))

# %%
print(f'OOB MAE: {oob_maerf2}')
print(f'OOB RMSE: {oob_rmserf2}')
print(f'R-squared: {modelrf2.oob_score_}')

# %%
predictionsrf2 = modelrf2.predict(Xdesc_test)

# %%
mserf2 = mean_squared_error(Ydesc_test, predictionsrf2)
maerf2 = mean_absolute_error(Ydesc_test, predictionsrf2)
rmserf2 = np.sqrt(mserf2)
r2rf2 = r2_score(Ydesc_test, predictionsrf2)
print(f'MAE: {maerf2}')
print(f'RMSE: {rmserf2}')
print(f'R-squared: {r2rf2}')

# %%
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# %%
scaler = StandardScaler()
Xdesc_train_scaled = scaler.fit_transform(Xdesc_train)
Xdesc_test_scaled = scaler.transform(Xdesc_test)

# %%
modelsvm = SVR(kernel = 'rbf')
modelsvm.fit(Xdesc_train_scaled, Ydesc_train)

# %%
maescoressvm = cross_val_score(modelsvm, Xdesc_train_scaled, Ydesc_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoressvm = cross_val_score(modelsvm, Xdesc_train_scaled, Ydesc_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoressvm = cross_val_score(modelsvm, Xdesc_train_scaled, Ydesc_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoressvm = np.sqrt(-rmsescoressvm)
mae_scoressvm = absolute(maescoressvm)

# %%
print(f'MAE: {mae_scoressvm.mean()},({mae_scoressvm.std()})')
print(f'RMSE: {rmse_scoressvm.mean()},({rmse_scoressvm.std()})')
print(f'R-squared: {r2_scoressvm.mean()}, ({r2_scoressvm.std()})')

# %%
predictionssvm = modelsvm.predict(Xdesc_test_scaled)

# %%
msesvm = mean_squared_error(Ydesc_test, predictionssvm)
maesvm = mean_absolute_error(Ydesc_test, predictionssvm)
rmsesvm = np.sqrt(msesvm)
r2svm = r2_score(Ydesc_test, predictionssvm)
print(f'MAE: {maesvm}')
print(f'RMSE: {rmsesvm}')
print(f'R-squared: {r2svm}')

# %%
from sklearn.neural_network import MLPRegressor

# %%


# %%
modelmlp = MLPRegressor(hidden_layer_sizes =(200, 200), max_iter=300, random_state = 42)
modelmlp.fit(Xdesc_train_scaled, Ydesc_train)

# %%
maescoresmlp = cross_val_score(modelmlp, Xdesc_train_scaled, Ydesc_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresmlp = cross_val_score(modelmlp, Xdesc_train_scaled, Ydesc_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresmlp = cross_val_score(modelmlp, Xdesc_train_scaled, Ydesc_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresmlp = np.sqrt(-rmsescoresmlp)
mae_scoresmlp = absolute(maescoresmlp)

# %%
print(f'MAE: {mae_scoresmlp.mean()},({mae_scoresmlp.std()})')
print(f'RMSE: {rmse_scoresmlp.mean()},({rmse_scoresmlp.std()})')
print(f'R-squared: {r2_scoresmlp.mean()}, ({r2_scoresmlp.std()})')

# %%
predictionsmlp = modelmlp.predict(Xdesc_test_scaled)

# %%
msemlp = mean_squared_error(Ydesc_test, predictionsmlp)
maemlp = mean_absolute_error(Ydesc_test, predictionsmlp)
rmsemlp = np.sqrt(msemlp)
r2mlp = r2_score(Ydesc_test, predictionsmlp)
print(f'MAE: {maemlp}')
print(f'RMSE: {rmsemlp}')
print(f'R-squared: {r2mlp}')

# %%
from sklearn.ensemble import BaggingRegressor

# %%
modelbagging = BaggingRegressor(n_estimators = 1000, oob_score = True, bootstrap = True, random_state = 42)
modelbagging.fit (Xdesc_train, Ydesc_train)

# %%
maescoresbagging = cross_val_score(modelbagging, Xdesc_train, Ydesc_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresbagging = cross_val_score(modelbagging, Xdesc_train, Ydesc_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresbagging = cross_val_score(modelbagging, Xdesc_train, Ydesc_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresbagging = np.sqrt(-rmsescoresbagging)
mae_scoresbagging = absolute(maescoresbagging)

# %%
print(f'MAE: {mae_scoresbagging.mean()},({mae_scoresbagging.std()})')
print(f'RMSE: {rmse_scoresbagging.mean()},({rmse_scoresbagging.std()})')
print(f'R-squared: {r2_scoresbagging.mean()}, ({r2_scoresbagging.std()})')

# %%
oob_predictionsbagging = modelbagging.oob_prediction_
oob_maebagging = mean_absolute_error(Ydesc_train, oob_predictionsbagging)
oob_rmsebagging = np.sqrt(mean_squared_error(Ydesc_train, oob_predictionsbagging))

# %%
print(f'OOB MAE: {oob_maebagging}')
print(f'OOB RMSE: {oob_rmsebagging}')
print(f'R-squared: {modelbagging.oob_score_}')

# %%
predictionsbagging = modelbagging.predict(Xdesc_test)

# %%
msebagging = mean_squared_error(Ydesc_test, predictionsbagging)
maebagging = mean_absolute_error(Ydesc_test, predictionsbagging)
rmsebagging = np.sqrt(msebagging)
r2bagging = r2_score(Ydesc_test, predictionsbagging)
print(f'MAE: {maebagging}')
print(f'RMSE: {rmsebagging}')
print(f'R-squared: {r2bagging}')

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
Ydesc_testKNN = pd.to_numeric(Ydesc_test, errors='coerce') 

# %%
Ydesc_trainKNN = pd.to_numeric(Ydesc_train, errors='coerce')

# %%
modelKNN = KNeighborsRegressor()
modelKNN.fit(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
param_gridKNN = {'n_neighbors': [ 1,2,3,4]}

# %%
grid_searchKNNMAE = GridSearchCV(modelKNN, param_gridKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_searchKNNMAE.fit(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
n_neighbors_values = param_gridKNN['n_neighbors']
mean_test_scoresKNNMAE = (-grid_searchKNNMAE.cv_results_['mean_test_score'])
std_test_scoresKNNMAE = grid_searchKNNMAE.cv_results_['std_test_score']

# %%
mean_test_scoresKNNMAE

# %%
grid_searchKNNRMSE = GridSearchCV(modelKNN, param_gridKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_searchKNNRMSE.fit(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
mean_test_scoresKNNRMSE = np.sqrt(-grid_searchKNNRMSE.cv_results_['mean_test_score'])
std_test_scoresKNNRMSE = grid_searchKNNRMSE.cv_results_['std_test_score']

# %%
mean_test_scoresKNNRMSE

# %%
grid_searchKNNr2 = GridSearchCV(modelKNN, param_gridKNN, scoring='r2', cv=5, n_jobs=-1)
grid_searchKNNr2.fit(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
mean_test_scoresKNNr2 =(grid_searchKNNr2.cv_results_['mean_test_score'])
std_test_scoresKNNr2 = grid_searchKNNr2.cv_results_['std_test_score']

# %%
mean_test_scoresKNNr2

# %%
errorsKNN = mean_test_scoresKNNMAE + mean_test_scoresKNNRMSE

# %%

fig, ax1 = plt.subplots()
ax1.plot(n_neighbors_values, mean_test_scoresKNNMAE, marker='x', linestyle='--', color='black', label='MAE')
ax1.set_xlabel('n_neighbors')
ax1.set_ylabel('MAE', color='black')
ax1.tick_params('y', colors='black')
ax1.set_title('Grid Search Results for KNN')
ax1.grid(True)
ax1.set_xticks(np.arange(1, 5, 1))
ax2 = ax1.twinx()
ax2.plot(n_neighbors_values, mean_test_scoresKNNRMSE, marker='x', linestyle='--', color='blue', label='RMSE')
ax2.set_ylabel('RMSE', color='blue')
ax2.tick_params('y', colors='blue')




# %%
modelKNN3 = KNeighborsRegressor(n_neighbors=3)
modelKNN3.fit(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
predictionsKNN3 = modelKNN3.predict(Xdesc_test_scaled)

# %%
mseKNN3 = mean_squared_error(Ydesc_test, predictionsKNN3)
maeKNN3 = mean_absolute_error(Ydesc_test, predictionsKNN3)
rmseKNN3 = np.sqrt(mseKNN3)
r2KNN3 = r2_score(Ydesc_test, predictionsKNN3)
print(f'MAE: { maeKNN3}')
print(f'RMSE: {rmseKNN3}')
print(f'R-squared: {r2KNN3}')

# %%
maescoresKNN3 = cross_val_score(modelKNN3, Xdesc_train_scaled, Ydesc_trainKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresKNN3 = cross_val_score(modelKNN3, Xdesc_train_scaled, Ydesc_trainKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresKNN3 = cross_val_score(modelKNN3, Xdesc_train_scaled, Ydesc_trainKNN, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresKNN3 = np.sqrt(-rmsescoresKNN3)
mae_scoresKNN3 = absolute(maescoresKNN3)

# %%
print(f'MAE: {mae_scoresKNN3.mean()},({mae_scoresKNN3.std()})')
print(f'RMSE: {rmse_scoresKNN3.mean()},({rmse_scoresKNN3.std()})')
print(f'R-squared: {r2_scoresKNN3.mean()}, ({r2_scoresKNN3.std()})')

# %%
from sklearn.feature_selection import VarianceThreshold

# %%
selector = VarianceThreshold(threshold=0) 

# %%
Xdesc_train_scaled_KNNtest = selector.fit_transform(Xdesc_train_scaled)

# %%
selector2 = VarianceThreshold(threshold=0.01 * Xdesc_train_scaled.var().max())
Xdesc_train_scaled_KNNtest2 = selector2.fit_transform(Xdesc_train_scaled)

# %%
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

# %%
mi_scores = mutual_info_regression(Xdesc_train_scaled, Ydesc_trainKNN)

# %%
Xdesc_train_scaled[:, mi_scores > 0.035].shape

# %%
maeKNN_values = []
rmseKNN_values = []
k_values = np.arange(0, 0.15, 0.005)
for k in k_values:
    selected_features_train = Xdesc_train_scaled[:, mi_scores > k]
    selected_features_test = Xdesc_test_scaled[:, mi_scores > k]
    modelKNNselected = KNeighborsRegressor(n_neighbors=3)
    modelKNNselected.fit(selected_features_train, Ydesc_trainKNN)
    predictionsKNNselected = modelKNNselected.predict(selected_features_test)
    mseKNNselected = mean_squared_error(Ydesc_testKNN, predictionsKNNselected)
    maeKNNselected = mean_absolute_error(Ydesc_testKNN, predictionsKNNselected)
    rmseKNNselected = np.sqrt(mseKNNselected)
    r2KNNselected = r2_score(Ydesc_testKNN, predictionsKNNselected)

    maeKNN_values.append(maeKNNselected)
    rmseKNN_values.append(rmseKNNselected)

    print(f'For k={k}:')
    print(f'MAE: {maeKNNselected}')
    print(f'RMSE: {rmseKNNselected}')
    print(f'R-squared: {r2KNNselected}')
    print('-' * 30)


# %%
plt.plot(k_values, maeKNN_values, marker = 'x')

# %%
modelKNNreduced = KNeighborsRegressor(n_neighbors=3)
modelKNNreduced.fit(Xdesc_train_scaled[:, mi_scores > 0.035], Ydesc_trainKNN)

# %%
maescoresKNNreduced = cross_val_score(modelKNNreduced, Xdesc_train_scaled[:, mi_scores > 0.035], Ydesc_trainKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresKNNreduced = cross_val_score(modelKNNreduced, Xdesc_train_scaled[:, mi_scores > 0.035], Ydesc_trainKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresKNNreduced = cross_val_score(modelKNNreduced, Xdesc_train_scaled[:, mi_scores > 0.035], Ydesc_trainKNN, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresKNNreduced = np.sqrt(-rmsescoresKNNreduced)
mae_scoresKNNreduced = absolute(maescoresKNNreduced)

# %%
print(f'MAE: {mae_scoresKNNreduced.mean()},({mae_scoresKNNreduced.std()})')
print(f'RMSE: {rmse_scoresKNNreduced.mean()},({rmse_scoresKNNreduced.std()})')
print(f'R-squared: {r2_scoresKNNreduced.mean()}, ({r2_scoresKNNreduced.std()})')

# %%
predictionsKNNreduced = modelKNNreduced.predict(Xdesc_test_scaled[:, mi_scores > 0.035])

# %%
mseKNNreduced = mean_squared_error(Ydesc_testKNN, predictionsKNNreduced)
maeKNNreduced = mean_absolute_error(Ydesc_testKNN, predictionsKNNreduced)
rmseKNNreduced = np.sqrt(mseKNNreduced)
r2KNNreduced = r2_score(Ydesc_testKNN, predictionsKNNreduced)
print(f'MAE: { maeKNNreduced}')
print(f'RMSE: {rmseKNNreduced}')
print(f'R-squared: {r2KNNreduced}')

# %%
pip install statsmodels

# %%
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

# %%
Ydesc_test_array = Ydesc_test.astype(float)

# %%
correlation_coefficient, p_valuecc = pearsonr(predictionsxgb, Ydesc_test_array)

# %%
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_valuecc}")

# %%
absolute_errorsxgb = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsxgb)]
absolute_errorsrf = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsrf2)]
absolute_errorssvm = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionssvm)]
absolute_errorsmlp = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsmlp)]
absolute_errorsbagging = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsbagging)]
absolute_errorsKNN = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsKNN3)]
absolute_errorsKNNreduced = [abs(actual - predicted) for actual, predicted in zip(Ydesc_test_array, predictionsKNNreduced)]


# %%
t_statistic, p_valuet = ttest_rel(absolute_errorsrf, absolute_errorsKNNreduced)

# %%
print(f"T-statistic {t_statistic}")
print(f"P-value: {p_valuet}")


