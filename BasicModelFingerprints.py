# %%
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

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
dftest = dftest.query('basicOrAcidic == "basic"')

# %%
dftest = dftest.query('pkafloat > 0 and pkafloat < 14')

# %%
dftest = dftest.reset_index(drop=True)

# %%
plt.hist(dftest['pkafloat'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of pKa Values')
plt.xlabel('pKa')
plt.ylabel('Frequency')
plt.show()

# %%
ROMolstest = dftest['ROMol']

# %%
fingerprints = []
for x in ROMolstest:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=4096)
        fingerprint_list = list(fingerprint)
        fingerprints.append(fingerprint_list)

# %%
XMFF = pd.DataFrame(fingerprints)

# %%
YMFF = pd.DataFrame(dftest['pKa'])

# %%
pip install -U scikit-learn as sklearn

# %%
from sklearn.model_selection import train_test_split

# %%
XMFF_train, XMFF_test, YMFF_train, YMFF_test = train_test_split(XMFF, YMFF['pKa'], test_size=0.25, random_state=1)

# %%
import xgboost as xgb
from xgboost import XGBRegressor

# %%
modelxgb = XGBRegressor()
modelxgb.fit(XMFF_train, YMFF_train)

# %%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import absolute
from sklearn.model_selection import KFold

# %%
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
maescoresxgb = cross_val_score(modelxgb, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
rmsescoresxgb = cross_val_score(modelxgb, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
r2_scoresxgb = cross_val_score(modelxgb, XMFF_train, YMFF_train, scoring='r2', cv=cv, n_jobs=-1)

# %%
rmse_scoresxgb = np.sqrt(-rmsescoresxgb)
mae_scoresxgb = absolute(maescoresxgb)

# %%
print(f'MAE: {mae_scoresxgb.mean()},({mae_scoresxgb.std()})')
print(f'RMSE: {rmse_scoresxgb.mean()},({rmse_scoresxgb.std()})')
print(f'R-squared: {r2_scoresxgb.mean()}, ({r2_scoresxgb.std()})')

# %%
predictionsxgb = modelxgb.predict(XMFF_test)

# %%
msexgb = mean_squared_error(YMFF_test, predictionsxgb)
maexgb = mean_absolute_error(YMFF_test, predictionsxgb)
rmsexgb = np.sqrt(msexgb)
r2xgb = r2_score(YMFF_test, predictionsxgb)

# %%
print(f'MAE: {maexgb}')
print(f'RMSE: {rmsexgb}')
print(f'R-squared: {r2xgb}')

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
modelrf = RandomForestRegressor(n_estimators=1000,oob_score =True, bootstrap=True, random_state=42)
modelrf.fit(XMFF_train, YMFF_train)

# %%
maescoresrf = cross_val_score(modelrf, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresrf = cross_val_score(modelrf, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresrf = cross_val_score(modelrf, XMFF_train, YMFF_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresrf = np.sqrt(-rmsescoresrf)
mae_scoresrf = absolute(maescoresrf)

# %%
print(f'MAE: {mae_scoresrf.mean()},({mae_scoresrf.std()})')
print(f'RMSE: {rmse_scoresrf.mean()},({rmse_scoresrf.std()})')
print(f'R-squared: {r2_scoresrf.mean()}, ({r2_scoresrf.std()})')

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid = {'max_depth': [None, 35, 40, 50]}

# %%
grid_search = GridSearchCV(modelrf, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(XMFF_train, YMFF_train)

# %%
max_depth_values = param_grid['max_depth']
mean_test_scores = np.sqrt(-grid_search.cv_results_['mean_test_score'])
std_test_scores = grid_search.cv_results_['std_test_score']

# %%
max_depth_values_plot = max_depth_values[1:]
mean_test_scores_plot = mean_test_scores[1:]
std_test_scores_plot = std_test_scores[1:]

# %%
plt.errorbar(max_depth_values_plot, mean_test_scores_plot, yerr=std_test_scores_plot, marker='x', linestyle='-', color='black')
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.xticks(max_depth_values_plot)
plt.axhline(y=mean_test_scores[0], color='red', linestyle='--', label=f'y = {mean_test_scores[0]}')
plt.grid(True)
plt.show()

# %%
param_grid2 = {'max_depth': [None, 52, 54, 56, 58, 60]}

# %%
grid_search2 = GridSearchCV(modelrf, param_grid2, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search2.fit(XMFF_train, YMFF_train)

# %%
max_depth_values = param_grid2['max_depth']
mean_test_scores2 = np.sqrt(-grid_search2.cv_results_['mean_test_score'])
std_test_scores2 = grid_search2.cv_results_['std_test_score']

# %%
max_depth_values_plot = max_depth_values[1:]
mean_test_scores2_plot = mean_test_scores2[1:]
std_test_scores2_plot = std_test_scores2[1:]

# %%
plt.errorbar(max_depth_values_plot, mean_test_scores2_plot, yerr=std_test_scores2_plot, marker='x', linestyle='-', color='black')
ymin, ymax = 1.55, 1.58
plt.ylim(ymin, ymax)
plt.title('Grid Search Results for Random Forest')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.xticks(max_depth_values_plot)
plt.axhline(y=mean_test_scores2[0], color='red', linestyle='--', label=f'y = {mean_test_scores2[0]}')
plt.grid(True)
plt.show()

# %%
modelrf2 = RandomForestRegressor(n_estimators=1000, max_depth=58, oob_score =True, bootstrap=True, random_state=42)
modelrf2.fit(XMFF_train, YMFF_train)

# %%
maescoresrf2 = cross_val_score(modelrf2, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresrf2 = cross_val_score(modelrf2, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresrf2 = cross_val_score(modelrf2, XMFF_train, YMFF_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresrf2 = np.sqrt(-rmsescoresrf2)
mae_scoresrf2 = absolute(maescoresrf2)

# %%
print(f'MAE: {mae_scoresrf2.mean()},({mae_scoresrf2.std()})')
print(f'RMSE: {rmse_scoresrf2.mean()},({rmse_scoresrf2.std()})')
print(f'R-squared: {r2_scoresrf2.mean()}, ({r2_scoresrf2.std()})')

# %%
oob_predictions = modelrf2.oob_prediction_
oob_mae = mean_absolute_error(YMFF_train, oob_predictions)
oob_rmse = np.sqrt(mean_squared_error(YMFF_train, oob_predictions))

# %%
print(f'OOB MAE: {oob_mae}')
print(f'OOB RMSE: {oob_rmse}')
print(f'R-squared: {modelrf2.oob_score_}')

# %%
predictionsrf2 = modelrf2.predict(XMFF_test)

# %%
mserf2 = mean_squared_error(YMFF_test, predictionsrf2)
maerf2 = mean_absolute_error(YMFF_test, predictionsrf2)
rmserf2 = np.sqrt(mserf2)
r2rf2 = r2_score(YMFF_test, predictionsrf2)

# %%
print(f'MAE: {maerf2}')
print(f'RMSE: {rmserf2}')
print(f'R-squared: {r2rf2}')

# %%
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# %%
modelsvm = SVR(kernel = 'rbf')
modelsvm.fit(XMFF_train, YMFF_train)

# %%
maescoressvm = cross_val_score(modelsvm, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoressvm = cross_val_score(modelsvm, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoressvm = cross_val_score(modelsvm, XMFF_train, YMFF_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoressvm = np.sqrt(-rmsescoressvm)
mae_scoressvm = absolute(maescoressvm)

# %%
print(f'MAE: {mae_scoressvm.mean()},({mae_scoressvm.std()})')
print(f'RMSE: {rmse_scoressvm.mean()},({rmse_scoressvm.std()})')
print(f'R-squared: {r2_scoressvm.mean()}, ({r2_scoressvm.std()})')

# %%
predictionssvm = modelsvm.predict(XMFF_test)

# %%
msesvm = mean_squared_error(YMFF_test, predictionssvm)
maesvm = mean_absolute_error(YMFF_test, predictionssvm)
rmsesvm = np.sqrt(msesvm)
r2svm = r2_score(YMFF_test, predictionssvm)
print(f'MAE: {maesvm}')
print(f'RMSE: {rmsesvm}')
print(f'R-squared: {r2svm}')

# %%
from sklearn.neural_network import MLPRegressor

# %%
modelmlp = MLPRegressor(hidden_layer_sizes =(200,200), max_iter=300, random_state = 42)
modelmlp.fit(XMFF_train, YMFF_train)

# %%
maescoresmlp = cross_val_score(modelmlp, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresmlp = cross_val_score(modelmlp, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresmlp = cross_val_score(modelmlp, XMFF_train, YMFF_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresmlp = np.sqrt(-rmsescoresmlp)
mae_scoresmlp = absolute(maescoresmlp)

# %%
print(f'MAE: {mae_scoresmlp.mean()},({mae_scoresmlp.std()})')
print(f'RMSE: {rmse_scoresmlp.mean()},({rmse_scoresmlp.std()})')
print(f'R-squared: {r2_scoresmlp.mean()}, ({r2_scoresmlp.std()})')

# %%
predictionsmlp = modelmlp.predict(XMFF_test)

# %%
msemlp = mean_squared_error(YMFF_test, predictionsmlp)
maemlp = mean_absolute_error(YMFF_test, predictionsmlp)
rmsemlp = np.sqrt(msemlp)
r2mlp = r2_score(YMFF_test, predictionsmlp)
print(f'MAE: {maemlp}')
print(f'RMSE: {rmsemlp}')
print(f'R-squared: {r2mlp}')

# %%
from sklearn.ensemble import BaggingRegressor

# %%
modelbagging = BaggingRegressor(n_estimators = 1000, oob_score = True, bootstrap = True, random_state = 42)
modelbagging.fit (XMFF_train, YMFF_train)

# %%
maescoresbagging = cross_val_score(modelbagging, XMFF_train, YMFF_train, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresbagging = cross_val_score(modelbagging, XMFF_train, YMFF_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresbagging = cross_val_score(modelbagging, XMFF_train, YMFF_train, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresbagging = np.sqrt(-rmsescoresbagging)
mae_scoresbagging = absolute(maescoresbagging)

# %%
print(f'MAE: {mae_scoresbagging.mean()},({mae_scoresbagging.std()})')
print(f'RMSE: {rmse_scoresbagging.mean()},({rmse_scoresbagging.std()})')
print(f'R-squared: {r2_scoresbagging.mean()}, ({r2_scoresbagging.std()})')

# %%
oob_predictionsbagging = modelbagging.oob_prediction_
oob_maebagging = mean_absolute_error(YMFF_train, oob_predictionsbagging)
oob_rmsebagging = np.sqrt(mean_squared_error(YMFF_train, oob_predictionsbagging))

# %%
print(f'OOB MAE: {oob_maebagging}')
print(f'OOB RMSE: {oob_rmsebagging}')
print(f'R-squared: {modelbagging.oob_score_}')

# %%
predictionsbagging = modelbagging.predict(XMFF_test)

# %%
msebagging = mean_squared_error(YMFF_test, predictionsbagging)
maebagging = mean_absolute_error(YMFF_test, predictionsbagging)
rmsebagging = np.sqrt(msebagging)
r2bagging = r2_score(YMFF_test, predictionsbagging)
print(f'MAE: {maebagging}')
print(f'RMSE: {rmsebagging}')
print(f'R-squared: {r2bagging}')

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
YMFF_testKNN = pd.to_numeric(YMFF_test, errors='coerce') 
YMFF_trainKNN = pd.to_numeric(YMFF_train, errors='coerce')

# %%
modelKNN = KNeighborsRegressor()
modelKNN.fit(XMFF_train, YMFF_trainKNN)

# %%
param_gridKNN = {'n_neighbors': [ 1,2,3,4,5,6,7]}

# %%
grid_searchKNNMAE = GridSearchCV(modelKNN, param_gridKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
grid_searchKNNMAE.fit(XMFF_train, YMFF_trainKNN)

# %%
n_neighbors_values = param_gridKNN['n_neighbors']
mean_test_scoresKNNMAE = (-grid_searchKNNMAE.cv_results_['mean_test_score'])
std_test_scoresKNNMAE = grid_searchKNNMAE.cv_results_['std_test_score']

# %%
grid_searchKNNRMSE = GridSearchCV(modelKNN, param_gridKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_searchKNNRMSE.fit(XMFF_train, YMFF_trainKNN)

# %%
mean_test_scoresKNNRMSE = np.sqrt(-grid_searchKNNRMSE.cv_results_['mean_test_score'])
std_test_scoresKNNRMSE = grid_searchKNNRMSE.cv_results_['std_test_score']

# %%
fig, ax1 = plt.subplots()
ax1.plot(n_neighbors_values, mean_test_scoresKNNMAE, marker='x', linestyle='--', color='black', label='MAE')
ax1.set_xlabel('n_neighbors')
ax1.set_ylabel('MAE', color='black')
ax1.tick_params('y', colors='black')
ax1.grid(True)
ax1.set_title('Grid Search Results for KNN')

ax2 = ax1.twinx()
ax2.plot(n_neighbors_values, mean_test_scoresKNNRMSE, marker='x', linestyle='--', color='blue', label='RMSE')
ax2.set_ylabel('RMSE', color='blue')
ax2.tick_params('y', colors='blue')
plt.show()

# %%
modelKNN5 = KNeighborsRegressor(n_neighbors=5)
modelKNN5.fit(XMFF_train, YMFF_trainKNN)

# %%
maescoresKNN5 = cross_val_score(modelKNN5, XMFF_train, YMFF_trainKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresKNN5 = cross_val_score(modelKNN5, XMFF_train, YMFF_trainKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresKNN5 = cross_val_score(modelKNN5, XMFF_train, YMFF_trainKNN, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresKNN5 = np.sqrt(-rmsescoresKNN5)
mae_scoresKNN5 = absolute(maescoresKNN5)

# %%
print(f'MAE: {mae_scoresKNN5.mean()},({mae_scoresKNN5.std()})')
print(f'RMSE: {rmse_scoresKNN5.mean()},({rmse_scoresKNN5.std()})')
print(f'R-squared: {r2_scoresKNN5.mean()}, ({r2_scoresKNN5.std()})')

# %%
predictionsKNN5 = modelKNN5.predict(XMFF_test)

# %%
mseKNN5 = mean_squared_error(YMFF_testKNN, predictionsKNN5)
maeKNN5 = mean_absolute_error(YMFF_testKNN, predictionsKNN5)
rmseKNN5 = np.sqrt(mseKNN5)
r2KNN5 = r2_score(YMFF_testKNN, predictionsKNN5)
print(f'MAE: { maeKNN5}')
print(f'RMSE: {rmseKNN5}')
print(f'R-squared: {r2KNN5}')

# %%
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

# %%
XMFF_trainarray = XMFF_train.values
XMFF_testarray = XMFF_test.values

# %%
mi_scores = mutual_info_regression(XMFF_trainarray, YMFF_trainKNN)

# %%
XMFF_trainarray[:, mi_scores > 0.0055].shape

# %%
maeKNN_values = []
rmseKNN_values = []
k_values = np.arange(0, 0.02, 0.0005)
for k in k_values:
    selected_features_train = XMFF_trainarray[:, mi_scores > k]
    selected_features_test = XMFF_testarray[:, mi_scores > k]
    modelKNNselected = KNeighborsRegressor(n_neighbors=5)
    modelKNNselected.fit(selected_features_train, YMFF_trainKNN)
    predictionsKNNselected = modelKNNselected.predict(selected_features_test)
    mseKNNselected = mean_squared_error(YMFF_testKNN, predictionsKNNselected)
    maeKNNselected = mean_absolute_error(YMFF_testKNN, predictionsKNNselected)
    rmseKNNselected = np.sqrt(mseKNNselected)
    r2KNNselected = r2_score(YMFF_testKNN, predictionsKNNselected)

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
modelKNNreduced.fit(XMFF_trainarray[:, mi_scores > 0.0065], YMFF_trainKNN)

# %%
maescoresKNNreduced = cross_val_score(modelKNNreduced, XMFF_trainarray[:, mi_scores > 0.0065], YMFF_trainKNN, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
rmsescoresKNNreduced = cross_val_score(modelKNNreduced, XMFF_trainarray[:, mi_scores > 0.0065], YMFF_trainKNN, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
r2_scoresKNNreduced = cross_val_score(modelKNNreduced, XMFF_trainarray[:, mi_scores > 0.0065], YMFF_trainKNN, scoring='r2', cv=5, n_jobs=-1)

# %%
rmse_scoresKNNreduced = np.sqrt(-rmsescoresKNNreduced)
mae_scoresKNNreduced = absolute(maescoresKNNreduced)

# %%
print(f'MAE: {mae_scoresKNNreduced.mean()},({mae_scoresKNNreduced.std()})')
print(f'RMSE: {rmse_scoresKNNreduced.mean()},({rmse_scoresKNNreduced.std()})')
print(f'R-squared: {r2_scoresKNNreduced.mean()}, ({r2_scoresKNNreduced.std()})')

# %%
predictionsKNNreduced = modelKNNreduced.predict(XMFF_testarray[:, mi_scores > 0.0065])

# %%
mseKNNreduced = mean_squared_error(YMFF_testKNN, predictionsKNNreduced)
maeKNNreduced = mean_absolute_error(YMFF_testKNN, predictionsKNNreduced)
rmseKNNreduced = np.sqrt(mseKNNreduced)
r2KNNreduced = r2_score(YMFF_testKNN, predictionsKNNreduced)
print(f'MAE: { maeKNNreduced}')
print(f'RMSE: {rmseKNNreduced}')
print(f'R-squared: {r2KNNreduced}')

# %%
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

# %%
YMFF_test_array = YMFF_test.astype(float)

# %%
correlation_coefficient, p_valuecc = pearsonr(predictionsKNNreduced, YMFF_test_array)

# %%
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_valuecc}")

# %%
absolute_errorsxgb = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsxgb)]
absolute_errorsrf = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsrf2)]
absolute_errorssvm = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionssvm)]
absolute_errorsmlp = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsmlp)]
absolute_errorsbagging = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsbagging)]
absolute_errorsKNN = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsKNN5)]
absolute_errorsKNNreduced = [abs(actual - predicted) for actual, predicted in zip(YMFF_test_array, predictionsKNNreduced)]

# %%
t_statistic, p_valuet = ttest_rel(absolute_errorsKNN, absolute_errorsKNNreduced)

# %%
print(f"T-statistic {t_statistic}")
print(f"P-value: {p_valuet}")

# %%



