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
dftest = dftest.query('basicOrAcidic == "basic"')

# %%
dftest = dftest.query('pkafloat > 0 and pkafloat < 14')

# %%
dftest = dftest.reset_index(drop=True)

# %%
dftest

# %%
ROMolstest = dftest['ROMol']

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
df_alldescrs.insert(0,'SMILES', dftest['SMILES'])

# %%
df_alldescrs

# %%
df_alldescrs2 = df_alldescrs.drop('Ipc', axis=1)
df_alldescrs2 = df_alldescrs2.dropna()

# %%
df_alldescrs2 = df_alldescrs2.reset_index(drop=True)

# %%
pip install -U scikit-learn as sklearn

# %%
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# %%
Xdesc, Ydesc = df_alldescrs2.drop(["pKa", "SMILES"], axis=1), pd.DataFrame(df_alldescrs2[["pKa"]])

# %%
Xdesc2, Ydesc2 = df_alldescrs2.drop(["pKa", "SMILES"], axis=1), pd.DataFrame(df_alldescrs2[["pKa","SMILES"]])

# %%
Xdesc_train, Xdesc_test, Ydesc_train, Ydesc_test = train_test_split(Xdesc, Ydesc2, test_size=0.25, random_state=1)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from numpy import absolute

# %%
scaler = StandardScaler()
Xdesc_train_scaled = scaler.fit_transform(Xdesc_train)
Xdesc_test_scaled = scaler.transform(Xdesc_test)

# %%
modelmlp = MLPRegressor(hidden_layer_sizes =(200, 200), max_iter=300, random_state = 42)
modelmlp.fit(Xdesc_train_scaled, Ydesc_train["pKa"])

# %%
predictionsmlp = modelmlp.predict(Xdesc_test_scaled)

# %%
msemlp = mean_squared_error(Ydesc_test["pKa"], predictionsmlp)
maemlp = mean_absolute_error(Ydesc_test["pKa"], predictionsmlp)
rmsemlp = np.sqrt(msemlp)
r2mlp = r2_score(Ydesc_test["pKa"], predictionsmlp)
print(f'MAE: {maemlp}')
print(f'RMSE: {rmsemlp}')
print(f'R-squared: {r2mlp}')

# %%
df_predictionsmlp = pd.DataFrame(predictionsmlp)

# %%
Ydesc_test.insert(0, 'predictions', predictionsmlp )

# %%
dfcomp = Ydesc_test

# %%
dfcomp

# %%
sample = dfcomp.sample(n=100)

# %%
sample

# %%
sample.to_csv('MLP2.csv', index=True) 


