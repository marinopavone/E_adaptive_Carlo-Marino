from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from chemical_brother.data_maker import DataMaker, ChemicalClass
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

#%%   Dataset building

data_maker = DataMaker("dataset/")
data_maker.set_contamination_classes(
    [
        ChemicalClass.SODIUM_HYDROXIDE,
        ChemicalClass.SODIUM_CHLORIDE,
        ChemicalClass.SODIUM_HYPOCHLORITE,
        ChemicalClass.POTASSIUM_NITRATE,
        ChemicalClass.CALCIUM_NITRATE,
        # ChemicalClass.POTABLE_WATER,
        # ChemicalClass.NELSEN,
        # ChemicalClass.PHOSPHORIC_ACID,
        # ChemicalClass.FORMIC_ACID,
        # ChemicalClass.HYDROCHLORIC_ACID,
        # ChemicalClass.HYDROGEN_PEROXIDE,
        # ChemicalClass.ACETONE,
        # ChemicalClass.ACETIC_ACID,
        # ChemicalClass.ETHANOL,
        # ChemicalClass.AMMONIA,
    ]
)
data = data_maker.make_full_dataset()
labels = data[["CLASS"]].to_numpy()
data = data.drop(columns=["CLASS"]).to_numpy()
#%%   encoding and normalization

scaler = MinMaxScaler()

label_encoder = LabelEncoder()
label_encoder.fit(labels)

scaler.fit(data)
train, test = data_maker.make_train_test_experiments(3, 500)
test_experiment_selection=[1,3,5]
train, test = data_maker.split_train_test_by_experiment(test_experiment_selection, 500)
x_train = train.drop(columns=["CLASS"]).to_numpy()
y_train = train[["CLASS"]].to_numpy()
x_test = test.drop(columns=["CLASS"]).to_numpy()
y_test = test[["CLASS"]].to_numpy()

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
#%%   Apply UMAP to reduce to 2D for visualization (or more for clustering)
n_neighbors=100
umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, n_jobs=-1)
X_umap = umap_reducer.fit_transform(x_train)



# Convert to DataFrame
df_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
df_umap["label"] = y_train  # Ground truth labels

#%% Plot the UMAP-reduced data with clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="UMAP1", y="UMAP2", hue="label", palette="tab10", data=df_umap, alpha=0.8)
plt.title("training")
plt.legend(title="Neighbors = " + str(n_neighbors))
plt.show()
#%% Plot the UMAP-reduced data with clusters

X_umap_test = umap_reducer.transform(x_test)
df_umap_test = pd.DataFrame(X_umap_test, columns=["UMAP1", "UMAP2"])
df_umap_test["label"] = y_test  # Ground truth labels

plt.figure(figsize=(10, 6))
sns.scatterplot(x="UMAP1", y="UMAP2", hue="label", palette="tab10", data=df_umap_test, alpha=0.8)
plt.title("test")
plt.legend(title="Neighbors = " + str(n_neighbors))
plt.show()