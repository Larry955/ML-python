from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mglearn

cancer = load_breast_cancer()

# scale the cancer data
scaler = StandardScaler()
scaler.fit(cancer.data)
cancer_scaled = scaler.transform(cancer.data)

# keep the first two principal components of the data
pca = PCA(n_components = 2)
# fit the PCA model to breast cancer data
pca.fit(cancer_scaled)

# transform data onto the first two principal components
cancer_pca = pca.transform(cancer_scaled)
print("Original Data: {}".format(str(cancer_scaled.shape)))
print("Reduced Data: {}".format(str(cancer_pca.shape)))

# plot first vs. second principal component, colored by class
plt.figure(figsize = (8,8))
mglearn.discrete_scatter(cancer_pca[:, 0], cancer_pca[:, 1], cancer.target)
#plt.scatter(cancer_pca[:, 0], cancer_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc = "best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components: {}".format(pca.components_))
plt.yticks([0, 1], ["First Component", "Second Component"])
plt.matshow(pca.components_)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation = 60, ha = "left")
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()



