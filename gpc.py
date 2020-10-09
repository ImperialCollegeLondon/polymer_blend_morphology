import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from extract_slices import extract_df

import matplotlib.pyplot as plt


#Read in csv files:
full_df = pd.read_csv("./Clustering_Results.csv")

# Generate dataframe
df = extract_df(chi_1=0.003, chi_2=0.006, chi_3 = 0,  dataframe = full_df, configuration="1g_eq2")

# Perform data augmentation
mask = ["A_RAW", "B_RAW"]

features = np.array(df[mask])
features_plus0005 = np.add(features, 0.005)
features_plus0002 = np.add(features, 0.002)
features_minus0005 = np.add(features, -0.005)
features_minus0002 = np.add(features, -0.002)

features_1 = np.append(features, features_plus0005, axis=0)
features_2 = np.append(features_1, features_minus0005, axis=0)
features_3 = np.append(features_2, features_plus0002, axis=0)
features_4 = np.append(features_3, features_minus0002, axis=0)

# Extract targets
targets = list(df["cluster_labels"])
targets_1 = targets + targets + targets + targets + targets

# Set up GPC
classifier = GaussianProcessClassifier(1.0 * RBF(1.0))

X_train, X_test, y_train, y_test = train_test_split(features_4, targets_1, test_size=0.20, random_state=42)
classifier.fit(X_train, y_train)


#Set-up plotting for maps
def mol_frac_gen(a_start, a_stop, b_start, b_stop, a_number, b_number):
    A_raw = np.linspace(a_start, a_stop, a_number)

    B_raw = np.linspace(b_start, b_stop, b_number)

    a_list = []
    b_list = []
    for a_i in A_raw:
        for b_i in B_raw:
            c_i = 1.0 - a_i - b_i
            if (a_i + b_i + c_i) == 1.0:
                a_list.append(a_i)
                b_list.append(b_i)
    
    mol_frac_list = np.column_stack((a_list, b_list))

    return mol_frac_list , a_list, b_list

plotlist, a_frac, b_frac = mol_frac_gen(a_start=0.1, a_stop=0.8, b_start=0.1, b_stop=0.45, a_number=400, b_number=400)

expected = y_test # correct ans
predicted = classifier.predict(X_test)
plotted = classifier.predict(plotlist)

prediction_df_dict = {
    "predict":plotted,
    "A_RAW":a_frac,
    "B_RAW":b_frac
}

prediction_df = pd.DataFrame(prediction_df_dict)


color_list = [
    "lightblue",
    "lightcoral",
    "navajowhite",
    "plum",
    "lightpink",
    "violet",
    "lightgreen",
    "mediumseagreen",
    "lightyellow",
    "mediumpurple"
    
]


test_color_list = [
    "blue",
    "red",
    "tan",
    "indigo",
    "deeppink"
]


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(1,1,1)

for cluster, col in zip(np.unique(plotted), color_list):
    # We need to extract the parts of the DF that correspond to each cluster
    # clusters = []
    a_raw = []
    b_raw = []
    for index, row in prediction_df.iterrows():
        if row["predict"] == cluster:
            # clusters.append(row["clusters"])
            a_raw.append(row["A_RAW"])
            b_raw.append(row["B_RAW"])

    ax1.scatter(a_raw, b_raw, c=col, label=str(cluster), alpha = 0.3)


#Set up dataframe for test data: 
test_df = pd.DataFrame(X_test, columns=["a","b"])
test_df["clusters"] = predicted


for cluster, col in zip(np.unique(plotted), test_color_list):
    a_test = []
    b_test = []
    for index, row in test_df.iterrows():
        if row["clusters"] == cluster:
            a_test.append(row["a"])
            b_test.append(row["b"])

    ax1.scatter(a_test, b_test, c=col, alpha=1.0)



ax1.set_xlabel(r"$a_0$")
ax1.set_ylabel(r"$b_0$")
ax1.set_xlim(0.1,0.8)
ax1.set_ylim(0.1,0.45)
ax1.legend(loc=1, bbox_to_anchor=(1.0, 1.0), framealpha=1.0)
plt.show()

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s\n" % metrics.confusion_matrix(expected, predicted))

print("Comparing expected and predicted:", np.column_stack((expected, predicted)))