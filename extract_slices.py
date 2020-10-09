import pandas as pd  

#Initialise the various rows to extract: 
def extract_df (chi_1, chi_2, chi_3, dataframe, configuration):
    # available configs are :
    # 1) all_constant_1 (all runs where chi_1 = chi_2 = chi_3 = 1value) : pass in chi_1
    # 2) all_constant_2 (all runs where chi_1 = chi_2 = chi_3 = 2 values): pass in chi_1 & 2
    # 3) 1g_eq2 (chi_i > chi_j = chi_k), pass in chi_1 & 2 while setting chi_3 to zero
    # 4) 2g_eq1 (chi_i = chi_j > chi_k), pass in chi_1 & 2 while setting chi_3 to zero

    A_RAW = []
    B_RAW = []
    chi_AB = []
    chi_AC = []
    chi_BC = []
    manual_cluster = []
    img_label = []

    if configuration == "all_constant_1":
        for index, row in dataframe.iterrows():
            if chi_1 == row["chi_AB"] == row["chi_AC"] == row["chi_BC"]:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])
    
    elif configuration == "all_constant_2":
        for index, row in dataframe.iterrows():
            if chi_1 == row["chi_AB"] == row["chi_AC"] == row["chi_BC"]:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])
        
            if chi_2 == row["chi_AB"] == row["chi_AC"] == row["chi_BC"]:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])

    elif configuration == "1g_eq2":
         for index, row in dataframe.iterrows():
            if  row["chi_AB"] == chi_1 and row["chi_AC"] == row["chi_BC"] == chi_2:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])
        
            if row["chi_AC"] == chi_1 and  row["chi_AB"] == row["chi_BC"] == chi_2:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])

            if row["chi_BC"] == chi_1 and row["chi_AC"] == row["chi_AB"] == chi_2:
                A_RAW.append(row["A_RAW"])
                B_RAW.append(row["B_RAW"])
                chi_AB.append(row["chi_AB"])
                chi_AC.append(row["chi_AC"])
                chi_BC.append(row["chi_BC"])
                manual_cluster.append(row["direct_manualcluster"])
                img_label.append(row["labels"])

    df_dict = {
        "A_RAW":A_RAW,
        "B_RAW":B_RAW,
        "chi_AB":chi_AB,
        "chi_AC":chi_AC,
        "chi_BC":chi_BC,
        "cluster_labels":manual_cluster,
        "img_labels":img_label
    }
    df = pd.DataFrame(df_dict)

    return df
    