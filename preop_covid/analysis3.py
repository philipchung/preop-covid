#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from case_data import CaseData
from lab_data import LabData
from pandas.api.types import CategoricalDtype
from preop_data import PreopSDE
from vaccine_data import VaccineData

# Define data paths and directories
project_dir = Path("/Users/chungph/Developer/preop-covid")
data_dir = project_dir / "data"
data_version = 2
raw_dir = data_dir / f"v{int(2)}" / "raw"
processed_dir = data_dir / f"v{int(2)}" / "processed"
[path.mkdir(parents=True, exist_ok=True) for path in (raw_dir, processed_dir)]  # type: ignore
cohort_details_path = raw_dir / "Cohort Details_9602.csv"
summary_path = raw_dir / "Summary_9602.csv"
cases_path = raw_dir / "Main_Case_9602.csv"
labs_path = raw_dir / "SF_309 Covid Labs.csv"
preop_smartdataelements_path = raw_dir / "CF_2023-02-27_ChungCOVIDPreAnesEval.csv"
hm_path = raw_dir / "CF_2023-02-27_ChungCOVIDHealthMaintenance.csv"
#%%
# Load Metadata
raw_cohort_df = pd.read_csv(cohort_details_path)
raw_summary_df = pd.read_csv(summary_path)

# Load & Clean Labs Data (Multiple Labs per MPOG_Case_ID)
lab_data = LabData(labs_df=labs_path, data_version=data_version)
labs_df = lab_data()

# Load & Clean Cases Data (One Row per MPOG_Case_ID)
case_data = CaseData(cases_df=cases_path, data_version=data_version)
# Associate Labs from each Patient to Cases for each Patient
cases_df = case_data.associate_labs_to_cases(labs_df=labs_df)

# Get only patients with a positive Preop COVID test
cases_with_positive_preop_covid = cases_df[cases_df.HasPositivePreopCovidTest]

#%%
# Load & Clean Vaccine Data (Multiple Vaccines per MPOG_Case_ID & MPOG_Patient_ID)
vaccine_data = VaccineData(vaccines_df=hm_path, data_version=data_version)
flu_vaccines_df = vaccine_data.flu_vaccines_df
covid_vaccines_df = vaccine_data.covid_vaccines_df

covid_vaccines_df.VaccineKind.value_counts()
#%%
# Load & Clean SmartDataElements Data (Multiple SDE per MPOG_Case_ID)
preop_data = PreopSDE(preop_df=preop_smartdataelements_path, data_version=data_version)
problems_df = preop_data.problems_df

problems_df.loc[problems_df.IsPresent].Problem.value_counts()

#%% [markdown]
# ## Reformat Tables so we have 1 row per MPOG_Cases_ID
# This involves aggregation of multiple rows from certain tables
# into a single row prior to joining these tables together

# %%
# Get Number of Pre-op COVID vaccines for each Case
# (row for every unique MPOG_Case_ID & Vaccine_UUID combination)
c = cases_df.reset_index().loc[:, ["MPOG_Case_ID", "MPOG_Patient_ID", "AnesStart"]]
covid_vaccines = pd.merge(
    left=c,
    right=covid_vaccines_df.reset_index(),
    how="left",
    left_on="MPOG_Patient_ID",
    right_on="MPOG_Patient_ID",
).set_index("MPOG_Case_ID")
# Only keep rows where vaccine was administered prior to Case Start
covid_vaccines = covid_vaccines.loc[covid_vaccines.AnesStart > covid_vaccines.VaccineDate]
# Aggregate Multiple Pre-op Vaccines for each MPOG_Case_ID into a list
# so we have 1 row per MPOG_Case_ID
covid_vaccines = covid_vaccines.groupby("MPOG_Case_ID")["VaccineUUID", "VaccineDate"].agg(
    {"VaccineUUID": list, "VaccineDate": list}
)
covid_vaccines["NumPreopVaccines"] = covid_vaccines.VaccineUUID.apply(len)
print(f"Num Cases with Preop Vaccine Data: {covid_vaccines.shape[0]}")

# TODO: deduplicate num_preop vaccines?

# Pivot Table for ROS Problems to get ProcID x Problems
# If any SmartDataEelement is True for a Problem (across any PreAnes Note)
# written for a specific ProcID, then we mark the Problem as True.
cases_problems = pd.pivot_table(
    data=problems_df,
    index="MPOG_Case_ID",
    columns="Problem",
    values="IsPresent",
    aggfunc=lambda x: True if any(x) else False,
    fill_value=False,
)
print(f"Num Cases with ROS Problems: {cases_problems.shape[0]}")

# Pivot Table for Organ Systems- (ProcID x OrganSystem)
# If any SmartDataElement in an OrganSystem is marked True (across any PreAnes Note)
# written for a specific ProcID, then we mark the OrganSystem as True.
cases_organ_systems = pd.pivot_table(
    data=problems_df,
    index="MPOG_Case_ID",
    columns="OrganSystem",
    values="IsPresent",
    aggfunc=lambda x: True if any(x) else False,
    fill_value=False,
)
print(f"Num Cases with ROS Problems Marked by Organ Systems: {cases_organ_systems.shape[0]}")
#%%
# Join Vaccines, SDE Problems & Organ Systems Data to original cases table
# Note: original cases table has all the Elixhauser Comorbidities & Complications
# as well as time between last PCR+ test and case

# We drop cases where we don't have SDE data

num_preop_covid_vaccine_dtype = CategoricalDtype(
    categories=["0", "1", "2", "3", "4+"], ordered=True
)
had_preop_covid_vaccine_dtype = CategoricalDtype(categories=["Yes", "No"], ordered=True)

# Join Vaccines (MPOG_Case_ID without vaccines are unvaccinated)
df = cases_df.join(covid_vaccines, how="left")

df.VaccineUUID = [[] if x is np.NaN else x for x in df.VaccineUUID]
df.VaccineDate = [[] if x is np.NaN else x for x in df.VaccineDate]
df.NumPreopVaccines = df.NumPreopVaccines.fillna(0).astype(int)
df["NumPreopVaccinesCat"] = df.NumPreopVaccines.apply(lambda x: "4+" if x >= 4 else f"{x}").astype(
    num_preop_covid_vaccine_dtype
)
df["HadCovidVaccine"] = df.NumPreopVaccines.apply(lambda x: "Yes" if x > 0 else "No").astype(
    had_preop_covid_vaccine_dtype
)

# Join SDE Data
df = df.join(cases_organ_systems, how="inner").join(cases_problems, how="inner")
print(f"Num Cases: {df.shape[0]}")

# Preview this Table
df

#%%
# Covid Vaccine Columns
covid_vaccine_cols = covid_vaccines.columns.tolist()
covid_vaccine_cols
#%%
# All ROS Problems
ros_problem_cols = cases_problems.columns.tolist()
ros_problem_cols
#%%
# All ROS Organ Systems
ros_organ_systems_cols = cases_organ_systems.columns.tolist()
ros_organ_systems_cols
#%%
# Case Data (Elixhauser Comorbidiites, Complications, PCR Data)
case_cols = cases_df.columns.tolist()
case_cols
#%% [markdown]
# Now we can interrogate almost any patient population based on combination of ROS
# or COVID Vaccination Status and look at the MPOG documented complication
# or Elixhauser Comorbidities
# NOTE:
# - Any Cohort selection is based on ROS Data from Pre-op Note
# - Any Outcome Complications or Elixhauser Comorbidities is based on case billing data
# (e.g. ICD codes)

#%%
#%%
def make_count_percent_plot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    xlabel: str,
    title: str,
    figsize: tuple[int, int] = (6, 10),
) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # Count Plot
    sns.histplot(
        data=data,
        x=x,
        hue=hue,
        stat="count",
        multiple="dodge",
        ax=ax[0],
    )
    ax[0].set(title=f"{hue}, Case Counts", xlabel=xlabel)
    for container in ax[0].containers:
        ax[0].bar_label(container, label_type="edge", fmt="%g")
    # Percent Plot
    sns.histplot(
        data=data,
        x=x,
        hue=hue,
        stat="percent",
        multiple="fill",
        ax=ax[1],
    )
    ax[1].set(title=f"{hue}, Percentage of Cases", xlabel=xlabel)
    for container in ax[1].containers:
        ax[1].bar_label(container, label_type="center", fmt="%.2f")
    plt.suptitle(title)
    plt.tight_layout()
    return fig, ax


def make_count_percent_plots(
    data: pd.DataFrame,
    x: str,
    hue: str | list[str],
    xlabel: str,
    title: str,
    figsize: tuple[int, int] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    if isinstance(hue, str):
        figsize = (10, 6) if figsize is None else figsize
        return make_count_percent_plot(
            data=data, x=x, hue=hue, xlabel=xlabel, title=title, figsize=figsize
        )
    else:
        figsize = (10, 12) if figsize is None else figsize
        num_hues = len(hue)
        fig, ax = plt.subplots(nrows=num_hues, ncols=2, figsize=figsize)

        for idx, h in enumerate(hue):
            ct_ax = ax[idx, 0]
            pct_ax = ax[idx, 1]
            # Count Plot
            sns.histplot(
                data=data,
                x=x,
                hue=h,
                stat="count",
                multiple="dodge",
                ax=ct_ax,
            )
            ct_ax.set(title=f"{h}, Case Counts", xlabel=xlabel)
            for container in ct_ax.containers:
                ct_ax.bar_label(container, label_type="edge", fmt="%g")
            # Percent Plot
            sns.histplot(
                data=data,
                x=x,
                hue=h,
                stat="percent",
                multiple="fill",
                ax=pct_ax,
            )
            pct_ax.set(title=f"{h}, Percentage of Cases", xlabel=xlabel)
            for container in pct_ax.containers:
                pct_ax.bar_label(container, label_type="center", fmt="%.2f")
            plt.suptitle(title)
            plt.tight_layout()
    return fig, ax


# Cohort = Patients with COPD (in ROS)
copd = df.loc[df.COPD]
# Outcome = PulmonaryComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=copd,
    x="NumPreopVaccinesCat",
    hue="HadPulmonaryComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadPulmonaryComplication",
)
#%%
# Outcome = CardiacComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=copd,
    x="NumPreopVaccinesCat",
    hue="HadCardiacComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadCardiacComplication",
)
# Outcome = AKIComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=copd,
    x="NumPreopVaccinesCat",
    hue="HadAKIComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadAKIComplication",
)
#%%
# Cohort = Patients with COPD or Asthma or Bronchitis or URI (in ROS)
acute_or_chronic_pulm_dz = df.loc[df.COPD | df.ASTHMA | df.BRONCHITIS | df.URI]
# Outcome = PulmonaryComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=acute_or_chronic_pulm_dz,
    x="NumPreopVaccinesCat",
    hue="HadPulmonaryComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadPulmonaryComplication",
)
# Outcome = CardiacComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=acute_or_chronic_pulm_dz,
    x="NumPreopVaccinesCat",
    hue="HadCardiacComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadCardiacComplication",
)
# Outcome = AKIComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=acute_or_chronic_pulm_dz,
    x="NumPreopVaccinesCat",
    hue="HadAKIComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadAKIComplication",
)
#%%
# cohort = Patient with any +Respiratory ROS
any_respiratory = df.loc[df.RESPIRATORY]
# Outcome = PulmonaryComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=any_respiratory,
    x="NumPreopVaccinesCat",
    hue="HadPulmonaryComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadPulmonaryComplication",
)
# Outcome = CardiacComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=any_respiratory,
    x="NumPreopVaccinesCat",
    hue="HadCardiacComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadCardiacComplication",
)
# Outcome = AKIComplication (by case ICD code)
fig, ax = make_count_percent_plots(
    data=any_respiratory,
    x="NumPreopVaccinesCat",
    hue="HadAKIComplication",
    xlabel="Number of Preop Covid Vaccines",
    title="Patients with COPD who HadAKIComplication",
)
#%% [markdown]
# ## Dimensionality Reduction & Clustering of ROS Problems
#%%
# Cluster
from sklearn.cluster import DBSCAN

X_df = df.loc[:, ["NumPreopVaccines"] + ros_problem_cols]
X = X_df.to_numpy()
dbscan = DBSCAN(eps=0.3, metric="cosine", n_jobs=-1)
y_dbscan = dbscan.fit_predict(X)
# Save the prediction as a column
X_df["y_dbscan"] = y_dbscan
# Check the distribution
X_df["y_dbscan"].value_counts()
#%%
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#%%
# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

X_df = df.loc[:, ["NumPreopVaccines"] + ros_problem_cols]
X = X_df.to_numpy()
hc = AgglomerativeClustering(
    n_clusters=7,
    metric="euclidean",
    linkage="ward",
    memory=(Path(__file__).parent / "temp").as_posix(),
)
y_hc = hc.fit_predict(X)
# Save the cluster prediciton as a column
X_df["y_hc"] = y_hc
# Check the distribution
X_df["y_hc"].value_counts()

# TODO: Leiden Alg for KNN graph
#%%
# NMF for fuzzy clustering/"topic" discovery
from sklearn.decomposition import NMF

X_df = df.loc[:, ["NumPreopVaccines"] + ros_problem_cols]
X = X_df.to_numpy()
# L1 regularization for sparse features in each topic
n_components = 15
nmf = NMF(
    n_components=n_components, init=None, beta_loss="frobenius", l1_ratio=1.0, random_state=42
)
# Transformed Data = W; Component Matrix = H
W = nmf.fit_transform(X)
H = nmf.components_
# Each row in H is a Transformed Feature.  Each Col in H is a topic (original feature).
# So a Transformed feature is a weighted blend of topics/orginal features
H_df = pd.DataFrame(H, columns=X_df.columns)

# Get Top Topics for each feature
topic_features = H_df.apply(
    lambda row: row.sort_values(ascending=False).index.tolist()[:10], axis=1
)
topic_features.apply(lambda top_k: ", ".join(top_k)).to_dict()
# TODO: visualize cluster membership for topic
#%% [markdown]
# ## NMF Using Nimfa Library
#%%
# Plot Cophenetic Correlation Coefficient vs. NMF rank to determine the optimal
# Number of factors (soft "clusters") to explain data
# (this computation takes a very long time)
import time

import nimfa

start = time.time()

rank_range = range(2, 51)
n_run = 10
total = len(rank_range) * n_run

# Define Feature Matrix
X_df = df.loc[:, ros_problem_cols]
X = X_df.to_numpy()
# Initialize NMF model & Estimate Rank
nmf = nimfa.Nmf(
    V=X,
    seed="random",
    objective="fro",
    update="euclidean",
    max_iter=200,
)
estimate_rank_results = nmf.estimate_rank(rank_range=rank_range, n_run=n_run, what="all")
cophenetic = [estimate_rank_results[x]["cophenetic"] for x in rank_range]

print(f"Estimate Rank Took: {time.time() - start} seconds")

# Plot Cophenetic Correlation
p1 = sns.relplot(
    x=rank_range,
    y=cophenetic,
    kind="line",
)
p1.set(
    title="Cophenetic Correlation Coefficient vs. NMF Rank",
    xlabel="Number of Components",
    ylabel="Cophenetic Correlation Coefficient",
)

#%%
# Convert Dict to Dataframe
estimate_rank_df = pd.DataFrame.from_dict(data=estimate_rank_results, orient="index")
estimate_rank_df
#%%
# Save Dataframe as Pickle
nmf_est_rank_path = data_dir / "v2" / "processed" / "nmf_estimate_rank2.pickle"
estimate_rank_df.to_pickle(path=nmf_est_rank_path)
#%%
# Read Pickled Dataframe it back to memory and transform it back into a dict
estimate_rank_df = pd.read_pickle(nmf_est_rank_path)
nmf_rank_est_dict = estimate_rank_df.to_dict(orient="index")
nmf_rank_est_dict

#%%
# Use NIMFA for NMF
import nimfa

X_df = df.loc[:, ros_problem_cols]
X = X_df.to_numpy()
nmf = nimfa.Nmf(
    V=X,
    seed="nndsvd",
    rank=20,
    objective="fro",
    update="euclidean",
    max_iter=200,
)
nmf_fit = nmf()
# Transformed Data: Samples as weighted topics
W = nmf_fit.basis()
# Component Matrix: Topics as weighted blend of original features
H = nmf_fit.coef()

# Each row in H is a Topic.  Each Col in H is the original feature).
# Topics are a blend of the original features
topic_names = [f"Topic{k}" for k in range(H.shape[0])]
H_df = pd.DataFrame(
    data=H,
    columns=X_df.columns,
    index=topic_names,
)
# Get Top 5 features for each topic & the percent each feature contributes to the topic
# Normalize the component matrix by each topic
H_df_norm = H_df.apply(lambda row: row / row.sum(), axis=1)
# Get top 5 and their percentage weights
topic_features_norm = H_df_norm.apply(
    lambda row: [f"{k} ({v:.2%})" for k, v in row.nlargest(5).items()], axis=1
)
# Format as a dataframe
topic_features_norm_df = pd.DataFrame.from_dict(
    data=topic_features_norm.to_dict(),
    orient="index",
    columns=[f"TopFeature{k+1}" for k in range(5)],
)
# Create mapping between topic name and a topic "alias"/identity
topic_features_map_df = topic_features_norm_df.apply(lambda row: "\n".join(row), axis=1).rename(
    "TopicBlend"
)
topic_features_map = topic_features_map_df.to_dict()
topic_name2alias = H_df_norm.apply(
    lambda row: [f"{k}" for k in row.nlargest(1).keys()][0], axis=1
).to_dict()
topic_aliases = [topic_name2alias[n] for n in topic_names]

# Put Transformed Data Matrix into a Dataframe
W_df = pd.DataFrame(data=W, columns=topic_names, index=X_df.index)

#%%
# Visualize NMF Topics with UMAP
import umap

# Compute 2 Dimensional UMAP Embeddings for visualization (from original 77 Feature Dimensions)
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)
#%%
# Normalize Transformed Data Matrix by Column so Each Topic is scaled from 0-1
W_df_col_norm = (W_df - W_df.min()) / (W_df.max() - W_df.min())

# Make Embedding Dataframe
umap_data = (
    pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df.index)
    .join(df.NumPreopVaccinesCat)
    .join(W_df_col_norm)
)

# Melt Topics to Long Table Format
umap_data_long = pd.melt(
    umap_data,
    id_vars=["UMAP1", "UMAP2", "NumPreopVaccinesCat"],
    value_vars=topic_names,
    var_name="Topic",
    value_name="Value",
)
umap_data_long = umap_data_long.merge(
    topic_features_map_df, how="left", left_on="Topic", right_index=True
)
umap_data_long["Title"] = umap_data_long.apply(
    lambda row: f"{row.Topic}:\n{row.TopicBlend}", axis=1
)
#%%
# Visualize UMAP + Cluster Labels
fig, ax = plt.subplots(figsize=(10, 10))
# sns.scatterplot(data=data, x="UMAP1", y="UMAP2", hue="NumPreopVaccinesCat", edgecolor=None, s=2)
topic_num = 11
topic = f"Topic{topic_num}"
topic_data = umap_data[topic]
cmap = "Spectral"
p = sns.scatterplot(
    data=umap_data,
    x="UMAP1",
    y="UMAP2",
    hue=topic,
    palette=cmap,
    edgecolor=None,
    s=2,
)
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
p.get_legend().remove()
p.figure.colorbar(sm, format=lambda x, _: f"{x:.0%}")
plt.title(f"{topic}:\n{topic_features_map[topic]}", fontsize=12)

#%%
# UMAP Plot Facet Grid for NMF ROS, Highlighting Magnitude of Each Topic for Each Case
cmap = "viridis"
g = sns.relplot(
    kind="scatter",
    data=umap_data_long,
    x="UMAP1",
    y="UMAP2",
    hue="Value",
    col="Title",
    col_wrap=4,
    palette=cmap,
    edgecolor=None,
    s=2,
    legend=False,
)
cbar_ax = g.fig.add_axes([1.015, 0.25, 0.015, 0.5])
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
g.figure.colorbar(sm, cax=cbar_ax, format=lambda x, _: f"{x:.0%}")
g.set_titles(col_template="{col_name}")
#%%
p = sns.scatterplot(
    data=umap_data,
    x="UMAP1",
    y="UMAP2",
    hue="NumPreopVaccinesCat",
    palette=cmap,
    edgecolor=None,
    s=2,
)
p.set
#%% [markdown]
# ## Select Topic Subpopulations Of Interest for Complication Analysis
#%%
# TODO:
# - Mask/Threshold Topics:
#   - 0: HLD
#   - 2: GERD
#   - 3: CHRONIC RENAL DISEASE
#   - 8: SLEEP APNEA
#   - 9: HYPERTENSION
#   - 11: DIABETES
#   - 12: ASTHMA
#   - 15: VALVULAR PROBLEMS/MURMURS (& Cardiac)
#   - 18: LIVER
#   - 19: PONV
# - stratify population by:
#   - covid vaccine (yes/no)
#   - flu vaccine (yes/no)
# - examine complications:
#   - pulmonary
#   - cardiac
#   - renal
#%%
# Normalize Transformed Data Matrix by Row
# Interpretation: % of each Topic that makes up each case's ROS
W_df_norm = W_df.apply(lambda row: row / row.sum(), axis=1)
# Apply Threshold Cut-off to get a "soft" cluster
# Note: clusters may overlap & a single case may belong to multiple clusters
threshold = 0.10  # Cluster membership if >10% of Case's ROS is explained by topic
mask = W_df_norm.applymap(lambda x: x > threshold)

# Percent of examples in each topic cluster (clusters may overlap)
percent_topic_clusters = (mask.sum() / mask.shape[0]).rename("DataFraction")
percent_topic_clusters = pd.DataFrame.from_dict(
    data=topic_name2alias, orient="index", columns=["TopFeature"]
).join(percent_topic_clusters)
percent_topic_clusters
#%%
# Plot Complications For Each Topic Cluster of Interest
topic = "Topic9"
hld_case_ids = W_df.loc[mask[topic]].index
hld_df = df.loc[hld_case_ids]

complications = ["HadPulmonaryComplication", "HadCardiacComplication", "HadAKIComplication"]
fig, ax = make_count_percent_plots(
    data=hld_df,
    x="HadCovidVaccine",
    hue=complications,
    title=("Complications for " f"{topic}:\n{topic_features_map[topic]}"),
    xlabel="HadCovidVaccine",
    figsize=(10, 18),
)
#%%
# TODO:
# 1. pick out top diagnoses (CV, pulm, renal) from ROS/problems_df that we want to examine
# 2. join ROS/problems into cases_df table (alongside Elixhauser Comorbidities)
#    so each case has associated ROS (binary values)
# 3. for each case, get total # of vaccines/boosters; get # of vaccines/booster & duration
#    since last vaccine/booster administration.
# 4. For patients with vaccine in past 6 months, compare outcomes (PACU duration, Hospital LOS,
#    Mortality)
# 5. repeat for patients with vaccine in past 12 months...
# (NEJM paper suggests clinical protection for 6 months: https://www.nejm.org/doi/full/10.1056/NEJMoa2118691)
# 6. Patients with Heart Disease
# %%
# Call w/ Dustin Notes
# - self antigen tests
# - walgreens?  most pre-op is being done at local provider
# - free-text comment
# (maybe just secondary analysis)

# limit to possible covid?  home test?
# look just at vaccination status.  Difference in all-comers.

# - is there a difference risk between 2-3 post-op complications between vaccinated & non-vaccinated

# ROS groups:
# - COPD (assoc. w/ COVID)
# - CHF (interaction w/ COVID)
# - h/o MI
# - stroke (inc. risk w/ stroke w/ COVID)
