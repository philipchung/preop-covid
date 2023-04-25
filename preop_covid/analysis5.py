# %%
from pathlib import Path

import matplotlib.pyplot as plt
import nimfa
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import umap
from pandas.api.types import CategoricalDtype
from utils import xie_beni_index

from preop_covid.case_data import CaseData
from preop_covid.lab_data import LabData
from preop_covid.preop_data import PreopSDE
from preop_covid.utils import forestplot
from preop_covid.vaccine_data import VaccineData

# Display only 2 significant figures for decimals & p-value
# Will switch to using scientific notation if more sig figs
pd.options.display.float_format = "{:.2g}".format

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
# %%
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

# %%
# Load & Clean Vaccine Data (Multiple Vaccines per MPOG_Case_ID & MPOG_Patient_ID)
vaccine_data = VaccineData(vaccines_df=hm_path, data_version=data_version)
flu_vaccines_df = vaccine_data.flu_vaccines_df
covid_vaccines_df = vaccine_data.covid_vaccines_df

covid_vaccines_df.VaccineKind.value_counts()
# %%
# Load & Clean SmartDataElements Data (Multiple SDE per MPOG_Case_ID)
preop_data = PreopSDE(preop_df=preop_smartdataelements_path, data_version=data_version)
problems_df = preop_data.problems_df

problems_df.loc[problems_df.IsPresent].Problem.value_counts()

# %% [markdown]
# ### Reformat Tables so we have 1 row per MPOG_Cases_ID
# This involves aggregation of multiple rows from certain tables like vaccines
# where it is possible to have multiple vaccines prior to a case.

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
covid_vaccines = covid_vaccines.groupby("MPOG_Case_ID")[["VaccineUUID", "VaccineDate"]].agg(
    {"VaccineUUID": list, "VaccineDate": list}
)
covid_vaccines["NumPreopVaccines"] = covid_vaccines.VaccineUUID.apply(len)
print(f"Num Cases with Preop Vaccine Data: {covid_vaccines.shape[0]}")

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

# Join Vaccines, SDE Problems & Organ Systems Data to original cases table
# Note: original cases table has all the Elixhauser Comorbidities & Complications
# as well as time between last PCR+ test and case

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

# Join SDE Data, dropping cases where we don't have ROS Data from SDE
df = df.join(cases_organ_systems, how="inner").join(cases_problems, how="inner")
print(f"Num Cases: {df.shape[0]}")

# Preview this Table
df

# %%
# Covid Vaccine Columns
covid_vaccine_cols = covid_vaccines.columns.tolist()
# All ROS Problems
ros_problem_cols = cases_problems.columns.tolist()
# All ROS Organ Systems
ros_organ_systems_cols = cases_organ_systems.columns.tolist()
# Case Data (Elixhauser Comorbidiites, Complications, PCR Data)
case_cols = cases_df.columns.tolist()


# Format Capitalization
def format_capitalization(text: str) -> str:
    if text in (
        "CAD",
        "CHF",
        "COPD",
        "CVA",
        "DVT",
        "GERD",
        "HIV/AIDS",
        "PONV",
        "PUD",
        "PVD",
        "TIA",
        "URI",
    ):
        # Return Capitalized Abbreviation Unchanged
        return text
    elif text == "PAST MI":
        return "Past MI"
    elif text == "PRIOR IUFD":
        return "Prior IUFD"
    else:
        # Capitalize Each Word
        return " ".join(x.capitalize() for x in text.split(" "))


ros_problem_cols2 = [format_capitalization(x) for x in ros_problem_cols]
ros_organ_systems_cols2 = [format_capitalization(x) for x in ros_organ_systems_cols]
# %% [markdown]
# ## Dimensionality Reduction & Clustering of ROS Problems
#
# There are 77 distinct ROS Problem +/-.  (Note: any comment is treated as "+" for that problem)
# There are an additional 18 Organ Systems which can be commented upon (these are the
# comment vs. negative checkbox).  For these, Any comment is treated as "+" and is only
# negative if "-" is checked.
#
# We can compose any combination of these ROS elements to get a specific subset cohort
# to look at complications.  This is a very large number of subset cohorts to investigate.
# We can empirically select these cohorts using clinical knowledge (above), or we can
# use machine learning techniques to "discover" natural clusters.
#
# We Use Non-Negative Matrix Factorization (NMF) as a Topic Modeling/"Soft" Clustering
# technique as well as a Dimensionality Reduction technique.  This approach allows us
# to find "Topics", which are a blend of ROS features.  These Topics represent
# prototypical clinical phenotypes that we will discover using machine learning.
# Each case's ROS gets transformed into blend of "Topics" with each Topic
# being a blend of ROS features.  Thus each case can be composed of multiple Topics.
# Since each case does not exclusively belong to a Topic, this can be considered
# a "Soft"/fuzzy clustering approach.
# %% [markdown]
# ### Determining the Optimal Number of NMF Topics ("Soft" Clusters)
#
# NOTE: in this analysis, we will just use the 77 ROS problems and not the
# 18 Organ Systems to generate Topic Clusters.
#
# NMF requires us to specify the number of Topics we want, which must be empirically determined.
# We sweep this hyperparameter from 2-50 Topics and then we will compute the
# Cophenetic Correlation Coefficient (higher value indicates better stability of clusters).
#
# This computation takes a very long time (hours) since we need to compute the NMF many times
# to assess the statistical stability.
# %%
rank_range = range(2, 51)
n_run = 10
total = len(rank_range) * n_run
# %%
# # Compute NMF for each Topic Rank for n_run (this takes a long time) to yield
# Cophenetic Correlation Coefficients
# start = time.time()

# # Define Feature Matrix
# X_df = df.loc[:, ros_problem_cols]
# X = X_df.to_numpy()
# # Initialize NMF model & Estimate Rank
# nmf = nimfa.Nmf(
#     V=X,
#     seed="random",
#     objective="fro",
#     update="euclidean",
#     max_iter=200,
# )
# estimate_rank_results = nmf.estimate_rank(rank_range=rank_range, n_run=n_run, what="all")
# cophenetic = [estimate_rank_results[x]["cophenetic"] for x in rank_range]

# print(f"Estimate Rank Took: {time.time() - start} seconds")

# # Convert Dict to Dataframe
# estimate_rank_df = pd.DataFrame.from_dict(data=estimate_rank_results, orient="index")
# estimate_rank_df
# #%%
# # Save Dataframe as Pickle
# nmf_est_rank_path = data_dir / "v2" / "processed" / "nmf_estimate_rank2.pickle"
# estimate_rank_df.to_pickle(path=nmf_est_rank_path)
# %%
# Read Pickled Dataframe it back to memory and transform it back into a dict
nmf_est_rank_path = data_dir / "v2" / "processed" / "nmf_estimate_rank2.pickle"
estimate_rank_df = pd.read_pickle(nmf_est_rank_path)
estimate_rank_results = estimate_rank_df.to_dict(orient="index")
estimate_rank_results
cophenetic = [estimate_rank_results[x]["cophenetic"] for x in rank_range]

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

# %% [markdown]
# ### Compute NMF for a specific Number of Topics
#
# Based on the cophenetic plot, we choose rank=20 topics because this is the lowest
# rank where we can get a high Cophenetic Correlation Coefficient.  Choosing fewer
# topics will still work, but may result in less table topic clusters.
# %%
# Compute NMF for 20 Topics
X_df = df.loc[:, ros_problem_cols].rename(columns=dict(zip(ros_problem_cols, ros_problem_cols2)))
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
# Put Transformed Data Matrix W into a Dataframe
W_df = pd.DataFrame(data=W, columns=topic_names, index=X_df.index)

# %%
# Get Top 5 features for each topic & the percent each feature contributes to the topic
# Normalize the component matrix by each topic
H_df_norm = H_df.apply(lambda row: row / row.sum(), axis=1)

# Categorical Dtype for Topics
topic_dtype = CategoricalDtype(
    categories=H_df_norm.index.tolist(),
    ordered=True,
)
# Categorical Dtype for Problems
ros_problems_dtype = CategoricalDtype(
    categories=H_df_norm.columns.tolist(),
    ordered=False,
)

H_df_norm.index = pd.CategoricalIndex(data=H_df_norm.index, dtype=topic_dtype)
H_df_norm.columns = pd.CategoricalIndex(data=H_df_norm.columns, dtype=ros_problems_dtype)
# %%


def get_topic_blend(row: pd.Series, top_n: int | None = 5, threshhold: float | None = 0.01) -> dict:
    """Determine blend of features in each topic by using ranking and weight thresholds.

    Args:
        row (pd.Series): row of features which represents the topic.
        top_n (int, optional): how many of the top weighted features to consider.  If None,
            will consider all features.
        threshhold (float, optional): only consider features that are higher than
            the fractional threshold.

    Returns:
        dict: Dict with keys of features that contribute to topic and values of weight for
            each feature.
    """
    if top_n is None:
        top_features_dict = row.sort_values(ascending=False)
    else:
        top_features_dict = row.nlargest(top_n)
    if threshhold is None:
        return top_features_dict.to_dict()
    else:
        return {k: v for k, v in top_features_dict.to_dict().items() if v >= threshhold}


# Topic Blends as Dicts
topic_features = H_df_norm.apply(
    lambda row: get_topic_blend(row, top_n=5, threshhold=0.05), axis=1
).rename("TopicFeatures")
topic_top5_features = H_df_norm.apply(
    lambda row: get_topic_blend(row, top_n=5, threshhold=None), axis=1
).rename("TopicFeatures")
# Format Topic Blends as List of Str ["Topic (%)", ...]
topic_features_lst = topic_features.apply(
    lambda d: [f"{k} ({v:.0%})" for k, v in d.items()]
).rename("TopicBlendList")
topic_top5_features_lst = topic_top5_features.apply(
    lambda d: [f"{k} ({v:.0%})" for k, v in d.items()]
).rename("TopicBlendList")
# Format Topic Blends as Single Str Label
topic_features_str = topic_features_lst.apply(lambda lst: " + ".join(lst)).rename("TopicBlend")
topic_features_str2 = topic_features_lst.apply(lambda lst: "\n".join(lst)).rename("TopicBlend")
topic_top5_features_str = topic_top5_features_lst.apply(lambda lst: " + ".join(lst)).rename(
    "TopicBlend"
)
topic_top5_features_str2 = topic_top5_features_lst.apply(lambda lst: "\n".join(lst)).rename(
    "TopicBlend"
)
# %%
# Display Top 5 Features for each Topic
topic_features_norm_df = pd.DataFrame.from_dict(
    data=topic_top5_features_lst.to_dict(),
    orient="index",
    columns=[f"TopFeature{k+1}" for k in range(5)],
)
topic_features_norm_df
# %%
# Display Topic Blends (Top 5 Features & Threshold > 3% Weight for Feature)
topic_features_str.to_dict()
# %%
topic_name2alias = H_df_norm.apply(
    lambda row: [f"{k}" for k in row.nlargest(1).keys()][0], axis=1
).to_dict()
topic_aliases = [topic_name2alias[n] for n in topic_names]

# %% [markdown]
# ### UMAP Visualizations
#
# We can use UMAP to visualize the NMF Topics.  UMAP is a technique to "project" the original
# 77 ROS problem dimensions to a 2D plot.  It is impossible to preserve all the information
# fidelity in this remapping, but helps create a map upon which we can visualize how
# our 20 Topics are distributed across our data.
# %%
# Compute 2 Dimensional UMAP Embeddings for visualization (from original 77 Feature Dimensions)
seed = 42
reducer = umap.UMAP(random_state=seed)
embedding = reducer.fit_transform(X)
# %%
# Normalize Transformed Data Matrix by Column so Each Topic is scaled from 0-1
W_df_col_norm = (W_df - W_df.min()) / (W_df.max() - W_df.min())

# Make Embedding Dataframe
umap_data = (
    pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"], index=df.index)
    .join(df.NumPreopVaccinesCat)
    .join(df.HadCovidVaccine)
    .join(W_df_col_norm)
)

# Melt Topics to Long Table Format
umap_data_long = pd.melt(
    umap_data,
    id_vars=["UMAP1", "UMAP2", "NumPreopVaccinesCat", "HadCovidVaccine"],
    value_vars=topic_names,
    var_name="Topic",
    value_name="Value",
)
umap_data_long = umap_data_long.merge(
    topic_features_str, how="left", left_on="Topic", right_index=True
)
umap_data_long["Title"] = umap_data_long.apply(
    lambda row: f"{row.Topic}:\n{row.TopicBlend}", axis=1
)

# %%
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
scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
g.figure.colorbar(scalar_mappable, cax=cbar_ax, format=lambda x, _: f"{x:.0%}")
g.set_titles(col_template="{col_name}")

# NOTE: you can see how our Topics are grouped within certain topics.  Because
# Topics are soft-clusters, we can see for some Topics (e.g. Topic 9) where there
# are distinct subpopulations of patients who are perfect matches (yellow color)
# for the Topic phenotype and others who are only partial matches (green color)
# NOTE: superposition of different Topics on the same region of UMAP indicate
# populations where both Topics are present.

# %% [markdown]
# ### Select Topic Subpopulations Of Interest for Complication Analysis
#
# We need to transform fuzzy/"soft" clusters to hard clusters that we can then use as a
# cohort to look at complication rates.
# * Each case is a blend of Topics--we can normalize the contribution of Topics to a case
# such that all the 20 topics all add up to 100%.  Then we choose a threshold.
# * **For example, if we choose a threshold=25%, This means that a clinical phenotype is
# determined to be present for a patient case if >25% of the Cases' ROS is explained by
# that clinical phenotype.**
# * This threshold converts the soft/fuzzy clusters into a discrete subpopulation of patients
# associated with the clinical phenotype.
# * This threshold allows for multiple clinical phenotypes to be
# present, but also requires a large portion of the patient’s ROS to be explained by
# the phenotype.
# * To validate that we have chosen an appropriate threshold, we can use the Xie-Beni
# index for internal cluster validation.  This is an index that measures cluster compactness
# and separation. Low values of Xie-Beni means that clusters are more compact and separate
# from one another.
#
# Note that it is possible for:
# * a Case to be in multiple Topic clusters.
# * a Case to not be in any Topic clusters.
#
# The goal is not to force all the cases into the Topics, which is what many other
# clustering algorithms do, but rather yield cases that are representative of the Topics
# within our dataset that we can then use as a cohort for further analysis.

# %%
# Normalize Transformed Data Matrix by Row
# Interpretation: % of each Topic that makes up each case's ROS
W_df_norm = W_df.apply(lambda row: row / row.sum(), axis=1)

# %%
# Determine appropriate threshold using Xie-Beni Index

thresholds = np.arange(start=0, stop=1.05, step=0.05)
thresholds[0], thresholds[-1] = 0.01, 0.99
xb_indices = []
for threshold in thresholds:
    # Apply threshold.  This yields a mask which is also the cluster assignments for each sample.
    mask = W_df_norm.applymap(lambda x: x > threshold)
    # Compute Xie-Beni Index
    xb_index = xie_beni_index(X=X, labels=mask)
    xb_indices += [xb_index]

xb = pd.DataFrame(data=xb_indices, index=thresholds, columns=["XieBeni"]).rename_axis(
    index="Threshholds"
)

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.lineplot(data=xb.reset_index(), x="Threshholds", y="XieBeni", ax=ax)
ax.set(
    title="Xie-Beni Index across Multiple Thresholds",
    ylabel="Xie-Beni Index",
    xlabel="Threshold Percent of ROS Explained by Clinical Phenotype",
)

# NOTE: based on this analysis, we can see that there is an "elbow" at around 0.2.
# Increasing the threshold much further than this would not yield significantly better
# separated clusters.

# %%
# Apply Threshold Cut-off to get clusters that we will use for downstream analysis
threshold = 0.25
mask = W_df_norm.applymap(lambda x: x > threshold)
# Percent of examples in each topic cluster (clusters may overlap)
num_case = mask.sum().rename("NumCases")
percent_case = (mask.sum() / mask.shape[0]).rename("FractionOfCases")
pd.DataFrame.from_dict(data=topic_name2alias, orient="index", columns=["TopFeature"]).join(
    topic_features_str
).join(num_case).join(percent_case)
# %%
# How many examples are not in any of the topics?
print("Number of examples belonging to no topics: ", mask.any(axis=1).eq(False).astype(int).sum())
print("Number of examples belonging to at least 1 topic: ", mask.any(axis=1).sum())

# %% [markdown]
# ## Determine Overlap Between Clusters
#
# Clusters generated using our thresholding procedure on topics are not mutually exclusive.
# This is purposeful.
#
# Exclusive clusters would force each patient case into only a single
# clinical phenotype. For example, if Hypertension and Asthma are two different
# clnical phenotypes and a patient has both, a traditional clustering algorithm like
# K-means would force assignment of that patient to either Hypertension cluster or
# Asthma cluster, but not both.
#
# Our approach is more realistic in that a patient may belong to multiple clinical
# phenotype clusters.  However, we should understand what degree of overlap exists.
# If the overlaps are significant, we may need to use Post Hoc techniques to reduce
# false discovery rate for multiple hypothesis testing as we are essentially testing
# the same population on different hypothesis.
#
# Jaccard Index = Set Intersection / Set Union
#
# We compute jaccard index between every cluster to see how much overlap.
# %%
# Compute the amount of overlap between clusters using Jaccard Index
topics = mask.columns
num_topics = len(topics)
topic_sets = {}
for topic in topics:
    topic_col = mask[topic]
    # Get ProcIDs belonging to topic cluster
    topic_ids = mask.index[topic_col].tolist()
    topic_sets[topic] = topic_ids


def jaccard(list1: list, list2: list) -> float:
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


jaccard_indices = np.zeros(shape=(num_topics, num_topics))
for i, topic_ids_group1 in enumerate(topic_sets.values()):
    for j, topic_ids_group2 in enumerate(topic_sets.values()):
        jaccard_indices[i, j] = jaccard(topic_ids_group1, topic_ids_group2)
jaccard_indices_df = pd.DataFrame(data=jaccard_indices, index=topics, columns=topics)
jaccard_indices_df
# %%
# Lets get the minimum and maximum Jaccard Similarity score between any 2 different groups
temp = jaccard_indices_df.replace(1, np.NaN)
print("Minimum Jaccard Similarity: ", temp.min().min())
print("Maximum Jaccard Similarity: ", temp.max().max())

# %% [markdown]
# ## For Each Topic Cluster: Odds Ratio of Having a Complication vs. Covid Vaccination Status
#
# Each topic cluster is a subpopulation of patients based on a clinical phenotype (the Topic).
# For each subpopulation, we can then conduct a retrospective case-control study.
# * Case: Patients who have received at least 1 COVID vaccine
# * Control: Patients who have not received COVID vaccine
#
# We then measure the odds of complication occuring.  The MPOG Database documents 4 types
# of complications that we can look at.
# * Pulmonary Complication: https://phenotypes.mpog.org/AHRQ%20Complication%20-%20Pulmonary%20-%20All
# * Cardiac Complication: [not publically documented on MPOG website]
# * Myocardial Infarction Complication: [not publically documented on MPOG website]
# * AKI Complication: https://phenotypes.mpog.org/MPOG%20Complication%20-%20Acute%20Kidney%20Injury%20(AKI)
#
# Odds Ratio = odds of complication in vaccinated / odds of complication in unvaccinated
#
# Interpretation of Odds Ratio:
# * Odds Ratio = 1: No difference between groups
# * Odds Ratio > 1: Unvaccinated group has more complications
# * Odds Ratio < 1: Unvaccinated group has fewer complications

# %%
# Compute Odds Ratios, p-values, confidence intervals


def compute_odds_ratio_and_chisquared(
    df: pd.DataFrame,
    var1: str,
    var2: str,
    topics: list[str],
    alpha: float = 0.05,
    statistical_significance_threshold: float = 0.05,
    null_odds: float = 1.0,
    var1_pos_name: str = "Yes",
    var2_pos_name: str = "Yes",
    invert_odds_ratios: bool = False,
) -> pd.DataFrame:
    _df = df.copy()
    # Get only relevant columns
    _df = _df.loc[:, [var1, var2, *topics]]

    results = []
    for topic in topics:
        # Narrow data to only rows in the topic cluster
        data = _df.loc[_df[topic]]
        # Create Contingency Table object
        table = sm.stats.Table.from_data(data)
        # Wrap in a 2x2 Contingency Table object
        t22 = sm.stats.Table2x2(table.table)
        odds_ratio = t22.oddsratio
        odds_ratio_lcb, odds_ratio_ucb = t22.oddsratio_confint(alpha=alpha)
        odds_ratio_pvalue = t22.oddsratio_pvalue(null=null_odds)
        significant = odds_ratio_pvalue < statistical_significance_threshold
        if invert_odds_ratios:
            odds_ratio = 1 / odds_ratio
            odds_ratio_lcb = 1 / odds_ratio_lcb
            odds_ratio_ucb = 1 / odds_ratio_ucb
            # Swap values since inverting flips LCB & UCB
            odds_ratio_lcb, odds_ratio_ucb = odds_ratio_ucb, odds_ratio_lcb
        # Chi-squared test of independence
        # chi_squared = t22.test_nominal_association()
        # Support
        num_cases = data.shape[0]
        num_var1_pos = data.loc[:, var1].eq(var1_pos_name).sum()
        num_var2_pos = data.loc[:, var2].eq(var2_pos_name).sum()

        result = {
            "NumCases": num_cases,
            f"Num{var1}": num_var1_pos,
            f"Num{var2}": num_var2_pos,
            "OddsRatio": odds_ratio,
            "OddsRatio_LCB": odds_ratio_lcb,
            "OddsRatio_UCB": odds_ratio_ucb,
            "OddsRatio_pvalue": odds_ratio_pvalue,
            "Significant": significant,
            # "ChiSquared_statistic": chi_squared.statistic,
            # "ChiSquared_df": chi_squared.df,
            # "ChiSquared_pvalue": chi_squared.pvalue,
        }
        results += [result]
    return pd.DataFrame(results, index=topics)


pulm_topic_stats = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadPulmonaryComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
num_unvaccinated = pulm_topic_stats.NumCases - pulm_topic_stats.NumHadCovidVaccine
pulm_topic_stats = pulm_topic_stats.assign(NumHadCovidVaccine=num_unvaccinated).rename(
    columns={
        "NumHadPulmonaryComplication2": "NumComplications",
        "NumHadCovidVaccine": "NumUnvaccinated",
    }
)

cardiac_topic_stats = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadCardiacComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
num_unvaccinated = cardiac_topic_stats.NumCases - cardiac_topic_stats.NumHadCovidVaccine
cardiac_topic_stats = cardiac_topic_stats.assign(NumHadCovidVaccine=num_unvaccinated).rename(
    columns={
        "NumHadCardiacComplication2": "NumComplications",
        "NumHadCovidVaccine": "NumUnvaccinated",
    }
)

aki_topic_stats = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadAKIComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
num_unvaccinated = aki_topic_stats.NumCases - aki_topic_stats.NumHadCovidVaccine
aki_topic_stats = aki_topic_stats.assign(NumHadCovidVaccine=num_unvaccinated).rename(
    columns={
        "NumHadAKIComplication2": "NumComplications",
        "NumHadCovidVaccine": "NumUnvaccinated",
    }
)

# For each topic, get p-values for each hypothesis test (each complication outcome)
p_vals_df = pd.concat(
    [
        df["OddsRatio_pvalue"].rename(k)
        for k, df in {
            "Pulmonary": pulm_topic_stats,
            "Cardiac": cardiac_topic_stats,
            "AKI": aki_topic_stats,
        }.items()
    ],
    axis=1,
)
# Benjamini/Hochberg procedure for p-value adjustment
reject_null_list, p_vals_adj_list = [], []
for topic_p_vals in p_vals_df.itertuples(index=False):
    reject_null, p_vals_adj, _, _ = sm.stats.multipletests(
        pvals=topic_p_vals, alpha=0.05, method="fdr_bh"
    )
    reject_null_list += [reject_null]
    p_vals_adj_list += [p_vals_adj]
p_vals_adj_df = pd.DataFrame(
    np.stack(p_vals_adj_list), index=p_vals_df.index, columns=p_vals_df.columns
)
reject_null_df = pd.DataFrame(
    np.stack(reject_null_list), index=p_vals_df.index, columns=p_vals_df.columns
)

# Replace OddsRatio_pvalue & Significant columns with adjusted values
pulm_topic_stats = pulm_topic_stats.assign(
    OddsRatio_pvalue=p_vals_adj_df["Pulmonary"], Significant=reject_null_df["Pulmonary"]
)
cardiac_topic_stats = cardiac_topic_stats.assign(
    OddsRatio_pvalue=p_vals_adj_df["Cardiac"], Significant=reject_null_df["Cardiac"]
)
aki_topic_stats = aki_topic_stats.assign(
    OddsRatio_pvalue=p_vals_adj_df["AKI"], Significant=reject_null_df["AKI"]
)

# Define Topic Labels (Topic Feature Blends)
topic_label = topic_features_str.to_frame().assign(
    Phenotype=[f"Phenotype {x}" for x in range(len(topic_features_str))]
)
topic_label = (
    topic_label.apply(lambda row: f"{row.Phenotype}: {row.TopicBlend}", axis=1)
    .rename("TopicBlend")
    .to_frame()
)

# Add Topic Labels & Complication Group Membership
pulm_topic_stats = topic_label.assign(Group="Pulmonary Complications").join(pulm_topic_stats)
cardiac_topic_stats = topic_label.assign(Group="Cardiac Complications").join(cardiac_topic_stats)
aki_topic_stats = topic_label.assign(Group="Acute Kidney Injury Complications").join(
    aki_topic_stats
)
# Combine all complications in single dataframe
data = pd.concat([pulm_topic_stats, cardiac_topic_stats, aki_topic_stats], axis=0).rename_axis(
    index="TopicName"
)
data = data.astype({"NumCases": str, "NumComplications": str, "NumUnvaccinated": str})
oddsratio_ci_str = data.apply(
    lambda row: f"{row.OddsRatio:.2f} ({row.OddsRatio_LCB:.2f} to {row.OddsRatio_UCB:.2f})", axis=1
)
data = data.assign(OddsRatio_CI=oddsratio_ci_str)

complication_group_dtype = CategoricalDtype(
    categories=[
        "Pulmonary Complications",
        "Cardiac Complications",
        "Acute Kidney Injury Complications",
    ],
    ordered=True,
)

data = data.astype({"Group": complication_group_dtype})
data = data.reset_index().astype({"TopicName": topic_dtype, "Group": complication_group_dtype})
# %%
print("Odds Ratios of Having a Complication if has had at least 1 Covid Vaccine")
# Display only Statistically Significant Topics
data_significant = data.loc[data.Significant].sort_values(by=["TopicName", "Group"])
data_significant
# %%
# Forest Plot of All Data
ax = forestplot(
    dataframe=data,
    estimate="OddsRatio",
    ll="OddsRatio_LCB",
    hl="OddsRatio_UCB",
    pval="OddsRatio_pvalue",
    starpval=True,
    varlabel="Group",
    groupvar="TopicBlend",
    xticks=range(0, 5),
    xline=1,
    xlabel="Odds Ratio",
    ylabel="Risk of Complications in COVID Unvaccinated Patients for Clinical Phenotypes ",
    annote=["NumCases", "NumComplications", "NumUnvaccinated", "OddsRatio_CI"],
    annoteheaders=["N", "Complications", "Unvaccinated", "Odds Ratio (95% Conf. Int.)"],
    color_alt_rows=True,
    table=True,
    figsize=(8, 20),
    # # Additional kwargs for customizations
    **{
        "marker": "D",  # set maker symbol as diamond
        "markersize": 35,  # adjust marker size
        "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
        "xlinecolor": "#808080",  # gray color for x-reference line
        "xtick_size": 12,  # adjust x-ticker fontsize
        "xlowerlimit": 0,
        "xupperlimit": 5,
    },
)
# %%
# Forest Plot of Only Significant Data
ax = forestplot(
    dataframe=data_significant,
    estimate="OddsRatio",
    ll="OddsRatio_LCB",
    hl="OddsRatio_UCB",
    pval="OddsRatio_pvalue",
    starpval=True,
    varlabel="Group",
    groupvar="TopicBlend",
    xticks=range(0, 5),
    xline=1,
    xlabel="Odds Ratio",
    ylabel="Risk of Complications in COVID Unvaccinated Patients for Clinical Phenotypes (Significant P-Values Only)",
    annote=["NumCases", "NumComplications", "NumUnvaccinated", "OddsRatio_CI"],
    annoteheaders=["N", "Complications", "Unvaccinated", "Odds Ratio (95% Conf. Int.)"],
    color_alt_rows=True,
    table=True,
    figsize=(8, 8),
    # # Additional kwargs for customizations
    **{
        "marker": "D",  # set maker symbol as diamond
        "markersize": 35,  # adjust marker size
        "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
        "xlinecolor": "#808080",  # gray color for x-reference line
        "xtick_size": 12,  # adjust x-ticker fontsize
        "xlowerlimit": 0,
        "xupperlimit": 5,
    },
)
# %%
