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

from preop_covid.case_data import CaseData
from preop_covid.lab_data import LabData
from preop_covid.preop_data import PreopSDE
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
# Put Transformed Data Matrix W into a Dataframe
W_df = pd.DataFrame(data=W, columns=topic_names, index=X_df.index)

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
topic_features_norm_df
# %%
# Create mapping between topic name and a topic "alias"/identity
topic_features_map_df = topic_features_norm_df.apply(lambda row: "\n".join(row), axis=1).rename(
    "TopicBlend"
)
topic_features_map_df2 = topic_features_norm_df.apply(lambda row: ", ".join(row), axis=1).rename(
    "TopicBlend"
)
topic_features_map = topic_features_map_df.to_dict()
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
    topic_features_map_df, how="left", left_on="Topic", right_index=True
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
#
# Each case is a blend of Topics--we can convert the contribution of Topics to a case
# such that all the 20 topics all add up to 100%.
# We empirically choose a threshold=**25%**.
# **This means that a clinical phenotype is determined to be present for a patient if >25%
# of the Cases' ROS is explained by that clinical phenotype.**  This threshold converts
# the soft/fuzzy clusters into a discrete subpopulation of patients associated with the
# clinical phenotype.  This threshold allows for multiple clinical phenotypes to be
# present, but also requires a large portion of the patient’s ROS to be explained by
# the phenotype.
#
# Note that by this definition, it is possible for a Case to be in multiple Topic clusters.
# It is possible for a Case to not be in any Topic clusters.  The goal is not to force
# all the cases into the Topics, which is what many other clustering algorithms do,
# but rather yield cases that are representative of the Topics within our dataset
# that we can then use as a cohort for further analysis.

# %%
# Normalize Transformed Data Matrix by Row
# Interpretation: % of each Topic that makes up each case's ROS
W_df_norm = W_df.apply(lambda row: row / row.sum(), axis=1)
# Apply Threshold Cut-off to get a "soft" cluster
# Note: clusters may overlap & a single case may belong to multiple clusters
threshold = 0.25  # Cluster membership if >10% of Case's ROS is explained by topic
mask = W_df_norm.applymap(lambda x: x > threshold)
# Percent of examples in each topic cluster (clusters may overlap)
num_case = mask.sum().rename("NumCases")
percent_case = (mask.sum() / mask.shape[0]).rename("FractionOfCases")
pd.DataFrame.from_dict(data=topic_name2alias, orient="index", columns=["TopFeature"]).join(
    topic_features_map_df2
).join(num_case).join(percent_case)

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
        chi_squared = t22.test_nominal_association()
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
            "ChiSquared_statistic": chi_squared.statistic,
            "ChiSquared_df": chi_squared.df,
            "ChiSquared_pvalue": chi_squared.pvalue,
        }
        results += [result]
    return pd.DataFrame(results, index=topics)


print("Odds Ratios of Having a Pulmonary Complication if has had at least 1 Covid Vaccine")
topic_statistics = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadPulmonaryComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
topic_statistics = topic_features_map_df2.to_frame().join(topic_statistics)
# Display only Statistically Significant Topics
topic_statistics.loc[topic_statistics.Significant]

# %%
print("Odds Ratios of Having a Cardiac Complication if has had at least 1 Covid Vaccine")
topic_statistics = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadCardiacComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
topic_statistics = topic_features_map_df2.to_frame().join(topic_statistics)
# Display only Statistically Significant Topics
topic_statistics.loc[topic_statistics.Significant]
# %%
print(
    "Odds Ratios of Having a Myocardial Infarction Complication if has had at least 1 Covid Vaccine"
)
topic_statistics = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadMyocardialInfarctionComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
topic_statistics = topic_features_map_df2.to_frame().join(topic_statistics)
# Display only Statistically Significant Topics
topic_statistics.loc[topic_statistics.Significant]
# %%
print("Odds Ratios of Having a AKI Complication if has had at least 1 Covid Vaccine")
topic_statistics = compute_odds_ratio_and_chisquared(
    df=df.copy().join(mask),
    var1="HadAKIComplication2",
    var2="HadCovidVaccine",
    topics=mask.columns,
    invert_odds_ratios=True,
)
topic_statistics = topic_features_map_df2.to_frame().join(topic_statistics)
# Display only Statistically Significant Topics
topic_statistics.loc[topic_statistics.Significant]
# %%
