# %%
from pathlib import Path

import nimfa
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from pandas.api.types import CategoricalDtype

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


# Categorical Dtype for Topics
topic_dtype = CategoricalDtype(
    categories=H_df.index.tolist(),
    ordered=True,
)
# Categorical Dtype for Problems
ros_problems_dtype = CategoricalDtype(
    categories=H_df.columns.tolist(),
    ordered=False,
)

# Set Indices as Categorical
topic_names = pd.CategoricalIndex(data=H_df.index, dtype=topic_dtype)
ros_problem_names = pd.CategoricalIndex(data=H_df.columns, dtype=ros_problems_dtype)
H_df.index = topic_names
H_df.columns = ros_problem_names
W_df.columns = topic_names

# %%
# Get Top 5 features for each topic & the percent each feature contributes to the topic
# Normalize the component matrix by each topic
H_df_norm = H_df.apply(lambda row: row / row.sum(), axis=1)


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
percent_threshold = 0.05
topic_features = H_df_norm.apply(
    lambda row: get_topic_blend(row, top_n=5, threshhold=percent_threshold), axis=1
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
print("Top 5 Features for each Topic")
topic_features_norm_df = pd.DataFrame.from_dict(
    data=topic_top5_features_lst.to_dict(),
    orient="index",
    columns=[f"TopFeature{k+1}" for k in range(5)],
)
topic_features_norm_df
# %%
print(f"Topic Blends (Top 5 Features & Threshold > {percent_threshold}% Weight for Feature)")
topic_features_str.to_dict()
# %%
primary_topic_name2alias = H_df_norm.apply(
    lambda row: [f"{k}" for k in row.nlargest(1).keys()][0], axis=1
).to_dict()
primary_topic_aliases = [primary_topic_name2alias[n] for n in topic_names]


# %% [markdown]
# ### Assign Majority Phenotype to each Case as a Label
# %%
def get_largest_index_in_series(series: pd.Series) -> str:
    d = series.to_dict()
    return max(d, key=d.get)


majority_topic = W_df.apply(get_largest_index_in_series, axis=1)
majority_topic
# %%

non_topic_variables = [""]

non_topic_variables_df = X_df.loc[:, non_topic_variables]
# %%
X_topics_df = majority_topic.join(W_df).join(X_df.loc[:, non_topic_variables])
X_topics_df
# %%
# TODO:
# - get dominant phenotype for each case as label
# - define data inputs/outputs for whole-population logistic regression
# - create subpopulations for logistic regression


# %%

# %%
# Normalize Transformed Data Matrix by Row
# Interpretation: % of each Topic that makes up each case's ROS
W_df_norm = W_df.apply(lambda row: row / row.sum(), axis=1)

# %%

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
