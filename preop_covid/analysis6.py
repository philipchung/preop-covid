# %%
from pathlib import Path
from typing import Any

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


majority_topic = W_df.apply(get_largest_index_in_series, axis=1).rename("MajorityTopic")
# %%
# Define Non-Topic Columns for Modeling
case_info_cols = [
    "ASA",
    "AnesUnits",
    "Age",
    "BMI",
    "Race",
    "Sex",
    "AnesDuration",
    "HadCovidVaccine",
]
case_info_df = df.loc[:, case_info_cols]
# Convert ASA to Numeric
asa_numeric = case_info_df.ASA.apply(lambda x: "".join(filter(str.isdigit, x))).astype(int)
# Convert Race to Categorical, then generate 1-hot Matrix across Categories
race_cat_dtype = CategoricalDtype(categories=case_info_df.Race.unique(), ordered=False)
race_cat = case_info_df.Race.astype(race_cat_dtype)
race_cat_dummy_df = pd.get_dummies(data=race_cat)
# Convert Biological Sex to Binary
sex_binary = (case_info_df.Sex == "Male").astype(int).rename("IsMaleSex")
# Convert Anesthesia Duration to Hours
anes_duration = case_info_df.AnesDuration.apply(lambda dt: dt.total_seconds() / (60 * 60))
# Convert HadCovidVaccine to Binary
had_covid_vaccine = (df.HadCovidVaccine == "Yes").astype(int)
# Reformat Non-Topic Columns for Modeling
case_info_df2 = (
    case_info_df.assign(
        ASA=asa_numeric,
        AnesDuration=anes_duration,
        IsMaleSex=sex_binary,
        HadCovidVaccine=had_covid_vaccine,
    )
    .drop(columns=["Race", "Sex"])
    .join(race_cat_dummy_df)
)
# %% [markdown]
# ## Logistic Regression for Odds Ratios
#
# 1. Compute odds of complication vs. covid vaccination status overall
# 2. For Each Clinical Phenotype: Odds Ratio of Having a Complication vs. Covid Vaccination Status
#   - we assign each case to the most dominant clinical phenotype
#   - there is no overlap of cases between these clinical phenotype subpopulations
# %%


def compute_logistic_regression(
    X: pd.DataFrame, y: pd.Series, alpha: float = 0.05, method: str = "bfgs", maxiter: int = 1000
) -> dict[str, Any]:
    model = sm.Logit(exog=X, endog=y, offset=None, missing="raise").fit(
        method=method, maxiter=maxiter
    )
    odds_ratios = pd.DataFrame(
        {
            "OR": model.params,
            "LowerCI": model.conf_int()[0],
            "UpperCI": model.conf_int()[1],
        }
    )
    odds_ratios = np.exp(odds_ratios)
    pvalues = model.pvalues.rename("PValues")
    significant = (pvalues <= alpha).rename("Significant")
    output = odds_ratios.join(pvalues).join(significant)
    # Support
    num_cases = X.shape[0]
    num_vaccinated = X.loc[:, "HadCovidVaccine"].sum()
    num_complications = y.sum()
    return {
        "output": output,
        "num_cases": num_cases,
        "num_vaccinated": num_vaccinated,
        "num_complications": num_complications,
        "model": model,
    }


## Get Data of All Cases
# Input Data Matrix for Modeling with Topics + Non-Topic Variables
X_all_cases_df = W_df.join(case_info_df2)
# Convert Presence of Pulmonary Complications to Binary
y = df.HadPulmonaryComplication2.apply(lambda x: x == "Yes").astype(int)

## Apply Logistic Regression to All Cases
all_cases_results = compute_logistic_regression(X=X_all_cases_df, y=y)
# Isolate Odds Ratios for Covid Vaccination
all_cases_odds_ratios = all_cases_results["output"]
all_cases_covid_vaccine_odds_ratio = all_cases_odds_ratios.loc["HadCovidVaccine"]
# Add Number of Cases
all_cases_covid_vaccine_odds_ratio = pd.Series(
    {
        "NumCases": all_cases_results["num_cases"],
        "NumVaccinated": all_cases_results["num_vaccinated"],
        "NumComplications": all_cases_results["num_complications"],
        **all_cases_covid_vaccine_odds_ratio.to_dict(),
    }
)
# %%
## Apply Logistic Regression independently to in each Clinical Phenotype
topic_results = []
for topic in topic_names:
    # Get Cases for each Clinical Phenotype
    topic_case_ids = majority_topic.loc[majority_topic == topic].index.tolist()
    # Drop Topic/Clinical Phenotype Input Variables since we are Subsetting on Them
    X_topic_df = X_all_cases_df.loc[topic_case_ids].drop(columns=topic_names)
    y_topic = y.loc[topic_case_ids]
    # Apply Logistic Regression to only Caes in the Clinical Phenotype
    topic_result = compute_logistic_regression(X=X_topic_df, y=y_topic)
    topic_results += [{"topic": topic, **topic_result}]

# All Odds ratios for each topic
odds_ratios = {d["topic"]: d["output"] for d in topic_results}

# Isolate Odds Ratios for Covid Vaccination
topic_covid_vaccine_odds_ratios = {k: v.loc["HadCovidVaccine"] for k, v in odds_ratios.items()}
# Make into DataFrame
topic_covid_vaccine_odds_ratios = pd.DataFrame.from_dict(
    topic_covid_vaccine_odds_ratios, orient="index"
)
# Num Cases for each topic
num_cases = {
    d["topic"]: {
        "NumCases": d["num_cases"],
        "NumVaccinated": d["num_vaccinated"],
        "NumComplications": d["num_complications"],
    }
    for d in topic_results
}
num_cases = pd.DataFrame.from_dict(num_cases, orient="index")

topic_covid_vaccine_odds_ratios = num_cases.join(topic_covid_vaccine_odds_ratios)
# %%
# Put all odds ratios in same table
covid_vaccine_odds_ratio_df = pd.concat(
    [
        all_cases_covid_vaccine_odds_ratio.rename("All Cases").to_frame().T,
        topic_covid_vaccine_odds_ratios,
    ],
    axis=0,
).rename_axis(index="Population")

# Topic Labels (Topic Feature Blends) for each Subpopulation
topic_label = topic_features_str.to_frame().assign(
    Phenotype=[f"Phenotype {x}" for x in range(len(topic_features_str))]
)
topic_label = topic_label.apply(lambda row: f"{row.Phenotype}: {row.TopicBlend}", axis=1)
# Add "All Cases"
population_label = (
    pd.Series({"All Cases": "All Cases", **topic_label.to_dict()}).rename("Population").to_frame()
)

# Single DataFrame with All Populations being considered
data = population_label.join(covid_vaccine_odds_ratio_df)
# Convert Confidence Interval to a String
oddsratio_ci_str = data.apply(
    lambda row: f"{row.OR:.2f} ({row.LowerCI:.2f} to {row.UpperCI:.2f})", axis=1
)
data = data.assign(
    OR_CI=oddsratio_ci_str,
    # Compute Inverse Odds Ratios (Risk of Harm for Unvaccinated)
    NumUnvaccinated=data.NumCases - data.NumVaccinated,
    OR_Inverse=1 / data.OR,
    LowerCI_Inverse=1 / data.UpperCI,
    UpperCI_Inverse=1 / data.LowerCI,
)
# Convert Inverse Odds Ratio Confidence Interval to a String
inv_oddsratio_ci_str = data.apply(
    lambda row: f"{row.OR_Inverse:.2f} ({row.LowerCI_Inverse:.2f} to {row.UpperCI_Inverse:.2f})",
    axis=1,
)
data = data.assign(OR_Inverse_CI=inv_oddsratio_ci_str)
data
# %%

# %%
# Forest Plot of All Data
ax = forestplot(
    dataframe=data,
    estimate="OR_Inverse",
    ll="LowerCI_Inverse",
    hl="UpperCI_Inverse",
    pval="PValues",
    starpval=True,
    varlabel="Population",
    xticks=range(0, 5),
    xline=1,
    xlabel="Odds Ratio",
    ylabel="Risk of Complications in COVID Unvaccinated Patients",
    annote=["NumCases", "NumComplications", "NumUnvaccinated", "OR_Inverse_CI"],
    annoteheaders=["N", "Complications", "Unvaccinated", "Odds Ratio (95% Conf. Int.)"],
    color_alt_rows=True,
    table=True,
    figsize=(6, 10),
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
print("Odds Ratios of Having a Complication if has had at least 1 Covid Vaccine")
# Display only Statistically Significant Topics
data_significant = data.loc[data.Significant]
data_significant

# %%
# Forest Plot of Only Significant Data
ax = forestplot(
    dataframe=data_significant,
    estimate="OR_Inverse",
    ll="LowerCI_Inverse",
    hl="UpperCI_Inverse",
    pval="PValues",
    starpval=True,
    varlabel="Population",
    xticks=range(0, 5),
    xline=1,
    xlabel="Odds Ratio",
    ylabel="Risk of Complications in COVID Unvaccinated Patients",
    annote=["NumCases", "NumComplications", "NumUnvaccinated", "OR_Inverse_CI"],
    annoteheaders=["N", "Complications", "Unvaccinated", "Odds Ratio (95% Conf. Int.)"],
    color_alt_rows=True,
    table=True,
    figsize=(6, 6),
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
