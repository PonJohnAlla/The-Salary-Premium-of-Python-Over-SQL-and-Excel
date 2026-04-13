"""
helper.py
All data loading, cleaning, and chart-building functions for the
Data Analyst Salary Premium Streamlit app.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import io

# ── Colour palette (matches the notebook) ──────────────────────────────────
PALETTE = {
    "Python":     "#1f77b4",
    "SQL Only":   "#d3d3d3",
    "Excel Only": "#808080",
    "accent":     "#e63946",
    "bg":         "#0d1117",
    "card":       "#161b22",
    "text":       "#e6edf3",
}

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "axes.titlecolor":   PALETTE["text"],
    "grid.color":        "#21262d",
    "axes.edgecolor":    "#30363d",
})


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════

def parse_salary(salary_str: str) -> float:
    """'$41K-$78K (Glassdoor est.)' → 59500.0"""
    try:
        s = salary_str.replace("$", "").replace("(Glassdoor est.)", "").strip()
        lo, hi = s.split("-")
        return (float(lo.replace("K", "").strip()) * 1000 +
                float(hi.replace("K", "").strip()) * 1000) / 2
    except Exception:
        return np.nan


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Avg Salary"] = df["Salary Estimate"].apply(parse_salary)
    df = df.dropna(subset=["Avg Salary"])
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

SKILLS_MAP = {
    "python":     "python",
    "sql":        "sql",
    "excel":      "excel",
    "tableau":    "tableau",
    "power_bi":   "power bi",
    "r":          " r ",
    "sas":        "sas",
    "aws":        "aws",
    "azure":      "azure",
    "spark":      "spark",
    "hadoop":     "hadoop",
    "looker":     "looker",
    "snowflake":  "snowflake",
    "bigquery":   "bigquery",
    "databricks": "databricks",
}


def add_skill_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["Job Description"].str.lower()
    for skill, kw in SKILLS_MAP.items():
        df[f"has_{skill}"] = desc.str.contains(kw, regex=False).astype(int)
    return df


def add_skill_group(df: pd.DataFrame) -> pd.DataFrame:
    def _group(row):
        if row["has_python"] == 1:
            return "Python"
        elif row["has_sql"] == 1:
            return "SQL Only"
        elif row["has_excel"] == 1:
            return "Excel Only"
        return "Other"
    df["Skill Group"] = df.apply(_group, axis=1)
    return df


def add_experience_level(df: pd.DataFrame) -> pd.DataFrame:
    df["Job Description"] = df["Job Description"].astype(str)
    df["Job Title"] = df["Job Title"].astype(str)
    text = (df["Job Title"] + " " + df["Job Description"]).str.lower()

    def _years(t):
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*years?", t)
        if m:
            return (int(m.group(1)) + int(m.group(2))) / 2
        m = re.search(r"(?:at least|minimum|min|over)\s*(\d+)\s*years?", t)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\+?\s*years?", t)
        if m:
            return int(m.group(1))
        if re.search(r"\b(entry level|entry-level|junior|associate)\b", t):
            return 1
        if re.search(r"\b(senior|lead|manager|principal|director)\b", t):
            return 6
        return np.nan

    def _cat(y):
        if pd.isna(y):
            return "Not Indicated"
        if y <= 1:
            return "Entry-Level (0-1 yrs)"
        elif y <= 4:
            return "Mid-Level (2-4 yrs)"
        return "Senior-Level (5+ yrs)"

    df["Years_Experience"] = text.apply(_years)
    df["Experience_Level"] = df["Years_Experience"].apply(_cat)
    return df


def prepare(path: str) -> pd.DataFrame:
    df = load_and_clean(path)
    df = add_skill_flags(df)
    df = add_skill_group(df)
    df = add_experience_level(df)
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════

def skill_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = (df.groupby("Skill Group")["Avg Salary"]
           .agg(["mean", "median", "count"])
           .round(2))
    s.columns = ["Mean Salary", "Median Salary", "Count"]
    return s.sort_values("Mean Salary", ascending=False)


def skill_mention_pct(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for skill in SKILLS_MAP:
        col = f"has_{skill}"
        if col in df.columns:
            rows.append({
                "Skill":    skill.replace("_", " ").title(),
                "Mentions": int(df[col].sum()),
                "Percent":  round(df[col].mean() * 100, 1),
            })
    out = pd.DataFrame(rows).sort_values("Mentions", ascending=False)
    out["Cumulative %"] = out["Percent"].cumsum()
    return out


def anova_test(df: pd.DataFrame):
    py  = df[df["Skill Group"] == "Python"]["Avg Salary"]
    sql = df[df["Skill Group"] == "SQL Only"]["Avg Salary"]
    xl  = df[df["Skill Group"] == "Excel Only"]["Avg Salary"]
    f, p = stats.f_oneway(py, sql, xl)
    return f, p, py.mean(), sql.mean(), xl.mean()


# ═══════════════════════════════════════════════════════════════════
# 4. CHARTS  (return matplotlib Figure objects)
# ═══════════════════════════════════════════════════════════════════

def _fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    return fig, ax


def chart_experience_dist(df: pd.DataFrame) -> plt.Figure:
    order = ["Senior-Level (5+ yrs)", "Mid-Level (2-4 yrs)", "Entry-Level (0-1 yrs)"]
    counts = (df[df["Experience_Level"] != "Not Indicated"]["Experience_Level"]
              .value_counts()
              .reindex(order)
              .fillna(0))
    fig, ax = _fig(9, 4)
    bars = ax.bar(counts.index, counts.values, color=PALETTE["Python"], width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4, color=PALETTE["text"], fontsize=9)
    ax.set_title("Job Postings by Experience Level", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Postings")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def chart_skill_mentions_all(df: pd.DataFrame) -> plt.Figure:
    sdf = skill_mention_pct(df)
    fig, ax = _fig(11, 5)
    colors = [PALETTE["Python"] if s.lower() in ("python", "sql", "excel") else "#30363d"
              for s in sdf["Skill"]]
    ax.barh(sdf["Skill"], sdf["Percent"], color=colors)
    for i, v in enumerate(sdf["Percent"]):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8, color=PALETTE["text"])
    ax.set_xlabel("% of Job Postings")
    ax.set_title("Tool Mentions in Data Analyst Job Postings", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def chart_top3_bar(df: pd.DataFrame) -> plt.Figure:
    sdf = skill_mention_pct(df).head(3)
    fig, ax = _fig(7, 4)
    bars = ax.bar(sdf["Skill"], sdf["Mentions"], color=PALETTE["Python"], width=0.5)
    ax.bar_label(bars, fmt="%d", padding=4, color=PALETTE["text"], fontsize=9)
    ax.set_title("Top 3 Skills in Job Postings", fontweight="bold", fontsize=13)
    ax.set_ylabel("Postings")
    plt.tight_layout()
    return fig


def chart_pareto(df: pd.DataFrame) -> plt.Figure:
    sdf = skill_mention_pct(df).head(5)
    fig, ax1 = _fig(10, 5)
    colors = [PALETTE["Python"] if s.lower() in ("python", "sql", "excel") else "#30363d"
              for s in sdf["Skill"]]
    sns.barplot(data=sdf, x="Skill", y="Percent", ax=ax1,
                hue="Skill", palette=dict(zip(sdf["Skill"], colors)), legend=False)
    ax1.set_ylabel("% of Postings")
    ax2 = ax1.twinx()
    ax2.plot(sdf["Skill"], sdf["Cumulative %"], color=PALETTE["accent"], marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative %", color=PALETTE["accent"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])
    ax1.set_title("Pareto: Top 5 Tool Mentions", fontweight="bold", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def chart_mean_salary(df: pd.DataFrame) -> plt.Figure:
    summary = skill_summary(df)
    plot_d = summary[summary.index.isin(["Python", "SQL Only", "Excel Only"])].copy()
    plot_d = plot_d.sort_values("Mean Salary", ascending=False)
    custom_pal = {k: PALETTE[k] for k in ["Python", "SQL Only", "Excel Only"]}
    fig, ax = _fig(8, 5)
    bars = sns.barplot(x=plot_d.index, y="Mean Salary", data=plot_d,
                       hue=plot_d.index, palette=custom_pal, legend=False, ax=ax)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 800,
                f'${bar.get_height():,.0f}',
                ha="center", va="bottom", fontsize=10, color=PALETTE["text"])
    ax.set_title("Average Salary by Skill Group", fontweight="bold", fontsize=13)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_xlabel("")
    plt.tight_layout()
    return fig


def chart_pct_premium(df: pd.DataFrame) -> plt.Figure:
    _, _, py_m, sql_m, xl_m = anova_test(df)
    pct_sql = ((py_m - sql_m) / sql_m) * 100
    pct_xl  = ((py_m - xl_m)  / xl_m)  * 100
    plot_d = pd.DataFrame({
        "Comparison": ["Python vs SQL Only", "Python vs Excel Only"],
        "Pct":        [pct_sql, pct_xl],
    })
    fig, ax = _fig(7, 4)
    bars = sns.barplot(x="Comparison", y="Pct", data=plot_d,
                       color=PALETTE["Python"], ax=ax)
    for idx, row in plot_d.iterrows():
        ax.text(idx, row["Pct"] + 0.2, f"{row['Pct']:.2f}%",
                ha="center", va="bottom", color=PALETTE["text"])
    ax.set_title("Python Salary Premium (%)", fontweight="bold", fontsize=13)
    ax.set_ylabel("% Increase")
    ax.set_xlabel("")
    plt.tight_layout()
    return fig


def chart_boxplot(df: pd.DataFrame) -> plt.Figure:
    groups = df[df["Skill Group"].isin(["Python", "SQL Only", "Excel Only"])]
    custom_pal = {k: PALETTE[k] for k in ["Python", "SQL Only", "Excel Only"]}
    fig, ax = _fig(9, 5)
    sns.boxplot(data=groups, x="Skill Group", y="Avg Salary",
                order=["Python", "Excel Only", "SQL Only"],
                hue="Skill Group", palette=custom_pal, legend=False, ax=ax)
    ax.set_title("Salary Distribution by Skill Group", fontweight="bold", fontsize=13)
    ax.set_ylabel("Avg Salary (USD)")
    ax.set_xlabel("")
    plt.tight_layout()
    return fig


def chart_industry(df: pd.DataFrame) -> plt.Figure:
    top_ind = df["Sector"].value_counts().head(6).index
    df_top = df[df["Sector"].isin(top_ind) & (df["Sector"] != "-1")]
    ind_sal = (df_top[df_top["Skill Group"].isin(["Python", "SQL Only", "Excel Only"])]
               .groupby(["Sector", "Skill Group"])["Avg Salary"]
               .mean().unstack("Skill Group").round(0))
    melted = ind_sal.reset_index().melt(id_vars="Sector",
                                        var_name="Skill Group",
                                        value_name="Avg Salary")
    custom_pal = {k: PALETTE[k] for k in ["Python", "SQL Only", "Excel Only"]}
    fig, ax = _fig(12, 5)
    sns.barplot(data=melted, x="Sector", y="Avg Salary",
                hue="Skill Group", palette=custom_pal,
                hue_order=["Python", "Excel Only", "SQL Only"], ax=ax)
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"${h:,.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE["text"])
    ax.set_title("Salary by Industry Sector & Skill Group", fontweight="bold", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Avg Salary (USD)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


def chart_company_size(df: pd.DataFrame) -> plt.Figure:
    size_order = [
        "1 to 50 employees", "51 to 200 employees",
        "201 to 500 employees", "501 to 1000 employees",
        "1001 to 5000 employees", "5001 to 10000 employees",
        "10000+ employees",
    ]
    df_sz = df[~df["Size"].isin(["-1", "Unknown"]) &
               df["Skill Group"].isin(["Python", "SQL Only", "Excel Only"])]
    sz_sal = (df_sz.groupby(["Size", "Skill Group"])["Avg Salary"]
              .mean().unstack("Skill Group").round(0))
    sz_sal = sz_sal.reindex([s for s in size_order if s in sz_sal.index])
    melted = sz_sal.reset_index().melt(id_vars="Size",
                                       var_name="Skill Group",
                                       value_name="Avg Salary")
    custom_pal = {k: PALETTE[k] for k in ["Python", "SQL Only", "Excel Only"]}
    fig, ax = _fig(13, 5)
    sns.barplot(data=melted, x="Size", y="Avg Salary",
                hue="Skill Group", palette=custom_pal,
                hue_order=["Python", "Excel Only", "SQL Only"], ax=ax)
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"${h:,.0f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7,
                            color=PALETTE["text"])
    ax.set_title("Salary by Company Size & Skill Group", fontweight="bold", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Avg Salary (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# 5. KEYWORD EXTRACTION
# ═══════════════════════════════════════════════════════════════════

GENERAL_STOPWORDS = [
    "data","experience","work","team","ability","skills","knowledge","strong","years",
    "business","analyst","analytics","working","role","job","company","requirements",
    "including","bachelor degree","master degree","equal opportunity","equal opportunity employer",
    "equal employment","opportunity employer","employment opportunity","qualified applicants",
    "receive consideration","medical dental","dental vision","paid time","competitive salary",
    "new york","york city","united states","san francisco","project management","time management",
    "best practices","ad hoc","internal external","fast paced","problem solving","decision making",
    "cross functional","written verbal","verbal written","actionable insights","sql python",
    "python sql","python java","sql programming","sql excel","sql server","sql queries",
    "power bi","microsoft excel","ms excel","pl sql","programming languages","excel powerpoint",
    "ms office","microsoft office","real time","preferred qualifications", "sexual orientation", "gender identity", "national origin", "veteran status",
"computer science", "related field", "marital status", "race color",
"color religion", "religion sex", "basis race", "origin age",
]

PYTHON_STOP  = GENERAL_STOPWORDS + ["vlookup","hlookup","pivot table","pivot tables","macros",
                                      "vba","spreadsheet","dashboards","data cleaning","ETL",
                                      "data modeling","etl","data warehouse","joins","queries",
                                      "database","schema","data extraction","relational database"]
SQL_STOP     = GENERAL_STOPWORDS + ["vlookup","hlookup","pivot table","pivot tables","macros",
                                      "vba","spreadsheet","dashboards","data cleaning","ETL",
                                      "pandas","numpy","scikit","tensorflow","keras","nlp",
                                      "machine learning","deep learning","automation","scripting",
                                      "data pipeline"]
EXCEL_STOP   = GENERAL_STOPWORDS + ["data modeling","etl","data warehouse","joins","queries",
                                      "database","schema","data extraction","relational database",
                                      "pandas","numpy","scikit","tensorflow","keras","nlp",
                                      "machine learning","deep learning","automation","scripting",
                                      "data pipeline","statistical analysis","statistical techniques"]


def _get_keywords(text_series, stopwords_list, top_n=10):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(2, 3), max_features=2000)
    X = vec.fit_transform(text_series)
    scores = X.sum(axis=0).A1
    words  = vec.get_feature_names_out()
    ranked = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

    # Build a flat set of individual stop tokens for fast lookup
    stop_tokens = set()
    for phrase in stopwords_list:
        for token in phrase.lower().split():
            stop_tokens.add(token)

    def _is_clean(phrase):
        # Reject if the phrase matches or contains any stopword phrase
        if any(s in phrase for s in stopwords_list):
            return False
        # Reject if any word in the phrase is a stop token
        if any(token in stop_tokens for token in phrase.split()):
            return False
        return True

    kws = [w for w, _ in ranked if _is_clean(w)]
    return kws[:top_n]


def _count_keywords(job_df, keywords):
    text = " ".join(job_df["Job Description"].astype(str)).lower()
    found = [(w, text.count(w)) for w in keywords if w in text]
    return sorted(found, key=lambda x: x[1], reverse=True)


def keyword_charts(df: pd.DataFrame):
    """Returns three figures: (python_fig, sql_fig, excel_fig)"""
    py_jobs = df[df["Job Description"].str.contains("python", case=False, na=False)]
    sq_jobs = df[df["Job Description"].str.contains("sql",    case=False, na=False)]
    xl_jobs = df[df["Job Description"].str.contains("excel",  case=False, na=False)]

    py_kws = _get_keywords(py_jobs["Job Description"], PYTHON_STOP)
    sq_kws = _get_keywords(sq_jobs["Job Description"], SQL_STOP)
    xl_kws = _get_keywords(xl_jobs["Job Description"], EXCEL_STOP)

    py_skills = _count_keywords(py_jobs, py_kws)[:6]
    sq_skills = _count_keywords(sq_jobs, sq_kws)[:6]
    xl_skills = _count_keywords(xl_jobs, xl_kws)[:6]

    figs = []
    for skills, title in [(py_skills, "Python"), (sq_skills, "SQL"), (xl_skills, "Excel")]:
        if not skills:
            continue
        words, counts = zip(*skills)
        fig, ax = _fig(8, 4)
        ax.barh(words, counts, color=PALETTE["Python"])
        ax.set_title(f"Top Skills Associated with {title}", fontweight="bold", fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        figs.append((title, fig))
    return figs
