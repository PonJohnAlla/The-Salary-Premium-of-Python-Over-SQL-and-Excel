"""
app.py  —  Data Analyst Salary Premium Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import helper as h
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Data Analyst Salary Premium Project",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ─────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING  (cached — bundled CSV, no upload needed)
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = Path(__file__).parent / "DataAnalyst.csv"

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    return h.prepare(str(DATA_PATH))

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "",
        options=[
            "🏠  Introduction & Data Prep",
            "🔍  Skill Landscape",
            "💰  Salary Analysis",
            "🏭  Industry & Size",
            "📝  Conclusion",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        '<p style="color:#8b949e;font-size:0.72rem;">Research by Group 1 · 2024
        <br>''Data: Glassdoor via Kaggle</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# ── PAGE: INTRODUCTION ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

if page.startswith("🏠"):

    st.markdown(
        """
        <div class="hero-banner">
          <h1>🔰 The Salary Premium of Python over SQL and Excel</h1>
          <p>
            As part of a sprint project for Eskwelabs Data Analytics Bootcamp, our team
            did a labor research on the data analyst field. The study conducted looked 
            at the increasing skill demand needed by career shifting or well-established
            data analysts. 
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
            """
            **Created by** Group 1 (Pon John Alla, Luis Cabugos Jr., David Ken Del Mundo, Matthew Joaquin, Beau Nieve)
            """
        )

    # ── Dataset Preview ────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗂️ Dataset Preview")

    with st.expander("View raw data sample", expanded=False):
        st.dataframe(
            df.head(10),
            use_container_width=True,
        )

    col_r, col_c = st.columns(2)
    col_r.metric("Rows", f"{df.shape[0]:,}")
    col_c.metric("Columns", f"{df.shape[1]:,}")
    
    st.markdown(
    """
    <div class="callout">
        <p><strong>Sources Used:</strong></p>
        <ul>
            <li>Data Source: LinkedIn Webscrape</li>
            <li>Future of Jobs Report: <a href="https://www.weforum.org/publications/the-future-of-jobs-report-2025/digest/" target="_blank">World Economic Forum, 2025</a></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

    # ── KPI row ────────────────────────────────────────────────────
    f_stat, p_val, py_m, sql_m, xl_m = h.anova_test(df)
    pct_sql = ((py_m - sql_m) / sql_m) * 100
    pct_xl  = ((py_m - xl_m)  / xl_m)  * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Postings Analyzed", f"{len(df):,}")
    c2.metric("Python Avg Salary",        f"${py_m:,.0f}")
    c3.metric("Python Premium vs SQL",    f"+{pct_sql:.1f}%")
    c4.metric("Python Premium vs Excel",  f"+{pct_xl:.1f}%")

    st.divider()

    # ── Research question ──────────────────────────────────────────
    col_q, col_aud = st.columns([3, 2], gap="large")

    with col_q:
        st.markdown("### 🔬 Research Question")
        st.markdown(
            """
            <div class="callout">
            Do data analyst job postings that require <strong>Python</strong> offer
            higher salaries than those requiring only <strong>SQL</strong> or
            <strong>Excel</strong> — and does that salary premium hold when we
            account for <em>industry</em> and <em>company size</em>?
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Context")
        st.markdown(
            """
            The U.S. Bureau of Labor Statistics projects **34% employment growth**
            for data professionals from 2024–2034 — roughly 23,400 annual openings.
            As demand scales, employers increasingly signal salary intent through the
            technical skills they list. This study mines those signals from
            **2,253 Glassdoor job postings**.
            """
        )

    with col_aud:
        st.markdown("### 🎯 Who Is This For?")
        st.markdown(
            """
            **Career Shifters** transitioning into data roles:
            - Should I learn Python or SQL first?
            - Is Excel still relevant in 2024?
            - How much will each skill add to my offer?

            **Practicing Analysts** evaluating upskilling ROI:
            - Will Python move the needle on my salary?
            - Is AWS/Azure worth adding to my stack?
            - Does the premium hold in my industry?
            """
        )

    st.divider()

    # ── Skill ladder ───────────────────────────────────────────────
    st.markdown("### 🪜 The Skill Salary Ladder")
    st.markdown(
        """
        <table class="tier-table">
          <tr>
            <td class="tier-label">TIER 4</td>
            <td class="t4">⬆ Python + SQL + Cloud (AWS / Azure) — Highest salary ceiling</td>
          </tr>
          <tr>
            <td class="tier-label">TIER 3</td>
            <td class="t3">Python + SQL — Strong premium</td>
          </tr>
          <tr>
            <td class="tier-label">TIER 2</td>
            <td class="t2">SQL only  or  Python only — Moderate salary</td>
          </tr>
          <tr>
            <td class="tier-label">TIER 1</td>
            <td class="t1">Excel only — Entry-level baseline</td>
          </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # ── Data cleaning summary ──────────────────────────────────────
    st.divider()
    st.markdown("### 🧹 Data Cleaning Overview")
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Raw Rows",      "2,253")
    dc2.metric("Rows Removed",  "1")
    dc3.metric("Clean Rows",    f"{len(df):,}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("data clean.png", caption="Data Cleaning Flow", width=450)


    st.markdown("#### Cleaning Process")
    st.markdown(
            """
            The dataset was loaded at 2,253 rows, cleaned by parsing the text-based salary column into a numeric Avg Salary field, and validated  with only 1 row dropped  leaving 2,252 rows ready for analysis.
            """
        )

    st.markdown(
        """
        <div>
        The <code>cleaning process</code> had to standardize and parce the pricing. utilizing the 
        <code>parse_salary()</code> function
         
        </div>
        
        """,
        unsafe_allow_html=True,
    )
    
    st.code("""
def parse_salary(salary_str):
    try:
        # Remove non-numeric characters except dash and K
        salary_str = salary_str.replace('$', '').replace('(Glassdoor est.)', '').strip()
        parts = salary_str.split('-')
        low  = float(parts[0].replace('K','').strip()) * 1000
        high = float(parts[1].replace('K','').strip()) * 1000
        return (low + high) / 2
    except:
        return np.nan
""", language="python")
    
   


# ═══════════════════════════════════════════════════════════════════
# ── PAGE: SKILL LANDSCAPE ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

elif page.startswith("🔍"):

    st.markdown("## 🔍 Skill Landscape")
    st.markdown(
        "Before comparing salaries, we need to understand *how often* each tool "
        "appears in job postings — and *what tasks* employers associate with each."
    )

    # ── Mention counts ─────────────────────────────────────────────
    st.markdown("### Tool Mention Frequency")
    sdf = h.skill_mention_pct(df)

    c_py  = int(sdf[sdf["Skill"] == "Python"]["Mentions"].iloc[0])
    c_sql = int(sdf[sdf["Skill"] == "Sql"]["Mentions"].iloc[0])
    c_xl  = int(sdf[sdf["Skill"] == "Excel"]["Mentions"].iloc[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Python mentions", f"{c_py:,}",  f"{c_py/len(df)*100:.1f}% of postings")
    m2.metric("SQL mentions",    f"{c_sql:,}", f"{c_sql/len(df)*100:.1f}% of postings")
    m3.metric("Excel mentions",  f"{c_xl:,}",  f"{c_xl/len(df)*100:.1f}% of postings")

    tab1, tab2, tab3 = st.tabs(["All Tools", "Top 3 Bar", "Pareto Chart"])
    with tab1:
        st.pyplot(h.chart_skill_mentions_all(df), use_container_width=True)
    with tab2:
        st.pyplot(h.chart_top3_bar(df), use_container_width=True)
    with tab3:
        st.pyplot(h.chart_pareto(df), use_container_width=True)

    st.markdown(
        """
        <div class="callout">
        <strong>Why focus on Python, SQL, and Excel?</strong><br>
        These three tools dominate the demand landscape — SQL leads by volume,
        Python follows, and Excel is a near-universal baseline. Together they
        account for the top tier of postings and represent a natural "skill ladder"
        from entry-level to advanced analytics capability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Experience distribution ────────────────────────────────────
    st.markdown("### Job Postings by Experience Level")
    st.pyplot(h.chart_experience_dist(df), use_container_width=True)
    st.markdown(
        """
        <div class="callout">
        The majority of postings expect multiple years of experience — the largest
        share asking for 5+. For career shifters this underscores the value of
        building a strong <em>skills portfolio</em> to offset limited tenure.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Keyword context ────────────────────────────────────────────
    st.markdown("### What Tasks Do Employers Associate with Each Tool?")
    st.markdown(
        "Using TF-IDF on job descriptions, we extracted the top associated "
        "phrases for each tool — filtered to remove generic HR boilerplate."
    )

    with st.spinner("Extracting keywords (this may take ~10 s)…"):
        kw_figs = h.keyword_charts(df)

    cols = st.columns(len(kw_figs))
    for col, (title, fig) in zip(cols, kw_figs):
        with col:
            st.markdown(
                f'<span class="badge badge-{title.lower()}">{title}</span>',
                unsafe_allow_html=True,
            )
            st.pyplot(fig, use_container_width=True)

    st.markdown(
        """
        | Skill | Typical Use Cases |
        |-------|-------------------|
        | **Python** | Machine learning, statistical modelling, automation, data pipelines |
        | **SQL** | Database querying, data extraction, ETL, schema management |
        | **Excel** | Pivot tables, business reporting, ad-hoc operational analysis |
        """
    )


# ═══════════════════════════════════════════════════════════════════
# ── PAGE: SALARY ANALYSIS ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

elif page.startswith("💰"):

    st.markdown("## 💰 Salary Analysis")

    # ── Summary table ──────────────────────────────────────────────
    st.markdown("### Average Salary by Skill Group")
    summary = h.skill_summary(df)
    st.dataframe(
        summary[summary.index.isin(["Python", "SQL Only", "Excel Only"])]
        .style.format({"Mean Salary": "${:,.0f}", "Median Salary": "${:,.0f}"}),
        use_container_width=True,
    )

    col_bar, col_pct = st.columns(2, gap="large")
    with col_bar:
        st.markdown("#### Mean Salary Comparison")
        st.pyplot(h.chart_mean_salary(df), use_container_width=True)
    with col_pct:
        st.markdown("#### Python Premium (%)")
        st.pyplot(h.chart_pct_premium(df), use_container_width=True)

    _, _, py_m, sql_m, xl_m = h.anova_test(df)
    pct_sql = ((py_m - sql_m) / sql_m) * 100
    pct_xl  = ((py_m - xl_m)  / xl_m)  * 100

    st.markdown(
        f"""
        <div class="callout green">
        Python-required postings pay on average
        <strong>${py_m:,.0f}</strong> — a <strong>+{pct_sql:.2f}%</strong> premium
        over SQL-only (<strong>${sql_m:,.0f}</strong>) and
        <strong>+{pct_xl:.2f}%</strong> over Excel-only
        (<strong>${xl_m:,.0f}</strong>).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Box plot ───────────────────────────────────────────────────
    st.markdown("### Salary Distribution (Box Plot)")
    st.pyplot(h.chart_boxplot(df), use_container_width=True)
    st.markdown(
        """
        <div class="callout">
        Python roles sit <strong>higher overall</strong> in the salary distribution.
        SQL-only and Excel-only overlap significantly, with Excel-only showing the
        lowest median — consistent with its role as an entry-level baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Statistical test ───────────────────────────────────────────
    st.markdown("### Statistical Significance — One-Way ANOVA")
    f_stat, p_val, _, _, _ = h.anova_test(df)

    s1, s2, s3 = st.columns(3)
    s1.metric("F-Statistic", f"{f_stat:.4f}")
    s2.metric("P-Value",     f"{p_val:.4f}")
    s3.metric("Significant?", "✅ Yes" if p_val < 0.05 else "❌ No",
              "p < 0.05" if p_val < 0.05 else "p ≥ 0.05")

    if p_val < 0.05:
        st.markdown(
            """
            <div class="callout green">
            The ANOVA test confirms the salary differences are <strong>statistically
            significant</strong> (p < 0.05), meaning they are extremely unlikely to
            be due to random chance alone.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("The difference is not statistically significant at the 0.05 level.")


# ═══════════════════════════════════════════════════════════════════
# ── PAGE: INDUSTRY & SIZE ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

elif page.startswith("🏭"):

    st.markdown("## 🏭 Does the Premium Hold Across Industries & Company Sizes?")
    st.markdown(
        "Controlling for external factors: is Python's salary advantage consistent, "
        "or does it vary by where you work?"
    )

    tab_ind, tab_sz = st.tabs(["By Industry Sector", "By Company Size"])

    with tab_ind:
        st.markdown("### Salary by Industry Sector")
        st.pyplot(h.chart_industry(df), use_container_width=True)
        st.markdown(
            """
            <div class="callout">
            <ul style="margin:0;padding-left:1.2rem;line-height:2">
              <li>Python commands higher salaries in <strong>4 of 5</strong> sectors analysed.</li>
              <li>Largest premiums in <strong>Information Technology</strong> and
                  <strong>Health Care</strong> (~$7K–$9K over SQL-only).</li>
              <li>Finance and Business Services show moderate gaps due to heavy reliance
                  on SQL-based reporting.</li>
              <li><strong>Education</strong> is the lone exception where SQL-only
                  slightly outperforms Python.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_sz:
        st.markdown("### Salary by Company Size")
        st.pyplot(h.chart_company_size(df), use_container_width=True)
        st.markdown(
            """
            <div class="callout">
            <ul style="margin:0;padding-left:1.2rem;line-height:2">
              <li>Python outperforms SQL-only in <strong>6 of 7</strong> company-size
                  bands.</li>
              <li>Strongest advantage in <strong>mid-to-large firms</strong>
                  (1K–5K employees): ~$5K–$7K premium.</li>
              <li>Smaller firms (1–50 employees) also reward Python (~$7K), likely
                  because small teams need one analyst who can do everything.</li>
              <li>Very large firms (5K–10K employees) are the only exception — SQL
                  expertise alone is highly valued in specialist database roles.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# ── PAGE: CONCLUSION ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════

elif page.startswith("📝"):

    st.markdown("## 📝 Conclusion & Recommendations")

    f_stat, p_val, py_m, sql_m, xl_m = h.anova_test(df)
    pct_sql = ((py_m - sql_m) / sql_m) * 100
    pct_xl  = ((py_m - xl_m)  / xl_m)  * 100

    st.markdown("### Three Main Takeaways")

    st.markdown(
        f"""
        <div class="callout green">
        <strong>1 · Python commands a statistically significant salary premium.</strong><br>
        Job postings requiring Python pay on average <strong>${py_m:,.0f}</strong> —
        {pct_sql:.1f}% more than SQL-only and {pct_xl:.1f}% more than Excel-only.
        One-way ANOVA confirms the result is not due to chance (p = {p_val:.4f}).
        </div>

        <div class="callout">
        <strong>2 · Python elevates analytical capability beyond foundational tools.</strong><br>
        Python enables machine learning, complex statistical analysis, and automation —
        positioning analysts for more sophisticated roles and higher compensation than
        those relying purely on SQL querying or spreadsheet work.
        </div>

        <div class="callout red">
        <strong>3 · The premium is real but contextual.</strong><br>
        The advantage is strongest in <strong>IT, Healthcare, and Finance</strong>
        and in <strong>small-to-mid-sized companies</strong>. It narrows or reverses
        in Education and very large enterprises where deep SQL specialisation is
        independently valued.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    col_cs, col_pa = st.columns(2, gap="large")

    with col_cs:
        st.markdown("### 🎓 For Career Shifters")
        st.markdown(
            """
            <table class="tier-table">
              <tr><td class="tier-label">Step 1</td>
                  <td class="t1"><strong>Excel</strong> — universal entry point,
                  still required across all posting types</td></tr>
              <tr><td class="tier-label">Step 2</td>
                  <td class="t2"><strong>SQL</strong> — most-mentioned skill by volume,
                  opens the widest range of job opportunities</td></tr>
              <tr><td class="tier-label">Step 3</td>
                  <td class="t3"><strong>Python</strong> — statistically-backed salary
                  premium; highest-leverage upskill once you have SQL</td></tr>
              <tr><td class="tier-label">Step 4</td>
                  <td class="t4"><strong>AWS / Azure</strong> — cloud layer on top of
                  Python forms the highest-earning combination in this dataset</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )

    with col_pa:
        st.markdown("### 📈 For Practicing Analysts")
        st.markdown(
            """
            | Question | Answer |
            |----------|--------|
            | Will Python move the needle on salary? | **Yes** — ~6.6% over SQL-only |
            | Where is the premium strongest? | IT, Healthcare, Finance |
            | Is cloud worth it? | Python + AWS/Azure = highest salary tier |
            | Any exceptions? | Education & very large firms (5K–10K) |
            """
        )

    st.divider()

    st.markdown("### Skills Under the Microscope")
    st.markdown(
        """
        | Skill | Why It Matters | Who Needs It |
        |-------|----------------|--------------|
        | **Excel** | Universal entry point — every analyst touches it | Career Shifters — start here |
        | **SQL** | Language of databases — most in-demand by volume | Career Shifters + Analysts |
        | **Python** | Automation, analysis, ML-readiness — the premium skill | Analysts ready to level up |
        | **AWS / Azure** | Cloud fluency for big-data pipelines — the next frontier | Senior Analysts / Engineers |
        """
    )

    st.divider()
    st.markdown(
        '<p style="color:#8b949e;font-size:0.8rem;text-align:center;">'
        'Analysis based on 2,253 Glassdoor Data Analyst postings · Group 1 Research · 2024'
        '</p>',
        unsafe_allow_html=True,
    )
