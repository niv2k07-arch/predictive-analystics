"""
app.py
-------
Student Performance Analytics Dashboard.

Pipeline: Raw CSV -> Clean -> Process -> Analyze -> Visualize -> Insights,
all wired into one interactive Streamlit app with live filters.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_cleaning import load_raw_data, clean_pipeline, detect_outliers_iqr
from data_processing import process_pipeline, PASS_THRESHOLD
from data_analysis import (
    overall_stats, department_analysis, subject_analysis,
    attendance_correlation, top_and_bottom_departments,
)
import visualizations as viz

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "student_data.csv")
CLEANED_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned", "cleaned_student_data.csv")

st.set_page_config(page_title="Student Performance Analytics", layout="wide")


# ---------------------------------------------------------------------------
# Data pipeline (cached so filters don't re-run cleaning every interaction)
# ---------------------------------------------------------------------------
@st.cache_data
def run_pipeline(path: str):
    raw_df = load_raw_data(path)
    clean_result = clean_pipeline(raw_df)
    cleaned_df = clean_result["cleaned_df"]
    quality_report = clean_result["quality_report"]

    processed_df = process_pipeline(cleaned_df, pass_threshold=PASS_THRESHOLD)
    outliers = detect_outliers_iqr(processed_df)
    quality_report["outliers_total"] = sum(v["count"] for v in outliers.values())
    quality_report["outliers_by_col"] = outliers

    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    processed_df.to_csv(CLEANED_PATH, index=False)

    return raw_df, processed_df, quality_report, outliers


if not os.path.exists(RAW_PATH):
    st.error(f"Raw dataset not found at `{RAW_PATH}`. Add a CSV there (see README) and reload.")
    st.stop()

raw_df, processed_df, quality_report, outliers = run_pipeline(RAW_PATH)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Student Performance Analytics")
st.caption("Raw Data → Cleaning → Processing → Analysis → Visualization → Insights")

# ---------------------------------------------------------------------------
# Filters (sidebar)
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")


def option_list(series: pd.Series):
    return ["All"] + sorted(series.dropna().unique().tolist())


dept_filter = st.sidebar.selectbox("Department", option_list(processed_df["Department"]))
gender_filter = st.sidebar.selectbox("Gender", option_list(processed_df["Gender"]))
semester_filter = st.sidebar.selectbox("Semester", option_list(processed_df["Semester"]))
perf_filter = st.sidebar.selectbox("Performance Category", option_list(processed_df["Performance_Category"]))

filtered_df = processed_df.copy()
if dept_filter != "All":
    filtered_df = filtered_df[filtered_df["Department"] == dept_filter]
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
if semester_filter != "All":
    filtered_df = filtered_df[filtered_df["Semester"] == semester_filter]
if perf_filter != "All":
    filtered_df = filtered_df[filtered_df["Performance_Category"] == perf_filter]

st.sidebar.caption(f"Showing **{len(filtered_df)}** of {len(processed_df)} cleaned records")
st.sidebar.divider()
st.sidebar.caption(f"Pass criterion: ≥ {PASS_THRESHOLD} marks in every subject")

# ---------------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------------
stats = overall_stats(filtered_df)

if not stats:
    st.warning("No records match the selected filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students", stats["total_students"])
c2.metric("Average Marks", stats["average_marks"])
c3.metric("Pass %", f"{stats['pass_percentage']}%")
c4.metric("Average Attendance", f"{stats['average_attendance']}%")

st.divider()

# ---------------------------------------------------------------------------
# Data quality section
# ---------------------------------------------------------------------------
st.header("🧹 Data Quality")

q1, q2, q3, q4, q5, q6 = st.columns(6)
q1.metric("Original Records", quality_report["original_records"])
q2.metric("Duplicate Records", quality_report["duplicates_before"])
q3.metric("Missing Values", quality_report["missing_before"])
q4.metric("Invalid Values", quality_report["invalid_before"])
q5.metric("Potential Outliers", quality_report["outliers_total"])
q6.metric("Cleaned Records", quality_report["cleaned_records"])

with st.expander("Before vs After Cleaning — details"):
    before_after = pd.DataFrame({
        "Metric": ["Records", "Missing Values", "Duplicates", "Invalid Values"],
        "Before": [
            quality_report["original_records"],
            quality_report["missing_before"],
            quality_report["duplicates_before"],
            quality_report["invalid_before"],
        ],
        "After": [
            quality_report["cleaned_records"],
            quality_report["missing_after"],
            quality_report["duplicates_after"],
            quality_report["invalid_after"],
        ],
    })
    st.table(before_after)
    st.caption(
        "Missing and invalid values were imputed with the column median (numeric) "
        "or 'Unknown' (categorical) rather than dropped, so student records are preserved. "
        "Duplicate Student_IDs were removed, keeping the first occurrence."
    )

st.divider()

# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------
st.header("🎓 Performance Analysis")

dept_df = department_analysis(filtered_df)
subj = subject_analysis(filtered_df)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Department Performance")
    st.pyplot(viz.department_performance_chart(dept_df))
    st.dataframe(dept_df, use_container_width=True, hide_index=True)
with col_right:
    st.subheader("Subject Performance")
    st.pyplot(viz.subject_performance_chart(subj["summary"]))
    st.dataframe(subj["summary"], use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Attendance analysis
# ---------------------------------------------------------------------------
st.header("📅 Attendance Analysis")
corr = attendance_correlation(filtered_df)
st.pyplot(viz.attendance_vs_marks_chart(filtered_df))
if corr is not None:
    strength = "weak" if abs(corr) < 0.3 else "moderate" if abs(corr) < 0.6 else "strong"
    direction = "positive" if corr >= 0 else "negative"
    st.caption(
        f"Observed correlation between attendance and average marks: **{corr}** "
        f"({strength} {direction} relationship). This is an observation from the data, "
        "not a claim that attendance directly causes better marks."
    )

st.divider()

# ---------------------------------------------------------------------------
# Distribution & outliers
# ---------------------------------------------------------------------------
st.header("📈 Performance Distribution & Outliers")
d1, d2 = st.columns(2)
with d1:
    st.subheader("Marks Distribution")
    st.pyplot(viz.marks_distribution_chart(filtered_df))
with d2:
    st.subheader("Performance Categories")
    st.pyplot(viz.performance_category_chart(filtered_df))

st.subheader("Outlier Analysis (Box Plots)")
st.pyplot(viz.outlier_boxplots(filtered_df))

st.divider()

# ---------------------------------------------------------------------------
# Key insights (generated from the actual filtered data, never hardcoded)
# ---------------------------------------------------------------------------
st.header("💡 Key Insights")

best_dept, worst_dept = top_and_bottom_departments(dept_df)
most_common_category = max(stats["performance_counts"], key=stats["performance_counts"].get) \
    if stats["performance_counts"] else "N/A"

insights = []
if best_dept:
    insights.append(f"**{best_dept}** has the highest average performance among departments.")
if worst_dept and worst_dept != best_dept:
    insights.append(f"**{worst_dept}** has the lowest average performance among departments.")
insights.append(f"The most common performance category is **{most_common_category}**.")
insights.append(f"Average attendance across the selected students is **{stats['average_attendance']}%**.")
if subj["highest_subject"]:
    insights.append(f"**{subj['highest_subject']}** is the highest-performing subject on average.")
if subj["lowest_subject"]:
    insights.append(f"**{subj['lowest_subject']}** is the lowest-performing subject on average.")
if corr is not None:
    insights.append(
        f"Attendance and marks show a {('positive' if corr >= 0 else 'negative')} "
        f"relationship (r = {corr}) in this data."
    )
insights.append(f"**{quality_report['outliers_total']}** potential outliers were detected via the IQR method.")

for line in insights:
    st.markdown(f"- {line}")
