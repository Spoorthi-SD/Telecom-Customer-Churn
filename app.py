import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Customer Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP - REMOVE TOP SPACE
       ======================================================== */

    .stApp {
        background-color: #0b0d10 !important;
        color: #f5f1ea !important;
        font-size: 17px !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0b0d10 !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    section.main > div {
        padding-top: 0rem !important;
    }

    section.main > div.block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    .main .block-container {
        max-width: 1450px !important;
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }

    [data-testid="stHeader"] {
        background-color: #0b0d10 !important;
        height: 0rem !important;
    }

    [data-testid="stToolbar"] {
        top: 0rem !important;
    }


    /* ========================================================
       REMOVE TOP MARGIN FROM FIRST CONTENT
       ======================================================== */

    .main .block-container > div:first-child {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }

    .main h1:first-of-type {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }


    /* ========================================================
       GENERAL FONT
       ======================================================== */

    p {
        color: #e0ddd7 !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #fffaf3 !important;
    }

    h1 {
        font-size: 2.5rem !important;
        line-height: 1.2 !important;
        margin-top: 0.15rem !important;
    }

    h2 {
        font-size: 1.95rem !important;
        line-height: 1.25 !important;
    }

    h3 {
        font-size: 1.5rem !important;
        line-height: 1.3 !important;
    }

    label {
        font-size: 16px !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #111418 !important;
        border-right: 1px solid #30353b !important;

        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;

        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.35rem !important;
        padding-bottom: 0.25rem !important;

        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;

        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }

    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;

        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;

        overflow-x: hidden !important;
        overflow-y: hidden !important;
    }


    /* ========================================================
       SIDEBAR TITLE
       ======================================================== */

    [data-testid="stSidebar"] h1 {
        font-size: 22px !important;
        line-height: 1.2 !important;

        margin: 0 !important;
        padding: 0 !important;

        color: #ffffff !important;
        font-weight: 800 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: normal !important;

        max-width: 100% !important;
    }


    /* ========================================================
       SIDEBAR SUBTITLE
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        margin-top: 0 !important;
        margin-bottom: 5px !important;

        max-width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 14px !important;
        line-height: 1.3 !important;

        color: #aeb3b8 !important;

        margin: 0 !important;
        padding: 0 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
    }


    /* ========================================================
       SIDEBAR HEADINGS
       ======================================================== */

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 800 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: normal !important;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 18px !important;
        line-height: 1.25 !important;

        margin: 6px 0 4px 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 17px !important;
        line-height: 1.25 !important;

        margin: 6px 0 4px 0 !important;
        padding: 0 !important;
    }


    /* ========================================================
       SIDEBAR TEXT
       ======================================================== */

    [data-testid="stSidebar"] p {
        font-size: 15px !important;
        line-height: 1.35 !important;

        margin: 0 !important;
        padding: 0 !important;

        color: #d9d6d0 !important;
        opacity: 1 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-break: normal !important;
    }

    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;

        font-size: 16px !important;
        font-weight: 800 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
    }


    /* ========================================================
       SIDEBAR DIVIDERS
       ======================================================== */

    [data-testid="stSidebar"] hr {
        margin: 6px 0 !important;
        border-color: #30353b !important;
    }


    /* ========================================================
       SIDEBAR NAVIGATION
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stRadio"] {
        margin: 0 !important;
        padding: 0 !important;

        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #ffffff !important;

        font-size: 16px !important;
        line-height: 1.3 !important;

        padding: 3px 0 !important;
        margin: 0 !important;

        max-width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] p {
        color: #ffffff !important;

        font-size: 16px !important;
        line-height: 1.3 !important;

        white-space: normal !important;
        overflow-wrap: break-word !important;
    }


    /* ========================================================
       SIDEBAR SLIDER
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        margin-top: 2px !important;
        margin-bottom: 4px !important;
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        font-size: 15px !important;
        line-height: 1.3 !important;

        color: #ffffff !important;
        margin-bottom: 2px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] p {
        font-size: 15px !important;
        color: #ffffff !important;
    }


    /* ========================================================
       GENERAL WIDGET LABELS
       ======================================================== */

    [data-testid="stWidgetLabel"] {
        color: #fffaf3 !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] label {
        color: #fffaf3 !important;
        opacity: 1 !important;

        font-weight: 700 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #fffaf3 !important;
        opacity: 1 !important;

        font-weight: 700 !important;
        font-size: 17px !important;
    }


    /* ========================================================
       NUMBER INPUT LABELS
       ======================================================== */

    div[data-testid="stNumberInput"] label,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"],
    div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] label,
    div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {

        color: #17191d !important;
        -webkit-text-fill-color: #17191d !important;

        opacity: 1 !important;
        visibility: visible !important;

        font-weight: 700 !important;
        font-size: 17px !important;
    }


    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] {
        background-color: #f1f3f6 !important;
        border-radius: 10px !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #f1f3f6 !important;

        border: 1px solid #d8dce2 !important;
        border-radius: 10px !important;

        min-height: 52px !important;

        opacity: 1 !important;
    }

    div[data-baseweb="select"] span {
        color: #17191d !important;
        opacity: 1 !important;

        font-weight: 600 !important;
        font-size: 17px !important;

        -webkit-text-fill-color: #17191d !important;
    }

    div[data-baseweb="select"] div {
        color: #17191d !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #30343a !important;
        color: #30343a !important;
        opacity: 1 !important;
    }


    /* ========================================================
       DROPDOWN
       ======================================================== */

    [data-baseweb="popover"] {
        background-color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"] * {
        color: #17191d !important;
        opacity: 1 !important;
    }

    [role="option"] {
        color: #17191d !important;
        background-color: #ffffff !important;

        opacity: 1 !important;

        font-size: 17px !important;
    }

    [role="option"] span {
        color: #17191d !important;
        opacity: 1 !important;
    }

    [role="option"]:hover {
        background-color: #eeeeee !important;
        color: #111418 !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-testid="stNumberInput"] {
        background-color: #f1f3f6 !important;
        border-radius: 10px !important;
        opacity: 1 !important;
    }

    div[data-testid="stNumberInput"] > div {
        background-color: #f1f3f6 !important;

        border: 1px solid #d8dce2 !important;
        border-radius: 10px !important;

        opacity: 1 !important;
    }

    div[data-testid="stNumberInput"] input {
        color: #17191d !important;
        background-color: #f1f3f6 !important;

        font-weight: 700 !important;
        font-size: 17px !important;

        opacity: 1 !important;

        -webkit-text-fill-color: #17191d !important;
    }

    div[data-testid="stNumberInput"] button {
        color: #17191d !important;
        background-color: #f1f3f6 !important;

        border: none !important;
        opacity: 1 !important;
    }

    div[data-testid="stNumberInput"] button svg {
        fill: #17191d !important;
        color: #17191d !important;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    input {
        color: #17191d !important;
        -webkit-text-fill-color: #17191d !important;

        opacity: 1 !important;
        font-size: 17px !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #ff765f !important;
        color: #111418 !important;

        border: 1px solid #ff765f !important;
        border-radius: 10px !important;

        font-weight: 800 !important;
        font-size: 16px !important;

        min-height: 48px !important;
    }

    .stButton > button p {
        color: #111418 !important;
        font-size: 16px !important;
    }

    .stButton > button:hover {
        background-color: #ff927e !important;
        border-color: #ff927e !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #171b20 !important;

        border: 1px solid #343a41 !important;
        border-radius: 14px !important;

        padding: 17px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #bfc4c9 !important;
        font-size: 16px !important;
    }

    [data-testid="stMetricValue"] {
        color: #fffaf3 !important;
        font-size: 29px !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #15191e !important;

        border-color: #343a41 !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: #f5f1ea !important;
        font-size: 16px !important;
    }

    [data-testid="stExpander"] summary p {
        color: #f5f1ea !important;
        font-size: 16px !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {
        color: #747b82 !important;

        text-align: center !important;

        font-size: 14px !important;

        margin-top: 25px !important;
        padding-top: 12px !important;

        border-top: 1px solid #292f35 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# COLORS
# ============================================================

CORAL = "#ff765f"
CYAN = "#63e6d6"
CREAM = "#f5f1ea"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "model/churn_model.pkl"
    )

    preprocessing = joblib.load(
        "model/preprocessing.pkl"
    )

    return model, preprocessing


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

@st.cache_data
def load_model_comparison():

    return pd.read_csv(
        "outputs/model_comparison.csv"
    )


# ============================================================
# LOAD THRESHOLD
# ============================================================

@st.cache_data
def load_threshold_analysis():

    return pd.read_csv(
        "outputs/threshold_analysis.csv"
    )


# ============================================================
# LOAD
# ============================================================

df = load_data()

model, preprocessing = load_model()

model_comparison = load_model_comparison()

threshold_df = load_threshold_analysis()

scaler = preprocessing["scaler"]

numerical_features = preprocessing[
    "numerical_features"
]

feature_names = preprocessing[
    "feature_names"
]


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(fig, height=380):

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="#0b0d10",

        plot_bgcolor="#0b0d10",

        height=height,

        font={
            "family": "Arial",
            "color": CREAM,
            "size": 15
        },

        title={
            "font": {
                "family": "Arial",
                "color": CREAM,
                "size": 20
            },
            "x": 0.02,
            "xanchor": "left"
        },

        margin={
            "l": 60,
            "r": 30,
            "t": 70,
            "b": 55
        },

        xaxis={
            "color": CREAM,
            "gridcolor": "#292f35",
            "linecolor": "#454b52",
            "tickfont": {
                "color": "#d9d6cf",
                "size": 14
            },
            "title_font": {
                "color": CREAM,
                "size": 15
            }
        },

        yaxis={
            "color": CREAM,
            "gridcolor": "#292f35",
            "linecolor": "#454b52",
            "tickfont": {
                "color": "#d9d6cf",
                "size": 14
            },
            "title_font": {
                "color": CREAM,
                "size": 15
            }
        },

        legend={
            "font": {
                "color": CREAM,
                "size": 14
            }
        }
    )

    return fig


# ============================================================
# SESSION STATE
# ============================================================

if "dashboard_page" not in st.session_state:

    st.session_state.dashboard_page = 1


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📡 Telecom Customer Churn"
    )

    st.caption(
        "Machine Learning Dashboard"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔮 Customer Prediction"
        ]
    )

    st.divider()

    st.markdown(
        "### Machine Learning Model"
    )

    st.markdown(
        "**Logistic Regression**"
    )

    st.caption(
        "Selected after comparing Logistic Regression, "
        "Random Forest and XGBoost."
    )

    st.divider()

    st.markdown(
        "### Prediction Threshold"
    )

    prediction_threshold = st.slider(
        "Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.40,
        step=0.05
    )

    st.caption(
        f"Customers with predicted churn probability "
        f"≥ {prediction_threshold:.0%} are classified "
        f"as likely to churn."
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    current_page = st.session_state.dashboard_page

    dashboard_names = [
        "Overview",
        "Churn Analysis",
        "Model Performance",
        "Explainable AI",
        "Threshold Analysis"
    ]

    st.markdown(
        f"**TELECOM CUSTOMER CHURN / 0{current_page}**"
    )

    st.title(
        dashboard_names[current_page - 1]
    )

    st.caption(
        "Machine learning based telecom customer churn analysis."
    )

    st.progress(
        current_page / 5
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    if current_page == 1:

        st.header(
            "Customer Churn Intelligence"
        )

        st.write(
            "Understand customer behaviour and identify "
            "the overall churn pattern across the telecom dataset."
        )

        total_customers = len(df)

        churned_customers = (
            df["Churn"] == "Yes"
        ).sum()

        churn_rate = (
            churned_customers /
            total_customers
        ) * 100

        average_tenure = df["tenure"].mean()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Customers",
                f"{total_customers:,}"
            )

        with c2:
            st.metric(
                "Churned Customers",
                f"{churned_customers:,}"
            )

        with c3:
            st.metric(
                "Churn Rate",
                f"{churn_rate:.1f}%"
            )

        with c4:
            st.metric(
                "Average Tenure",
                f"{average_tenure:.1f} Months"
            )

        st.divider()

        st.subheader(
            "Customer Overview"
        )

        col1, col2 = st.columns(2)

        with col1:

            churn_counts = (
                df["Churn"]
                .value_counts()
                .reset_index()
            )

            churn_counts.columns = [
                "Churn",
                "Customers"
            ]

            fig = px.pie(
                churn_counts,
                names="Churn",
                values="Customers",
                hole=0.55,
                title="Customer Churn Distribution",
                color="Churn",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig = style_chart(
                fig,
                340
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            percentages = (
                df["Churn"]
                .value_counts(normalize=True)
                .reset_index()
            )

            percentages.columns = [
                "Churn",
                "Percentage"
            ]

            percentages["Percentage"] *= 100

            fig = px.bar(
                percentages,
                x="Churn",
                y="Percentage",
                text="Percentage",
                title="Churn Percentage",
                color="Churn",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig = style_chart(
                fig,
                340
            )

            fig.update_layout(
                xaxis_title="",
                yaxis_title="Percentage (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.info(
            "What is churn?\n\n"
            "Churn means a customer leaves or cancels "
            "their telecom service. The machine learning "
            "model estimates which customers may be likely "
            "to churn."
        )


    # ========================================================
    # CHURN ANALYSIS
    # ========================================================

    elif current_page == 2:

        st.header(
            "Churn Analysis"
        )

        st.write(
            "Explore how contracts, internet services, "
            "payment methods and customer behaviour relate to churn."
        )

        col1, col2 = st.columns(2)

        with col1:

            contract_churn = (
                pd.crosstab(
                    df["Contract"],
                    df["Churn"],
                    normalize="index"
                ) * 100
            )

            contract_churn = (
                contract_churn
                .reset_index()
                .melt(
                    id_vars="Contract",
                    var_name="Churn",
                    value_name="Percentage"
                )
            )

            fig = px.bar(
                contract_churn,
                x="Contract",
                y="Percentage",
                color="Churn",
                barmode="group",
                title="Churn by Contract Type",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig = style_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            internet_churn = (
                pd.crosstab(
                    df["InternetService"],
                    df["Churn"],
                    normalize="index"
                ) * 100
            )

            internet_churn = (
                internet_churn
                .reset_index()
                .melt(
                    id_vars="InternetService",
                    var_name="Churn",
                    value_name="Percentage"
                )
            )

            fig = px.bar(
                internet_churn,
                x="InternetService",
                y="Percentage",
                color="Churn",
                barmode="group",
                title="Churn by Internet Service",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig = style_chart(fig)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        payment_churn = (
            pd.crosstab(
                df["PaymentMethod"],
                df["Churn"],
                normalize="index"
            ) * 100
        )

        payment_churn = (
            payment_churn
            .reset_index()
            .melt(
                id_vars="PaymentMethod",
                var_name="Churn",
                value_name="Percentage"
            )
        )

        fig = px.bar(
            payment_churn,
            x="PaymentMethod",
            y="Percentage",
            color="Churn",
            barmode="group",
            title="Churn by Payment Method",
            color_discrete_map={
                "No": CYAN,
                "Yes": CORAL
            }
        )

        fig = style_chart(
            fig,
            390
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Customer Behaviour"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.box(
                df,
                x="Churn",
                y="tenure",
                color="Churn",
                title="Tenure Distribution by Churn",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig = style_chart(
                fig,
                330
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges by Churn",
                color_discrete_map={
                    "No": CYAN,
                    "Yes": CORAL
                }
            )

            fig = style_chart(
                fig,
                330
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        fig = px.box(
            df,
            x="Churn",
            y="TotalCharges",
            color="Churn",
            title="Total Charges by Churn",
            color_discrete_map={
                "No": CYAN,
                "Yes": CORAL
            }
        )

        fig = style_chart(
            fig,
            330
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    elif current_page == 3:

        st.header(
            "Model Performance"
        )

        st.write(
            "Compare Logistic Regression, Random Forest "
            "and XGBoost using multiple classification metrics."
        )

        final_model = model_comparison[
            model_comparison["Model"]
            == "Logistic Regression"
        ]

        if not final_model.empty:

            row = final_model.iloc[0]

            st.success(
                "Final Model: Logistic Regression"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric(
                    "Accuracy",
                    f"{row['Accuracy']:.4f}"
                )

            with c2:
                st.metric(
                    "Precision",
                    f"{row['Precision']:.4f}"
                )

            with c3:
                st.metric(
                    "Recall",
                    f"{row['Recall']:.4f}"
                )

            with c4:
                st.metric(
                    "F1 Score",
                    f"{row['F1 Score']:.4f}"
                )

            with c5:
                st.metric(
                    "ROC-AUC",
                    f"{row['ROC-AUC']:.4f}"
                )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            model_comparison.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            model_comparison,
            x="Model",
            y="ROC-AUC",
            text="ROC-AUC",
            color="Model",
            title="ROC-AUC Comparison",
            color_discrete_sequence=[
                CORAL,
                CYAN,
                "#c8a96b"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside"
        )

        fig = style_chart(
            fig,
            380
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Model Evaluation"
        )

        col1, col2 = st.columns(2)

        with col1:

            if os.path.exists(
                "outputs/confusion_matrix.png"
            ):

                st.image(
                    "outputs/confusion_matrix.png",
                    caption="Confusion Matrix",
                    use_container_width=True
                )

        with col2:

            if os.path.exists(
                "outputs/roc_curve.png"
            ):

                st.image(
                    "outputs/roc_curve.png",
                    caption="ROC Curve",
                    use_container_width=True
                )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    elif current_page == 4:

        st.header(
            "Explainable AI"
        )

        st.write(
            "Understand which customer features influence "
            "the model using Logistic Regression coefficients "
            "and SHAP explainability."
        )

        st.subheader(
            "Global Feature Importance"
        )

        col1, col2 = st.columns(2)

        with col1:

            if os.path.exists(
                "outputs/feature_importance.png"
            ):

                st.image(
                    "outputs/feature_importance.png",
                    caption="Logistic Regression Feature Coefficients",
                    use_container_width=True
                )

        with col2:

            if os.path.exists(
                "outputs/shap_feature_importance.png"
            ):

                st.image(
                    "outputs/shap_feature_importance.png",
                    caption="SHAP Feature Importance",
                    use_container_width=True
                )

        st.subheader(
            "SHAP Analysis"
        )

        col1, col2 = st.columns(2)

        with col1:

            if os.path.exists(
                "outputs/shap_summary.png"
            ):

                st.image(
                    "outputs/shap_summary.png",
                    caption="SHAP Feature Impact",
                    use_container_width=True
                )

        with col2:

            st.markdown(
                """
                ### How to Read SHAP

                **Positive SHAP value**

                Pushes the model toward a higher estimated
                churn probability.

                **Negative SHAP value**

                Pushes the model toward a lower estimated
                churn probability.

                **Important**

                SHAP explains model behaviour. These values
                describe how the model arrives at a prediction
                and should not be interpreted as proof of
                causal relationships.
                """
            )

        st.divider()

        st.subheader(
            "Individual Customer Explanation"
        )

        individual_file = (
            "outputs/shap_individual_explanation.csv"
        )

        if os.path.exists(individual_file):

            individual_df = pd.read_csv(
                individual_file
            )

            if len(individual_df) > 0:

                feature_column = None
                shap_column = None

                for column in individual_df.columns:

                    clean_column = (
                        str(column)
                        .strip()
                        .lower()
                        .replace("_", " ")
                        .replace("-", " ")
                    )

                    if clean_column in [
                        "feature",
                        "features"
                    ]:

                        feature_column = column

                    if clean_column in [
                        "shap value",
                        "shap values",
                        "shapvalue",
                        "shap"
                    ]:

                        shap_column = column

                if (
                    feature_column is not None
                    and shap_column is not None
                ):

                    individual_df = individual_df.rename(
                        columns={
                            feature_column: "Feature",
                            shap_column: "SHAP Value"
                        }
                    )

                    individual_df["SHAP Value"] = pd.to_numeric(
                        individual_df["SHAP Value"],
                        errors="coerce"
                    )

                    individual_df = individual_df.dropna(
                        subset=["SHAP Value"]
                    )

                    positive = individual_df[
                        individual_df["SHAP Value"] > 0
                    ].copy()

                    negative = individual_df[
                        individual_df["SHAP Value"] < 0
                    ].copy()

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            "### 🔴 Factors increasing churn risk"
                        )

                        if positive.empty:

                            st.write(
                                "No positive contributing factors."
                            )

                        else:

                            for _, row in positive.head(6).iterrows():

                                st.write(
                                    f"• **{row['Feature']}** "
                                    f"({row['SHAP Value']:+.3f})"
                                )

                    with col2:

                        st.markdown(
                            "### 🟢 Factors reducing churn risk"
                        )

                        if negative.empty:

                            st.write(
                                "No negative contributing factors."
                            )

                        else:

                            for _, row in negative.head(6).iterrows():

                                st.write(
                                    f"• **{row['Feature']}** "
                                    f"({row['SHAP Value']:+.3f})"
                                )

                else:

                    st.warning(
                        "The individual SHAP file does not contain "
                        "the expected Feature and SHAP Value columns."
                    )

                    st.caption(
                        "Available columns: "
                        + ", ".join(
                            map(
                                str,
                                individual_df.columns
                            )
                        )
                    )

        else:

            st.info(
                "Individual SHAP explanation file not found."
            )

        st.caption(
            "SHAP values explain the model's prediction "
            "for an individual customer. They represent "
            "model contribution, not causal effects."
        )


    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    elif current_page == 5:

        st.header(
            "Threshold Analysis"
        )

        st.write(
            "Study how changing the classification threshold "
            "affects precision, recall and F1 Score."
        )

        st.metric(
            "Current Prediction Threshold",
            f"{prediction_threshold:.0%}"
        )

        st.info(
            f"Customers with predicted churn probability "
            f"≥ {prediction_threshold:.0%} are classified "
            f"as likely to churn."
        )

        fig = px.line(
            threshold_df,
            x="Threshold",
            y=[
                "Precision",
                "Recall",
                "F1-Score"
            ],
            markers=True,
            title="Precision, Recall and F1 Score by Threshold"
        )

        fig = style_chart(
            fig,
            400
        )

        fig.update_layout(
            xaxis_title="Classification Threshold",
            yaxis_title="Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Threshold Comparison"
        )

        st.dataframe(
            threshold_df.style.format(
                {
                    "Threshold": "{:.2f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-Score": "{:.4f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Lower thresholds identify more customers as "
            "potential churners, increasing recall but usually "
            "reducing precision. Higher thresholds generally "
            "increase precision while reducing recall."
        )


    # ========================================================
    # DASHBOARD NAVIGATION
    # ========================================================

    st.divider()

    nav1, nav2, nav3 = st.columns(
        [1, 2, 1]
    )

    with nav1:

        if current_page > 1:

            if st.button(
                "← Previous",
                use_container_width=True
            ):

                st.session_state.dashboard_page -= 1

                st.rerun()

    with nav2:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                color:#f5f1ea;
                padding-top:8px;
                font-size:16px;
            ">
                PAGE {current_page} / 5
                <br>
                <b>{dashboard_names[current_page - 1]}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    with nav3:

        if current_page < 5:

            if st.button(
                "Next →",
                use_container_width=True
            ):

                st.session_state.dashboard_page += 1

                st.rerun()


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

else:

    st.markdown(
        "**TELECOM CUSTOMER CHURN / PREDICTION**"
    )

    st.title(
        "Customer Prediction"
    )

    st.caption(
        "Enter customer details to estimate churn probability."
    )


    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.header(
        "Customer Profile"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                "No",
                "Yes"
            ]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            [
                "No",
                "Yes"
            ]
        )

    with col4:

        dependents = st.selectbox(
            "Dependents",
            [
                "No",
                "Yes"
            ]
        )

    with col5:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )


    # ========================================================
    # SERVICES
    # ========================================================

    st.header(
        "Services"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes"
            ]
        )

    with col3:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col2:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col3:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    # ========================================================
    # BILLING
    # ========================================================

    st.header(
        "Billing"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "No",
                "Yes"
            ]
        )

    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.50
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )


    # ========================================================
    # THRESHOLD
    # ========================================================

    st.divider()

    st.info(
        f"Prediction threshold: "
        f"{prediction_threshold:.0%}"
    )


    # ========================================================
    # PREDICT
    # ========================================================

    if st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    ):

        senior_citizen_value = (
            1
            if senior_citizen == "Yes"
            else 0
        )

        customer_data = pd.DataFrame(
            {
                "gender": [gender],
                "SeniorCitizen": [
                    senior_citizen_value
                ],
                "Partner": [partner],
                "Dependents": [dependents],
                "tenure": [tenure],
                "PhoneService": [phone_service],
                "MultipleLines": [multiple_lines],
                "InternetService": [internet_service],
                "OnlineSecurity": [online_security],
                "OnlineBackup": [online_backup],
                "DeviceProtection": [device_protection],
                "TechSupport": [tech_support],
                "StreamingTV": [streaming_tv],
                "StreamingMovies": [streaming_movies],
                "Contract": [contract],
                "PaperlessBilling": [paperless_billing],
                "PaymentMethod": [payment_method],
                "MonthlyCharges": [
                    monthly_charges
                ],
                "TotalCharges": [
                    total_charges
                ]
            }
        )

        customer_encoded = pd.get_dummies(
            customer_data,
            drop_first=True
        )

        customer_encoded = customer_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        customer_encoded[
            numerical_features
        ] = scaler.transform(
            customer_encoded[
                numerical_features
            ]
        )

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        prediction = (
            1
            if probability >= prediction_threshold
            else 0
        )

        st.divider()

        st.header(
            "Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}"
            )

        with result_col2:

            if prediction == 1:

                st.error(
                    "⚠️ Likely to Churn"
                )

            else:

                st.success(
                    "✓ Likely to Stay"
                )

        st.subheader(
            "Churn Probability"
        )

        st.progress(
            float(probability)
        )

        st.subheader(
            "Prediction Explanation"
        )

        try:

            background = shap.sample(
                pd.read_csv(
                    "outputs/X_train.csv"
                ),
                100,
                random_state=42
            )

            explainer = shap.LinearExplainer(
                model,
                background
            )

            shap_values = explainer(
                customer_encoded
            )

            values = shap_values.values[0]

            explanation_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": values
                }
            )

            explanation_df[
                "Absolute SHAP"
            ] = explanation_df[
                "SHAP Value"
            ].abs()

            explanation_df = (
                explanation_df
                .sort_values(
                    "Absolute SHAP",
                    ascending=False
                )
            )

            positive_factors = (
                explanation_df[
                    explanation_df["SHAP Value"] > 0
                ]
                .head(6)
            )

            negative_factors = (
                explanation_df[
                    explanation_df["SHAP Value"] < 0
                ]
                .head(6)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🔴 Factors increasing churn risk"
                )

                if positive_factors.empty:

                    st.write(
                        "No major factors increasing churn risk."
                    )

                else:

                    for _, row in positive_factors.iterrows():

                        st.write(
                            f"• **{row['Feature']}** "
                            f"({row['SHAP Value']:+.3f})"
                        )

            with col2:

                st.markdown(
                    "### 🟢 Factors reducing churn risk"
                )

                if negative_factors.empty:

                    st.write(
                        "No major factors reducing churn risk."
                    )

                else:

                    for _, row in negative_factors.iterrows():

                        st.write(
                            f"• **{row['Feature']}** "
                            f"({row['SHAP Value']:+.3f})"
                        )

            st.caption(
                "SHAP values explain how the machine learning "
                "model contributed to this prediction. They "
                "should not be interpreted as causal effects."
            )

        except Exception:

            st.warning(
                "Individual SHAP explanation could not be generated."
            )

        with st.expander(
            "View Customer Details"
        ):

            st.dataframe(
                customer_data.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        Telecom Customer Churn Prediction System
        • Machine Learning • Explainable AI • SHAP
    </div>
    """,
    unsafe_allow_html=True
)