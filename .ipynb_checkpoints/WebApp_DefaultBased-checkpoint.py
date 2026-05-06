import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# LOGIN SYSTEM
# =========================
def login():

    st.title("🔐 Login to Credit Risk System")

    # hardcoded credentials (you can change)
    USERNAME = "admin"
    PASSWORD = "1234"

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid Credentials")

# Session check
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Stop app if not logged in
if not st.session_state["logged_in"]:
    login()
    st.stop()


# =========================
# LOAD DATA + MODEL
# =========================
df = pd.read_csv("FinalData.csv")

model = joblib.load("default_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Risk System", layout="wide")

st.title("💳 Credit Risk Analytics System")
if st.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 KPI & Insights", "📈 Dashboard", "💳 Risk Prediction"])

# =========================================================
# TAB 1 — KPI & INSIGHTS
# =========================================================
with tab1:

    st.header("📊 Key Performance Indicators")

    total_customers = len(df)
    default_rate = df["Default_Flag"].mean() * 100
    avg_income = df["NETMONTHLYINCOME"].mean()
    avg_credit_score = df["Credit_Score"].mean()

    avg_enquiries = df["tot_enq"].mean()
    avg_delinquency = df["num_times_delinquent"].mean()
    dpd_30 = (df["num_times_30p_dpd"] > 0).mean() * 100
    dpd_60 = (df["num_times_60p_dpd"] > 0).mean() * 100

    approved = df[df["Approved_Flag"].isin(["P1", "P2"])].shape[0]
    rejected = df[df["Approved_Flag"].isin(["P3", "P4"])].shape[0]

    approval_rate = (approved / len(df)) * 100
    rejection_rate = (rejected / len(df)) * 100

    # KPI ROW 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Default Rate", f"{default_rate:.2f}%")
    col3.metric("Approval Rate", f"{approval_rate:.2f}%")
    col4.metric("Avg Credit Score", f"{avg_credit_score:.0f}")

    # KPI ROW 2
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Income", f"₹{avg_income:,.0f}")
    col6.metric("Avg Enquiries", f"{avg_enquiries:.2f}")
    col7.metric("DPD 30+ %", f"{dpd_30:.2f}%")
    col8.metric("DPD 60+ %", f"{dpd_60:.2f}%")

    st.markdown("---")

    st.header("📌 Key Insights")

    st.write("• Customers with **high delinquency** are most likely to default")
    st.write("• **Credit score below ~650** shows strong risk increase")
    st.write("• **High enquiry count** = aggressive credit behavior")
    st.write("• **DPD 60+ is a major red flag** for rejection")
    st.write("• Higher income improves approval probability, but not alone")
    st.write("• Employment stability indirectly reduces risk")

# =========================================================
# TAB 2 — DASHBOARD
# =========================================================
with tab2:

    st.header("📈 Credit Risk Dashboard")

    # =========================
    # SEGMENTS (CREATE IF NOT EXISTS)
    # =========================
    df["Income_Segment"] = pd.cut(
        df["NETMONTHLYINCOME"],
        bins=[0, 20000, 50000, 100000, 200000, np.inf],
        labels=["<20K", "20K-50K", "50K-1L", "1L-2L", ">2L"]
    )
    df["Score_Bin"] = pd.cut(df["Credit_Score"], bins=5)
    df["Enquiry_Segment"] = pd.cut(df["tot_enq"], bins=[-1,2,5,10,50],
                                  labels=["Low (0-2)", "Medium (3-5)", "High (6-10)", "Very High"])
    def dpd_category(row):
        if row["num_times_60p_dpd"] > 0:
            return "60+ DPD"
        elif row["num_times_30p_dpd"] > 0:
            return "30+ DPD"
        else:
            return "No DPD"

    df["DPD_Category"] = df.apply(dpd_category, axis=1)
    # =========================
    # FILTERS
    # =========================
    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        income_filter = st.multiselect(
            "Income Segment",
            df["Income_Segment"].unique(),
            default=df["Income_Segment"].unique()
        )

    with colf2:
        score_filter = st.multiselect(
            "Credit Score Bin",
            df["Score_Bin"].unique(),
            default=df["Score_Bin"].unique()
        )

    with colf3:
        enquiry_filter = st.multiselect(
            "Enquiry Segment",
            df["Enquiry_Segment"].unique(),
            default=df["Enquiry_Segment"].unique()
        )

    filtered_df = df[
        df["Income_Segment"].isin(income_filter) &
        df["Score_Bin"].isin(score_filter) &
        df["Enquiry_Segment"].isin(enquiry_filter)
    ]

    if filtered_df.empty:
        st.warning("No data for selected filters")
        st.stop()

    # =========================
    # ROW 1
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Customers by Credit Score Bin")

        data = filtered_df["Score_Bin"].value_counts().sort_index()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + (max(data.values)*0.01), f"{int(v)}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Customers by Credit Score Bin")
        ax.set_xlabel("Credit Score Bin")
        ax.set_ylabel("Total Customers")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("### 📌 Key Insight")
        top_score_bin = data.idxmax()
        st.write(f"• Majority of customers fall in **{top_score_bin} credit score range**")

    with col2:
        st.subheader("Approval vs Rejection Rate")

        fig, ax = plt.subplots()
        
        approved = filtered_df[filtered_df["Approved_Flag"].isin(["P1", "P2"])].shape[0]
        rejected = filtered_df[filtered_df["Approved_Flag"].isin(["P3", "P4"])].shape[0]
        
        values = [approved, rejected]
        labels = ["Approved", "Rejected"]

        if sum(values) > 0:
            ax.pie(
                values,
                labels=labels,
                autopct="%1.2f%%",
                startangle=90
            )

            # donut hole
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            fig.gca().add_artist(centre_circle)

            ax.set_title("Customer Approval vs Rejection Distribution")
            ax.axis("equal")

            st.pyplot(fig)
        else:
            st.warning("No data available")
        st.markdown("### 📌 Key Insight")
        if approved > rejected:
            st.write("• Majority of applications are **approved**, indicating strong portfolio quality")
        else:
            st.write("• Rejections are relatively high — indicates **tight credit policy or risky applicants**")
    # =========================
    # ROW 2
    # =========================
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Default Rate by Credit Score Bin")

        data = filtered_df.groupby("Score_Bin")["Default_Flag"].mean()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Default Rate by Credit Score")
        ax.set_xlabel("Credit Score Bin")
        ax.set_ylabel("Default Rate")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        high_default_score = data.idxmax()
        low_default_score = data.idxmin()
        st.markdown("### 📌 Key Insight")
        st.write(f"• Highest default rate observed in **{high_default_score} score segment**")
        st.write(f"• Lowest default risk in **{low_default_score} score segment**")

    with col4:
        st.subheader("Default Rate by Income Segment")

        data = filtered_df.groupby("Income_Segment")["Default_Flag"].mean()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Default Rate by Income")
        ax.set_xlabel("Income Segment")
        ax.set_ylabel("Default Rate")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        high_default_income = data.idxmax()
        low_default_income = data.idxmin()
        st.markdown("### 📌 Key Insight")
        st.write(f"• Highest default risk in **{high_default_income} income segment**")
        st.write(f"• Lowest default risk in **{low_default_income} income segment**")

    # =========================
    # ROW 3
    # =========================
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Total Customers by Income Segment")

        data = filtered_df["Income_Segment"].value_counts().sort_index()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + (max(data.values)*0.01), f"{int(v)}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Customers by Income Segment")
        ax.set_xlabel("Income Segment")
        ax.set_ylabel("Total Customers")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("### 📌 Key Insight")
        top_income_segment = data.idxmax()

        st.write(f"• Most customers belong to **{top_income_segment} income group**")

    with col6:
        st.subheader("Default Rate by Enquiry Segment")

        data = filtered_df.groupby("Enquiry_Segment")["Default_Flag"].mean()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Default Rate by Enquiries")
        ax.set_xlabel("Enquiry Segment")
        ax.set_ylabel("Default Rate")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("### 📌 Key Insight")
        high_enquiry_default = data.idxmax()

        st.write(f"• Higher enquiries (**{high_enquiry_default}**) show **increased default risk**")

    # =========================
    # ROW 4
    # =========================
    col7, col8 = st.columns(2)
    with col7:
        
        st.subheader("Total Customers by Enquiry Segment")

        data = filtered_df["Enquiry_Segment"].value_counts().sort_index()

        fig, ax = plt.subplots()
        data.plot(kind="bar", ax=ax)
        for i, v in enumerate(data.values):
            ax.text(i, v + (max(data.values)*0.01), f"{int(v)}", ha='center')

        ax.spines[['top','right']].set_visible(False)
        ax.set_title("Customers by Enquiry Segment")
        ax.set_xlabel("Enquiry Segment")
        ax.set_ylabel("Total Customers")
        ax.grid(axis='y')

        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("### 📌 Key Insight")
        top_enquiry_segment = data.idxmax()

        st.write(f"• Majority customers fall under **{top_enquiry_segment} enquiry category**")
    
    with col8:

        st.subheader("Customers by DPD Category")

        fig, ax = plt.subplots()

        dpd_counts = filtered_df["DPD_Category"].value_counts()

        # Ensure all categories exist
        no_dpd = dpd_counts.get("No DPD", 0)
        dpd_30 = dpd_counts.get("30+ DPD", 0)
        dpd_60 = dpd_counts.get("60+ DPD", 0)

        values = [no_dpd, dpd_30, dpd_60]
        labels = ["No DPD", "30+ DPD", "60+ DPD"]

        if sum(values) == 0:
            st.warning("No data available")
        else:
            ax.pie(
                values,
                labels=labels,
                autopct="%1.2f%%",
                startangle=90,
                wedgeprops=dict(width=0.4)
            )

            ax.axis("equal")
            ax.set_title("Customer Distribution by DPD Category")

            st.pyplot(fig)
            st.markdown("### 📌 Key Insight")
            if dpd_60 > dpd_30:
                st.write("• **60+ DPD customers dominate** → serious repayment issues")
            elif dpd_30 > 0:
                st.write("• Presence of **30+ DPD indicates early-stage risk buildup**")
            else:
                st.write("• Majority customers have **no delinquency (healthy portfolio)**")

# =========================================================
# TAB 3 — RISK PREDICTION (UNCHANGED)
# =========================================================
with tab3:

    st.title("💳 Credit Risk Prediction System")
    st.markdown("Enter applicant details:")

    def safe(x):
        try:
            return float(x)
        except:
            return None

    Credit_Score = st.text_input("Credit Score", placeholder="e.g. 650")
    NETMONTHLYINCOME = st.text_input("Monthly Income", placeholder="e.g. 40000")

    num_times_delinquent = st.text_input("Total Delinquencies", placeholder="e.g. 2")
    num_times_30p_dpd = st.text_input("30+ DPD Count", placeholder="e.g. 1")
    num_times_60p_dpd = st.text_input("60+ DPD Count", placeholder="e.g. 0")

    tot_enq = st.text_input("Total Enquiries", placeholder="e.g. 5")

    CC_TL = st.text_input("Credit Card Accounts", placeholder="e.g. 1")
    PL_TL = st.text_input("Personal Loan Accounts", placeholder="e.g. 1")
    Secured_TL = st.text_input("Secured Loans", placeholder="e.g. 0")
    Unsecured_TL = st.text_input("Unsecured Loans", placeholder="e.g. 1")

    Time_With_Curr_Empr = st.text_input("Employment Duration (months)", placeholder="e.g. 24")

    def risk_guardrails(data):

        credit, income, delinquent, dpd30, dpd60, enq, sec, unsec, emp = data

        if delinquent >= 10:
            return "REJECTED ❌ (High Delinquency)"

        if dpd60 >= 5:
            return "REJECTED ❌ (Severe Payment Delay)"

        if credit < 550:
            return "REJECTED ❌ (Low Credit Score)"

        if enq >= 15:
            return "MANUAL REVIEW ⚠️ (High Enquiries)"

        if income <= 0:
            return "REJECTED ❌ (Invalid Income)"

        if unsec > sec * 3 and unsec > 3:
            return "MANUAL REVIEW ⚠️ (High Unsecured Exposure)"

        if emp < 3:
            return "MANUAL REVIEW ⚠️ (Unstable Employment)"

        return None

    if st.button("Predict Risk"):

        Credit_Score = safe(Credit_Score)
        NETMONTHLYINCOME = safe(NETMONTHLYINCOME)
        num_times_delinquent = safe(num_times_delinquent)
        num_times_30p_dpd = safe(num_times_30p_dpd)
        num_times_60p_dpd = safe(num_times_60p_dpd)
        tot_enq = safe(tot_enq)
        CC_TL = safe(CC_TL)
        PL_TL = safe(PL_TL)
        Secured_TL = safe(Secured_TL)
        Unsecured_TL = safe(Unsecured_TL)
        Time_With_Curr_Empr = safe(Time_With_Curr_Empr)

        if None in [Credit_Score, NETMONTHLYINCOME, num_times_delinquent,
                    num_times_30p_dpd, num_times_60p_dpd, tot_enq,
                    CC_TL, PL_TL, Secured_TL, Unsecured_TL, Time_With_Curr_Empr]:

            st.error("⚠️ Please fill all fields correctly")
            st.stop()

        total_loans = Secured_TL + Unsecured_TL + 1

        Unsecured_Loan_Ratio = Unsecured_TL / total_loans
        log_unsec_exposure = np.log1p(Unsecured_TL)
        enquiry_to_income = tot_enq / (NETMONTHLYINCOME + 1)

        Employment_Stability = Time_With_Curr_Empr
        Age_Oldest_TL = Time_With_Curr_Empr * 2
        Age_Newest_TL = max(1, Time_With_Curr_Empr // 2)

        input_data = np.array([[Credit_Score, NETMONTHLYINCOME,
        num_times_delinquent, num_times_30p_dpd, num_times_60p_dpd,
        tot_enq, 0, 0, Unsecured_TL, Secured_TL, 0,
        CC_TL, PL_TL, Employment_Stability, 0,
        Age_Oldest_TL, Age_Newest_TL, 0, 0, 0, enquiry_to_income]])

        scaled = scaler.transform(input_data)
        risk_prob = model.predict_proba(scaled)[0][1]

        if risk_prob < 0.3:
            ml_risk = "LOW RISK"
        elif risk_prob < 0.6:
            ml_risk = "MEDIUM RISK"
        else:
            ml_risk = "HIGH RISK"

        rule_result = risk_guardrails([
            Credit_Score, NETMONTHLYINCOME, num_times_delinquent,
            num_times_30p_dpd, num_times_60p_dpd,
            tot_enq, Secured_TL, Unsecured_TL, Time_With_Curr_Empr
        ])

        if rule_result:
            final_decision = rule_result
            risk_level = "HIGH RISK ❌ (Rule Override)"
        else:
            risk_level = ml_risk
            final_decision = (
                "APPROVED ✅" if ml_risk == "LOW RISK"
                else "MANUAL REVIEW ⚠️" if ml_risk == "MEDIUM RISK"
                else "REJECTED ❌"
            )

        st.subheader("📊 Results")
        st.write("Risk Probability:", round(risk_prob, 3))
        st.write("Risk Level:", risk_level)
        st.write("Final Decision:", final_decision)