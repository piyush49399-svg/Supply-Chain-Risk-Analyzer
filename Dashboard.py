
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Supply Chain Risk Analyzer",
    page_icon="📦",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "data" / "DataCoSupplyChainDataset.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(FILE_PATH, encoding="latin1")
    return data

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.title("📦 Supply Chain Delivery Risk Analyzer")
st.caption("Analyze delivery risk, shipping performance, and orders requiring attention.")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filter Orders")

shipping_modes = sorted(df["Shipping Mode"].dropna().unique())
selected_modes = st.sidebar.multiselect(
    "Shipping Mode",
    shipping_modes,
    default=shipping_modes
)

delivery_statuses = sorted(df["Delivery Status"].dropna().unique())
selected_statuses = st.sidebar.multiselect(
    "Delivery Status",
    delivery_statuses,
    default=delivery_statuses
)


# Calculate actual delivery delay in days
df["Delivery Delay (Days)"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# Classify delivery performance
df["Delay Category"] = df["Delivery Delay (Days)"].apply(
    lambda days: (
        "Late" if days > 0
        else "On Time" if days == 0
        else "Early"
    )
)


filtered_df = df[
    df["Shipping Mode"].isin(selected_modes)
    & df["Delivery Status"].isin(selected_statuses)
].copy()

# ---------------- KPI METRICS ----------------
st.subheader("Key Performance Indicators")

total_orders = len(filtered_df)
risk_rate = (
    filtered_df["Late_delivery_risk"].mean() * 100
    if total_orders > 0 else 0
)
late_orders = (
    (filtered_df["Delivery Status"] == "Late delivery").sum()
)
on_time_orders = (
    (filtered_df["Delivery Status"] == "Shipping on time").sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Orders", f"{total_orders:,}")
c2.metric("Risk Label Rate", f"{risk_rate:.1f}%")
c3.metric("Late Delivery Status", f"{late_orders:,}")
c4.metric("Shipping On Time", f"{on_time_orders:,}")

if total_orders == 0:
    st.warning("No orders match the selected filters.")
    st.stop()

st.caption(
    "Risk Label Rate is based on the dataset's existing "
    "Late_delivery_risk column; it is not a model prediction."
)

# ---------------- CHARTS ----------------
st.subheader("Shipping Performance")
# compare delivery risk across shipping modes and delivery status distribution
left, right = st.columns(2)

with left:
    shipping_risk = (
        filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .mul(100)
        .reset_index(name="Risk Label Rate (%)")
    )

    fig1 = px.bar(
        shipping_risk,
        x="Shipping Mode",
        y="Risk Label Rate (%)",
        color="Shipping Mode",
        title="Risk Label Rate by Shipping Mode",
        text_auto=".1f"
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with right:
    status_counts = (
        filtered_df["Delivery Status"]
        .value_counts()
        .rename_axis("Delivery Status")
        .reset_index(name="Orders")
    )

    fig2 = px.pie(
        status_counts,
        names="Delivery Status",
        values="Orders",
        title="Delivery Status Distribution",
        hole=0.4
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------- DELIVERY DELAY ANALYSIS ----------------
st.subheader("Actual vs Scheduled Delivery")

delay_summary = (
    filtered_df.groupby("Delay Category")
    .size()
    .reset_index(name="Number of Orders")
)

fig3 = px.bar(
    delay_summary,
    x="Delay Category",
    y="Number of Orders",
    color="Delay Category",
    title="Orders Delivered Early, On Time, or Late",
    text_auto=True,
    category_orders={
        "Delay Category": ["Early", "On Time", "Late"]
    }
)

st.plotly_chart(fig3, use_container_width=True)

average_delay = filtered_df["Delivery Delay (Days)"].mean()

st.metric(
    "Average Delay (Days)",
    f"{average_delay:.2f}"
)


# ---------------- ORDER TABLE ----------------
st.subheader("Orders Requiring Attention")

attention_df = filtered_df[
    (filtered_df["Late_delivery_risk"] == 1)
    | (filtered_df["Delivery Status"] == "Late delivery")
].copy()

preferred_columns = [
    "Order Id",
    "Customer City",
    "Shipping Mode",
    "Delivery Status",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk"
]

available_columns = [
    col for col in preferred_columns
    if col in attention_df.columns
]

st.write(f"Orders flagged for attention: {len(attention_df):,}")

st.dataframe(
    attention_df[available_columns].head(100),
    use_container_width=True,
    hide_index=True
)

# ---------------- DOWNLOAD ----------------
csv_data = attention_df[available_columns].to_csv(index=False)

st.download_button(
    label="Download flagged orders as CSV",
    data=csv_data,
    file_name="orders_requiring_attention.csv",
    mime="text/csv"
)

st.caption(
    "Note: This dashboard uses historical dataset labels and statuses. "
    "It does not yet predict future delivery delays."
)

# ==========================================
# EARLY-RISK MODEL PREDICTION
# ==========================================


st.divider()
st.header("Early Delivery Risk Estimate")

st.caption(
    "Experimental model estimate based on selected order details. "
    "This is not a guaranteed prediction of future delivery."
)

MODEL_PATH = BASE_DIR / "models" / "early_risk_model.joblib"


if MODEL_PATH.exists():
    try:
        risk_model = joblib.load(MODEL_PATH)

        with st.form("early_risk_form"):
            col1, col2 = st.columns(2)

            with col1:
                price = st.number_input(
                    "Product price",
                    min_value=0.0,
                    value=50.0
                )
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1
                )
                discount = st.number_input(
                    "Discount amount",
                    min_value=0.0,
                    value=0.0
                )
                discount_rate = st.number_input(
                    "Discount rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )

            with col2:
                payment_type = st.selectbox(
                    "Payment type",
                    ["DEBIT", "TRANSFER", "CASH", "PAYMENT"]
                )
                segment = st.selectbox(
                    "Customer segment",
                    ["Consumer", "Corporate", "Home Office"]
                )
                market = st.selectbox(
                    "Market",
                    ["USCA", "LATAM", "Europe", "Pacific Asia",
                     "Africa"]
                )
                region = st.selectbox(
                    "Order region",
                    ["West of USA", "East of USA", "Central America",
                     "Western Europe", "Northern Europe"]
                )

                shipping_mode = st.selectbox(
                    "Shipping mode",
                    ["Standard Class", "First Class",
                     "Second Class", "Same Day"]
                )

                department = st.text_input(
                    "Department name",
                    value="Fan Shop"
                )
                category = st.text_input(
                     "Category Name",
                    value="Cleats"
)
            submitted = st.form_submit_button(
                "Estimate Delivery Risk"
            )

        if submitted:
            input_data = pd.DataFrame([{
                "Order Item Product Price": price,
                "Order Item Quantity": quantity,
                "Order Item Discount": discount,
                "Order Item Discount Rate": discount_rate,
                "Type": payment_type,
                "Customer Segment": segment,
                "Market": market,
                "Order Region": region,
                "Shipping Mode": shipping_mode,
                "Department Name": department,
                "Category Name": category
            }])

            prediction = risk_model.predict(input_data)[0]
            probability = risk_model.predict_proba(input_data)[0][1]

            st.metric(
                "Estimated risk score",
                f"{probability:.1%}"
            )

            if prediction == 1:
                st.warning(
                    "The model classified this order as higher risk."
                )
            else:
                st.success(
                    "The model classified this order as lower risk."
                )

            st.caption(
                "Experimental estimate only. Validate on genuinely "
                "future orders before operational use."
            )

    except Exception as error:
        st.error(f"Could not load or use the model: {error}")

else:
    st.error(f"Model file not found: {MODEL_PATH}")
