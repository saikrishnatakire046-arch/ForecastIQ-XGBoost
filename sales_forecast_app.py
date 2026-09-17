import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ForecastIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "sales_data.csv"
FORECAST_PATH = BASE_DIR / "new_overall_forecast.csv"
PRODUCT_FORECAST_PATH = BASE_DIR / "new_product_forecast.csv"
REGION_FORECAST_PATH = BASE_DIR / "new_region_forecast.csv"


# ============================================================
# CONSTANTS
# ============================================================

HISTORICAL_CUTOFF = pd.Timestamp("2026-09-10")

MODEL_NAME = "Final Trial 84 Log-XGBoost"

MODEL_FEATURES = [
    "day_of_week",
    "year",
    "month",
    "day",
    "quarter",
    "week_of_year",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "Price",
    "Discount_Percentage",
    "Revenue",
    "Promotion_Flag",
    "Stock_Availability",
    "Holiday_Flag",
    "Local_Event_Flag",
    "Competitor_Price",
    "Economic_Indicator",
    "Marketing_Spend",
    "price_competitor_ratio",
    "discount_price_interaction",
    "marketing_promotion_interaction",
    "revenue_per_unit",
    "price_bin",
    "marketing_spend_bin",
    "high_discount_flag"
]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_overall_forecast():

    forecast = pd.read_csv(FORECAST_PATH)

    if "Forecast_Date" in forecast.columns:
        forecast["Forecast_Date"] = pd.to_datetime(
            forecast["Forecast_Date"],
            errors="coerce"
        )

    if "Date" in forecast.columns:
        forecast["Date"] = pd.to_datetime(
            forecast["Date"],
            errors="coerce"
        )

    return forecast


@st.cache_data
def load_product_forecast():

    forecast = pd.read_csv(PRODUCT_FORECAST_PATH)

    if "Forecast_Date" in forecast.columns:
        forecast["Forecast_Date"] = pd.to_datetime(
            forecast["Forecast_Date"],
            errors="coerce"
        )

    return forecast


@st.cache_data
def load_region_forecast():

    forecast = pd.read_csv(REGION_FORECAST_PATH)

    if "Forecast_Date" in forecast.columns:
        forecast["Forecast_Date"] = pd.to_datetime(
            forecast["Forecast_Date"],
            errors="coerce"
        )

    return forecast


# ============================================================
# LOAD FILES
# ============================================================

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load sales_data.csv: {e}")
    st.stop()


try:
    forecast_df = load_overall_forecast()
except Exception as e:
    st.error(
        f"Unable to load new_overall_forecast.csv: {e}"
    )
    st.stop()


try:
    product_forecast_df = load_product_forecast()
except Exception as e:
    st.warning(
        f"Unable to load new_product_forecast.csv: {e}"
    )
    product_forecast_df = pd.DataFrame()


try:
    region_forecast_df = load_region_forecast()
except Exception as e:
    st.warning(
        f"Unable to load new_region_forecast.csv: {e}"
    )
    region_forecast_df = pd.DataFrame()


def standardize_forecast_dates(data):

    data = data.copy()

    if "Forecast_Date" in data.columns:

        data["Forecast_Date"] = pd.to_datetime(
            data["Forecast_Date"],
            errors="coerce"
        )

        if "Date" not in data.columns:
            data["Date"] = data["Forecast_Date"]

    elif "Date" in data.columns:

        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

        data["Forecast_Date"] = data["Date"]

    return data


forecast_df = standardize_forecast_dates(
    forecast_df
)


# ============================================================
# HISTORICAL DATA
# ============================================================

if "Date" not in df.columns:
    st.error("sales_data.csv must contain a Date column.")
    st.stop()

historical_df = df[
    df["Date"] <= HISTORICAL_CUTOFF
].copy()

historical_df = historical_df.sort_values(
    "Date"
)


# ============================================================
# BASIC CLEANING
# ============================================================

for column in [
    "Units_Sold",
    "Revenue",
    "Price",
    "Discount_Percentage",
    "Competitor_Price",
    "Marketing_Spend"
]:

    if column in historical_df.columns:

        historical_df[column] = pd.to_numeric(
            historical_df[column],
            errors="coerce"
        ).fillna(0)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_sum(data, column):

    if column not in data.columns:
        return 0

    return pd.to_numeric(
        data[column],
        errors="coerce"
    ).fillna(0).sum()


def safe_mean(data, column):

    if column not in data.columns or data.empty:
        return 0

    return pd.to_numeric(
        data[column],
        errors="coerce"
    ).mean()


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def money(value):
    return f"₹{value:,.0f}"


def number(value):
    return f"{value:,.0f}"


def download_csv(data, filename):

    st.download_button(
        label="⬇️ Download CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


def download_excel(data, filename):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        data.to_excel(
            writer,
            index=False,
            sheet_name="Forecast"
        )

    st.download_button(
        label="⬇️ Download Excel",
        data=output.getvalue(),
        file_name=filename,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )


def daily_sales(data):

    if data.empty:
        return pd.DataFrame()

    return (
        data.groupby("Date", as_index=False)
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .sort_values("Date")
    )


def weekly_sales(data):

    if data.empty:
        return pd.DataFrame()

    temp = data.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    return (
        temp.set_index("Date")
        .resample("W")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .reset_index()
    )


def group_units(data, column):

    if column not in data.columns:
        return pd.DataFrame()

    return (
        data.groupby(column, dropna=False)["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def prepare_forecast(
    data,
    start_date,
    horizon
):

    temp = data.copy()

    if "Forecast_Date" not in temp.columns:
        return pd.DataFrame()

    temp["Forecast_Date"] = pd.to_datetime(
        temp["Forecast_Date"],
        errors="coerce"
    )

    end_date = (
        start_date
        + pd.Timedelta(days=int(horizon) - 1)
    )

    temp = temp[
        (temp["Forecast_Date"] >= start_date)
        & (temp["Forecast_Date"] <= end_date)
    ].copy()

    return temp.sort_values(
        "Forecast_Date"
    )


def estimate_price_elasticity(
    data,
    product=None
):

    temp = data.copy()

    if (
        product is not None
        and "Product_Name" in temp.columns
    ):

        subset = temp[
            temp["Product_Name"].astype(str)
            == str(product)
        ].copy()

        if len(subset) >= 20:
            temp = subset

    if not all(
        column in temp.columns
        for column in ["Price", "Units_Sold"]
    ):
        return -0.50

    temp = temp[
        ["Price", "Units_Sold"]
    ].dropna()

    temp = temp[
        (temp["Price"] > 0)
        & (temp["Units_Sold"] > 0)
    ]

    if len(temp) < 20:
        return -0.50

    try:

        x = np.log(
            temp["Price"].values
        )

        y = np.log(
            temp["Units_Sold"].values
        )

        elasticity = np.polyfit(
            x,
            y,
            1
        )[0]

        return float(
            np.clip(
                elasticity,
                -3.0,
                0.25
            )
        )

    except Exception:

        return -0.50


def scenario_multiplier(
    data,
    product,
    price,
    discount,
    promotion,
    stock,
    holiday,
    local_event,
    competitor_price,
    marketing_spend
):

    subset = data.copy()

    if (
        product is not None
        and "Product_Name" in subset.columns
    ):

        product_data = subset[
            subset["Product_Name"].astype(str)
            == str(product)
        ]

        if not product_data.empty:
            subset = product_data

    multiplier = 1.0

    elasticity = estimate_price_elasticity(
        data,
        product
    )

    if "Price" in subset.columns:

        historical_price = pd.to_numeric(
            subset["Price"],
            errors="coerce"
        ).median()

        if (
            pd.notna(historical_price)
            and historical_price > 0
            and price > 0
        ):

            multiplier *= (
                price / historical_price
            ) ** elasticity

    if (
        "Discount_Percentage" in subset.columns
        and "Units_Sold" in subset.columns
    ):

        discounted = subset.loc[
            subset["Discount_Percentage"] > 0,
            "Units_Sold"
        ].mean()

        normal = subset.loc[
            subset["Discount_Percentage"] <= 0,
            "Units_Sold"
        ].mean()

        if (
            pd.notna(discounted)
            and pd.notna(normal)
            and normal > 0
        ):

            discount_ratio = discounted / normal

            multiplier *= (
                1
                + (discount_ratio - 1)
                * min(discount / 100, 0.5)
            )

    if promotion == 1:
        multiplier *= 1.05

    multiplier *= np.clip(
        stock / 100,
        0.50,
        1.00
    )

    if holiday == 1:
        multiplier *= 1.08

    if local_event == 1:
        multiplier *= 1.05

    if (
        competitor_price > 0
        and price > 0
    ):

        price_gap = (
            competitor_price - price
        ) / competitor_price

        multiplier *= np.clip(
            1 + 0.35 * price_gap,
            0.85,
            1.20
        )

    if "Marketing_Spend" in subset.columns:

        historical_marketing = pd.to_numeric(
            subset["Marketing_Spend"],
            errors="coerce"
        ).median()

        if (
            pd.notna(historical_marketing)
            and historical_marketing > 0
            and marketing_spend > 0
        ):

            ratio = (
                marketing_spend
                / historical_marketing
            )

            multiplier *= np.clip(
                1 + 0.10 * (ratio - 1),
                0.85,
                1.25
            )

    return float(
        np.clip(
            multiplier,
            0.20,
            3.00
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

pages = [
    "📊 Executive Dashboard",
    "📦 Product-Based Forecast",
    "📍 Region-Based Forecast",
    "Location Intelligence",
    "Product Intelligence",
    "Quarter Intelligence",
    "Demand Drivers",
    "Pricing Intelligence",
    "Promotion Intelligence",
    "Category Intelligence",
    "Sales Channel Intelligence",
    "Customer Intelligence",
    "Weekly Sales Intelligence",
    "Forecast Intelligence",
    "Model Intelligence",
    "Cross-Analysis Explorer",
    "Demand Opportunity Finder",
    "Leaderboards",
    "Product × Location Finder",
    "Ask ForecastIQ",
    "Data Explorer",
    "New Prediction"
]

st.sidebar.title("📈 ForecastIQ")

st.sidebar.caption(
    "Daily Sales Forecasting Dashboard"
)

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.info(
    """
**Historical Data**

06-Jan-2023 → 10-Sep-2026

**Forecast Data**

11-Sep-2026 → 31-Dec-2027

**Model**

Final Trial 84 Log-XGBoost
"""
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "📊 Executive Dashboard":

    st.title("📊 Executive Dashboard")

    total_units = safe_sum(
        historical_df,
        "Units_Sold"
    )

    total_revenue = safe_sum(
        historical_df,
        "Revenue"
    )

    daily = daily_sales(
        historical_df
    )

    average_daily = (
        daily["Units_Sold"].mean()
        if not daily.empty
        else 0
    )

    forecast_units = safe_sum(
        forecast_df,
        "Predicted_Units_Sold"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Historical Units",
        number(total_units)
    )

    c2.metric(
        "Historical Revenue",
        money(total_revenue)
    )

    c3.metric(
        "Average Daily Sales",
        f"{average_daily:,.1f}"
    )

    c4.metric(
        "Forecast Units",
        number(forecast_units)
    )

    st.subheader("Historical Sales Trend")

    if not daily.empty:

        st.line_chart(
            daily.set_index("Date")[
                ["Units_Sold"]
            ]
        )

    st.subheader("Future Forecast")

    if (
        "Date" in forecast_df.columns
        and "Predicted_Units_Sold"
        in forecast_df.columns
    ):

        future = (
            forecast_df
            .groupby("Date")[
                "Predicted_Units_Sold"
            ]
            .sum()
        )

        st.line_chart(future)


# ============================================================
# PRODUCT FORECAST
# ============================================================

elif page == "📦 Product-Based Forecast":

    st.title("📦 Product-Based Forecast")

    if product_forecast_df.empty:

        st.error(
            "new_product_forecast.csv "
            "could not be loaded."
        )

        st.stop()

    required = [
        "Forecast_Date",
        "Predicted_Units_Sold"
    ]

    missing = [
        column
        for column in required
        if column not in product_forecast_df.columns
    ]

    if missing:

        st.error(
            f"Missing columns: {missing}"
        )

        st.write(
            "Columns found:",
            list(product_forecast_df.columns)
        )

        st.stop()

    products = sorted(
        historical_df["Product_Name"]
        .dropna()
        .astype(str)
        .unique()
    ) if "Product_Name" in historical_df.columns else []

    stores = sorted(
        historical_df["Store_Location"]
        .dropna()
        .astype(str)
        .unique()
    ) if "Store_Location" in historical_df.columns else []

    if not products or not stores:

        st.error(
            "Product or store data is unavailable."
        )

        st.stop()

    c1, c2, c3 = st.columns(3)

    with c1:

        selected_product = st.selectbox(
            "📦 Product",
            products
        )

    with c2:

        selected_store = st.selectbox(
            "🏪 Store",
            stores
        )

    with c3:

        current_date = st.date_input(
            "📅 Current Date",
            value=HISTORICAL_CUTOFF.date(),
            min_value=HISTORICAL_CUTOFF.date()
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        price = st.number_input(
            "💰 Price",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    with c2:

        discount = st.number_input(
            "🏷️ Discount %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    with c3:

        promotion = st.selectbox(
            "📢 Promotion",
            ["No", "Yes"]
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        stock = st.slider(
            "📦 Stock Availability %",
            0,
            100,
            100
        )

    with c2:

        holiday = st.selectbox(
            "🎉 Holiday",
            ["No", "Yes"]
        )

    with c3:

        local_event = st.selectbox(
            "📍 Local Event",
            ["No", "Yes"]
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        competitor_price = st.number_input(
            "💰 Competitor Price",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    with c2:

        horizon = st.number_input(
            "🔮 Forecast Horizon",
            min_value=1,
            max_value=477,
            value=30,
            step=1
        )

    generate = st.button(
        "🚀 Generate Forecast",
        type="primary",
        use_container_width=True
    )

    if generate:

        start_date = (
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        )

        result = prepare_forecast(
            product_forecast_df,
            start_date,
            int(horizon)
        )

        if result.empty:

            st.warning(
                "No forecast is available for "
                "the selected date and horizon."
            )

        else:

            result = result.copy()

            multiplier = scenario_multiplier(
                historical_df,
                selected_product,
                price,
                discount,
                1 if promotion == "Yes" else 0,
                stock,
                1 if holiday == "Yes" else 0,
                1 if local_event == "Yes" else 0,
                competitor_price,
                safe_mean(
                    historical_df,
                    "Marketing_Spend"
                )
            )

            result["Predicted_Units_Sold"] = (
                pd.to_numeric(
                    result["Predicted_Units_Sold"],
                    errors="coerce"
                )
                .fillna(0)
                * multiplier
            )

            result["Predicted_Units_Sold"] = (
                result["Predicted_Units_Sold"]
                .clip(lower=0)
                .round()
                .astype(int)
            )

            result = result[
                [
                    "Forecast_Date",
                    "Predicted_Units_Sold"
                ]
            ]

            st.success(
                "Product forecast generated successfully."
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            download_csv(
                result,
                "product_forecast.csv"
            )

            download_excel(
                result,
                "product_forecast.xlsx"
            )


# ============================================================
# REGION FORECAST
# ============================================================

elif page == "📍 Region-Based Forecast":

    st.title("📍 Region-Based Forecast")

    if region_forecast_df.empty:

        st.error(
            "new_region_forecast.csv is not available."
        )

        st.stop()

    required = [
        "Forecast_Date",
        "Predicted_Units_Sold"
    ]

    missing = [
        column
        for column in required
        if column not in region_forecast_df.columns
    ]

    if missing:

        st.error(
            f"Missing columns: {missing}"
        )

        st.stop()

    c1, c2 = st.columns(2)

    with c1:

        current_date = st.date_input(
            "📅 Current Date",
            value=HISTORICAL_CUTOFF.date(),
            min_value=HISTORICAL_CUTOFF.date()
        )

    with c2:

        horizon = st.number_input(
            "🔮 Forecast Horizon",
            min_value=1,
            max_value=477,
            value=30,
            step=1
        )

    if st.button(
        "🚀 Generate Forecast",
        type="primary",
        use_container_width=True
    ):

        start_date = (
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        )

        result = prepare_forecast(
            region_forecast_df,
            start_date,
            int(horizon)
        )

        if result.empty:

            st.warning(
                "No region forecast is available."
            )

        else:

            st.success(
                "Region forecast generated successfully."
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            download_csv(
                result,
                "region_forecast.csv"
            )

            download_excel(
                result,
                "region_forecast.xlsx"
            )


# ============================================================
# INTELLIGENCE PAGES
# ============================================================

elif page == "Location Intelligence":

    st.title("📍 Location Intelligence")

    if "Store_Location" in historical_df.columns:

        result = group_units(
            historical_df,
            "Store_Location"
        )

        st.bar_chart(
            result.set_index("Store_Location")
        )

        st.dataframe(
            result,
            use_container_width=True
        )

    else:

        st.warning(
            "Store_Location column is unavailable."
        )


elif page == "Product Intelligence":

    st.title("📦 Product Intelligence")

    if "Product_Name" in historical_df.columns:

        result = group_units(
            historical_df,
            "Product_Name"
        )

        st.bar_chart(
            result.set_index("Product_Name")
        )

        st.dataframe(
            result,
            use_container_width=True
        )

    else:

        st.warning(
            "Product_Name column is unavailable."
        )


elif page == "Quarter Intelligence":

    st.title("📅 Quarter Intelligence")

    temp = historical_df.copy()

    temp["Year"] = temp["Date"].dt.year

    temp["Quarter"] = (
        "Q"
        + temp["Date"].dt.quarter.astype(str)
    )

    result = (
        temp.groupby(
            ["Year", "Quarter"]
        )["Units_Sold"]
        .sum()
        .reset_index()
    )

    result["Period"] = (
        result["Year"].astype(str)
        + "-"
        + result["Quarter"]
    )

    st.bar_chart(
        result.set_index("Period")
    )

    st.dataframe(
        result,
        use_container_width=True
    )


elif page == "Demand Drivers":

    st.title("📈 Demand Drivers")

    columns = [
        "Price",
        "Discount_Percentage",
        "Revenue",
        "Stock_Availability",
        "Competitor_Price",
        "Economic_Indicator",
        "Marketing_Spend",
        "Units_Sold"
    ]

    available = [
        column
        for column in columns
        if column in historical_df.columns
    ]

    correlation = historical_df[
        available
    ].corr(numeric_only=True)

    if "Units_Sold" in correlation.columns:

        result = (
            correlation["Units_Sold"]
            .sort_values(ascending=False)
            .to_frame("Correlation")
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.bar_chart(result)


elif page == "Pricing Intelligence":

    st.title("💰 Pricing Intelligence")

    if (
        "Price" in historical_df.columns
        and "Units_Sold" in historical_df.columns
    ):

        st.scatter_chart(
            historical_df[
                ["Price", "Units_Sold"]
            ].set_index("Price")
        )

        st.metric(
            "Average Price",
            money(
                safe_mean(
                    historical_df,
                    "Price"
                )
            )
        )


elif page == "Promotion Intelligence":

    st.title("🎯 Promotion Intelligence")

    if "Promotion_Flag" in historical_df.columns:

        result = (
            historical_df
            .groupby("Promotion_Flag")[
                "Units_Sold"
            ]
            .agg(
                ["sum", "mean", "count"]
            )
            .reset_index()
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.bar_chart(
            result.set_index("Promotion_Flag")
        )


elif page == "Category Intelligence":

    st.title("🏷️ Category Intelligence")

    if "Category" in historical_df.columns:

        result = group_units(
            historical_df,
            "Category"
        )

        st.bar_chart(
            result.set_index("Category")
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Sales Channel Intelligence":

    st.title("🛒 Sales Channel Intelligence")

    if "Sales_Channel" in historical_df.columns:

        result = group_units(
            historical_df,
            "Sales_Channel"
        )

        st.bar_chart(
            result.set_index("Sales_Channel")
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Customer Intelligence":

    st.title("👥 Customer Intelligence")

    if "Customer_Segment" in historical_df.columns:

        result = group_units(
            historical_df,
            "Customer_Segment"
        )

        st.bar_chart(
            result.set_index("Customer_Segment")
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Weekly Sales Intelligence":

    st.title("📆 Weekly Sales Intelligence")

    result = weekly_sales(
        historical_df
    )

    if not result.empty:

        st.line_chart(
            result.set_index("Date")[
                ["Units_Sold"]
            ]
        )

        st.line_chart(
            result.set_index("Date")[
                ["Revenue"]
            ]
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Forecast Intelligence":

    st.title("🔮 Forecast Intelligence")

    result = forecast_df.copy()

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce"
    )

    result["Predicted_Units_Sold"] = (
        pd.to_numeric(
            result["Predicted_Units_Sold"],
            errors="coerce"
        )
        .fillna(0)
        .round()
        .astype(int)
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Forecast",
        number(
            result["Predicted_Units_Sold"].sum()
        )
    )

    c2.metric(
        "Average Daily",
        f"{result['Predicted_Units_Sold'].mean():,.1f}"
    )

    c3.metric(
        "Peak Daily",
        number(
            result["Predicted_Units_Sold"].max()
        )
    )

    st.line_chart(
        result.groupby("Date")[
            "Predicted_Units_Sold"
        ].sum()
    )

    st.dataframe(
        result,
        use_container_width=True
    )


elif page == "Model Intelligence":

    st.title("🤖 Model Intelligence")

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        "**Algorithm:** XGBoost Regressor"
    )

    st.write(
        "**Transformation:** log1p → expm1"
    )

    st.write(
        f"**Total Features:** {len(MODEL_FEATURES)}"
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Feature": MODEL_FEATURES
            }
        ),
        use_container_width=True,
        hide_index=True
    )


elif page == "Cross-Analysis Explorer":

    st.title("🔍 Cross-Analysis Explorer")

    dimensions = [
        column
        for column in [
            "Product_Name",
            "Category",
            "Store_Location",
            "Sales_Channel",
            "Customer_Segment",
            "Season",
            "Holiday_Name"
        ]
        if column in historical_df.columns
    ]

    if dimensions:

        selected = st.selectbox(
            "Select Dimension",
            dimensions
        )

        metric = st.selectbox(
            "Select Metric",
            [
                "Units_Sold",
                "Revenue"
            ]
        )

        result = (
            historical_df
            .groupby(selected)[metric]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.bar_chart(
            result.set_index(selected)
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Demand Opportunity Finder":

    st.title("💡 Demand Opportunity Finder")

    if "Product_Name" in historical_df.columns:

        result = (
            historical_df
            .groupby("Product_Name")
            .agg(
                Units_Sold=("Units_Sold", "sum"),
                Revenue=("Revenue", "sum")
            )
            .reset_index()
        )

        result["Revenue_per_Unit"] = (
            result["Revenue"]
            / result["Units_Sold"].replace(0, np.nan)
        )

        result = result.sort_values(
            "Units_Sold",
            ascending=False
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.bar_chart(
            result.set_index("Product_Name")[
                ["Units_Sold"]
            ]
        )


elif page == "Leaderboards":

    st.title("🏆 Leaderboards")

    for column, title in [
        ("Product_Name", "Top Products"),
        ("Store_Location", "Top Locations"),
        ("Category", "Top Categories")
    ]:

        if column in historical_df.columns:

            st.subheader(title)

            st.dataframe(
                group_units(
                    historical_df,
                    column
                ).head(10),
                use_container_width=True
            )


elif page == "Product × Location Finder":

    st.title("📦 × 📍 Product × Location Finder")

    if (
        "Product_Name" in historical_df.columns
        and "Store_Location" in historical_df.columns
    ):

        products = sorted(
            historical_df["Product_Name"]
            .dropna()
            .unique()
        )

        locations = sorted(
            historical_df["Store_Location"]
            .dropna()
            .unique()
        )

        c1, c2 = st.columns(2)

        with c1:

            selected_product = st.selectbox(
                "Product",
                ["All"] + products
            )

        with c2:

            selected_location = st.selectbox(
                "Location",
                ["All"] + locations
            )

        result = historical_df.copy()

        if selected_product != "All":

            result = result[
                result["Product_Name"]
                == selected_product
            ]

        if selected_location != "All":

            result = result[
                result["Store_Location"]
                == selected_location
            ]

        c1, c2 = st.columns(2)

        c1.metric(
            "Units Sold",
            number(
                safe_sum(
                    result,
                    "Units_Sold"
                )
            )
        )

        c2.metric(
            "Revenue",
            money(
                safe_sum(
                    result,
                    "Revenue"
                )
            )
        )

        st.dataframe(
            result,
            use_container_width=True
        )


elif page == "Ask ForecastIQ":

    st.title("💬 Ask ForecastIQ")

    question = st.text_input(
        "Ask a question about the sales data"
    )

    if question:

        query = question.lower()

        if "top product" in query:

            result = (
                historical_df
                .groupby("Product_Name")[
                    "Units_Sold"
                ]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        elif "forecast" in query:

            st.success(
                f"Total forecast: "
                f"{safe_sum(forecast_df, 'Predicted_Units_Sold'):,.0f}"
            )

        elif "revenue" in query:

            st.success(
                f"Historical revenue: "
                f"{safe_sum(historical_df, 'Revenue'):,.0f}"
            )

        else:

            st.info(
                """
Try:

- What are the top products?
- What is the forecast?
- What is the total revenue?
"""
            )


elif page == "Data Explorer":

    st.title("🗃️ Data Explorer")

    st.write(
        f"Rows: {historical_df.shape[0]:,}"
    )

    st.write(
        f"Columns: {historical_df.shape[1]:,}"
    )

    st.dataframe(
        historical_df,
        use_container_width=True,
        height=500
    )


# ============================================================
# NEW PREDICTION
# ============================================================

elif page == "New Prediction":

    st.title("🚀 New Prediction")

    products = sorted(
        historical_df["Product_Name"]
        .dropna()
        .astype(str)
        .unique()
    ) if "Product_Name" in historical_df.columns else []

    regions = sorted(
        historical_df["Store_Location"]
        .dropna()
        .astype(str)
        .unique()
    ) if "Store_Location" in historical_df.columns else []

    c1, c2, c3 = st.columns(3)

    with c1:

        selected_product = st.selectbox(
            "Product",
            ["All Products"] + products
        )

    with c2:

        selected_region = st.selectbox(
            "Store / Region",
            ["All Regions"] + regions
        )

    with c3:

        current_date = st.date_input(
            "Current Date",
            value=HISTORICAL_CUTOFF.date(),
            min_value=HISTORICAL_CUTOFF.date()
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        horizon = st.number_input(
            "Forecast Horizon",
            min_value=1,
            max_value=477,
            value=30,
            step=1
        )

    with c2:

        price = st.number_input(
            "Price",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    with c3:

        discount = st.number_input(
            "Discount %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        promotion = st.selectbox(
            "Promotion",
            ["No", "Yes"]
        )

    with c2:

        stock = st.slider(
            "Stock Availability %",
            0,
            100,
            100
        )

    with c3:

        holiday = st.selectbox(
            "Holiday",
            ["No", "Yes"]
        )

    c1, c2 = st.columns(2)

    with c1:

        local_event = st.selectbox(
            "Local Event",
            ["No", "Yes"]
        )

    with c2:

        competitor_price = st.number_input(
            "Competitor Price",
            min_value=0.01,
            value=100.0,
            step=1.0
        )

    marketing_spend = st.number_input(
        "Marketing Spend",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    if st.button(
        "🚀 Generate New Prediction",
        type="primary",
        use_container_width=True
    ):

        start_date = (
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        )

        result = prepare_forecast(
            forecast_df,
            start_date,
            int(horizon)
        )

        if result.empty:

            st.error(
                "No baseline forecast is available."
            )

        else:

            result = result.copy()

            multiplier = scenario_multiplier(
                historical_df,
                (
                    None
                    if selected_product == "All Products"
                    else selected_product
                ),
                price,
                discount,
                1 if promotion == "Yes" else 0,
                stock,
                1 if holiday == "Yes" else 0,
                1 if local_event == "Yes" else 0,
                competitor_price,
                marketing_spend
            )

            result["Baseline_Forecast"] = (
                pd.to_numeric(
                    result["Predicted_Units_Sold"],
                    errors="coerce"
                ).fillna(0)
            )

            result["Scenario_Multiplier"] = multiplier

            result["Predicted_Units_Sold"] = (
                result["Baseline_Forecast"]
                * multiplier
            )

            result["Predicted_Units_Sold"] = (
                result["Predicted_Units_Sold"]
                .clip(lower=0)
                .round()
                .astype(int)
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Baseline Units",
                number(
                    result["Baseline_Forecast"].sum()
                )
            )

            c2.metric(
                "Scenario Units",
                number(
                    result["Predicted_Units_Sold"].sum()
                )
            )

            c3.metric(
                "Scenario Multiplier",
                f"{multiplier:.3f}x"
            )

            st.success(
                "New prediction generated successfully."
            )

            chart = result[
                [
                    "Forecast_Date",
                    "Baseline_Forecast",
                    "Predicted_Units_Sold"
                ]
            ].copy()

            chart = chart.set_index(
                "Forecast_Date"
            )

            st.line_chart(
                chart
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            download_csv(
                result,
                "new_prediction.csv"
            )

            download_excel(
                result,
                "new_prediction.xlsx"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ForecastIQ • Daily Sales Forecasting using "
    "Trial 84 Log-XGBoost | "
    "Historical: 06-Jan-2023 to 10-Sep-2026 | "
    "Forecast: 11-Sep-2026 to 31-Dec-2027"
)
