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

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

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

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    if "Date" in forecast.columns:

        forecast["Date"] = pd.to_datetime(
            forecast["Date"],
            errors="coerce"
        )

    if "Predicted_Units_Sold" in forecast.columns:

        forecast["Predicted_Units_Sold"] = (
            pd.to_numeric(
                forecast["Predicted_Units_Sold"],
                errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
            .round()
            .astype(int)
        )

    return forecast


@st.cache_data
def load_product_forecast():

    forecast = pd.read_csv(
        PRODUCT_FORECAST_PATH
    )

    if "Date" in forecast.columns:

        forecast["Date"] = pd.to_datetime(
            forecast["Date"],
            errors="coerce"
        )

    if "Predicted_Units_Sold" in forecast.columns:

        forecast["Predicted_Units_Sold"] = (
            pd.to_numeric(
                forecast["Predicted_Units_Sold"],
                errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
            .round()
            .astype(int)
        )

    return forecast


@st.cache_data
def load_region_forecast():

    forecast = pd.read_csv(
        REGION_FORECAST_PATH
    )

    if "Date" in forecast.columns:

        forecast["Date"] = pd.to_datetime(
            forecast["Date"],
            errors="coerce"
        )

    if "Predicted_Units_Sold" in forecast.columns:

        forecast["Predicted_Units_Sold"] = (
            pd.to_numeric(
                forecast["Predicted_Units_Sold"],
                errors="coerce"
            )
            .fillna(0)
            .clip(lower=0)
            .round()
            .astype(int)
        )

    return forecast


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load sales_data.csv: {e}"
    )

    st.stop()


try:

    forecast_df = load_overall_forecast()

except Exception as e:

    st.error(
        "Unable to load new_overall_forecast.csv. "
        f"Error: {e}"
    )

    st.stop()


# ------------------------------------------------------------
# PRODUCT FORECAST
# ------------------------------------------------------------

try:

    product_forecast_df = load_product_forecast()

except Exception as e:

    product_forecast_df = pd.DataFrame()

    st.warning(
        "Unable to load "
        "new_product_forecast.csv "
        f"Error: {e}"
    )


# ------------------------------------------------------------
# REGION FORECAST
# ------------------------------------------------------------

try:

    region_forecast_df = load_region_forecast()

except Exception as e:

    region_forecast_df = pd.DataFrame()

    st.warning(
        "Unable to load "
        "new_region_forecast.csv "
        f"Error: {e}"
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

historical_df = df[
    df["Date"] <= HISTORICAL_CUTOFF
].copy()

historical_df = historical_df.sort_values(
    "Date"
)


# ============================================================
# BASIC CLEANING
# ============================================================

if "Units_Sold" in historical_df.columns:

    historical_df["Units_Sold"] = pd.to_numeric(
        historical_df["Units_Sold"],
        errors="coerce"
    ).fillna(0)


if "Revenue" in historical_df.columns:

    historical_df["Revenue"] = pd.to_numeric(
        historical_df["Revenue"],
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

    if column not in data.columns or len(data) == 0:

        return 0

    return pd.to_numeric(
        data[column],
        errors="coerce"
    ).mean()


def weekly_sales(data):

    if data.empty:

        return pd.DataFrame()

    temp = data.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    result = (
        temp.set_index("Date")
        .resample("W")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .reset_index()
    )

    return result


def daily_sales(data):

    if data.empty:

        return pd.DataFrame()

    result = (
        data.groupby(
            "Date",
            as_index=False
        )
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .sort_values("Date")
    )

    return result


def group_units(data, column):

    if column not in data.columns:

        return pd.DataFrame()

    return (
        data.groupby(
            column,
            dropna=False
        )["Units_Sold"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def group_revenue(data, column):

    if column not in data.columns:

        return pd.DataFrame()

    return (
        data.groupby(
            column,
            dropna=False
        )["Revenue"]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def top_group(
    data,
    group_columns,
    metric="Units_Sold"
):

    valid_columns = [
        c
        for c in group_columns
        if c in data.columns
    ]

    if (
        not valid_columns
        or metric not in data.columns
    ):

        return pd.DataFrame()

    return (
        data.groupby(
            valid_columns
        )[metric]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )


def download_csv(data, filename):

    csv_data = data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv_data,
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


def money(value):

    return f"₹{value:,.0f}"


def number(value):

    return f"{value:,.0f}"


def percentage(value):

    return f"{value:.2f}%"


def safe_float(value, default=0.0):

    try:

        value = float(value)

        if not np.isfinite(value):

            return default

        return value

    except Exception:

        return default


# ============================================================
# SCENARIO CALCULATION
# ============================================================

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
            temp["Product_Name"] == product
        ].copy()

        if len(subset) >= 20:

            temp = subset

    required = [
        "Price",
        "Units_Sold"
    ]

    if not all(
        c in temp.columns
        for c in required
    ):

        return -0.50

    temp = temp[
        ["Price", "Units_Sold"]
    ].dropna()

    temp = temp[
        (temp["Price"] > 0)
        &
        (temp["Units_Sold"] > 0)
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

        elasticity = float(
            np.clip(
                elasticity,
                -3.0,
                0.25
            )
        )

        return elasticity

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

        product_subset = subset[
            subset["Product_Name"] == product
        ]

        if len(product_subset) > 0:

            subset = product_subset

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

            price_ratio = (
                price / historical_price
            )

            price_effect = (
                price_ratio ** elasticity
            )

            multiplier *= price_effect

    if (
        "Discount_Percentage" in subset.columns
        and "Units_Sold" in subset.columns
    ):

        temp = subset.copy()

        temp["Discount_Percentage"] = pd.to_numeric(
            temp["Discount_Percentage"],
            errors="coerce"
        )

        temp["Units_Sold"] = pd.to_numeric(
            temp["Units_Sold"],
            errors="coerce"
        )

        discounted = temp[
            temp["Discount_Percentage"] > 0
        ]["Units_Sold"].mean()

        normal = temp[
            temp["Discount_Percentage"] <= 0
        ]["Units_Sold"].mean()

        if (
            pd.notna(discounted)
            and pd.notna(normal)
            and normal > 0
        ):

            discount_ratio = (
                discounted / normal
            )

            requested_discount = np.clip(
                discount / 100,
                0,
                1
            )

            discount_effect = (
                1
                + (discount_ratio - 1)
                * min(requested_discount, 0.5)
            )

            multiplier *= discount_effect

    if promotion == 1:

        if (
            "Promotion_Flag" in subset.columns
            and "Units_Sold" in subset.columns
        ):

            promo = subset[
                subset["Promotion_Flag"] == 1
            ]["Units_Sold"].mean()

            non_promo = subset[
                subset["Promotion_Flag"] == 0
            ]["Units_Sold"].mean()

            if (
                pd.notna(promo)
                and pd.notna(non_promo)
                and non_promo > 0
            ):

                multiplier *= np.clip(
                    promo / non_promo,
                    0.8,
                    1.5
                )

            else:

                multiplier *= 1.05

        else:

            multiplier *= 1.05

    stock_ratio = np.clip(
        stock / 100,
        0.50,
        1.00
    )

    multiplier *= stock_ratio

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

            marketing_ratio = (
                marketing_spend
                / historical_marketing
            )

            multiplier *= np.clip(
                1 + 0.10 * (
                    marketing_ratio - 1
                ),
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
# FORECAST FILTER HELPERS
# ============================================================

def prepare_overall_forecast(
    forecast_data,
    start_date,
    horizon
):

    temp = forecast_data.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    temp["Predicted_Units_Sold"] = (
        pd.to_numeric(
            temp["Predicted_Units_Sold"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )

    temp = temp[
        temp["Date"] >= start_date
    ].copy()

    temp = temp.sort_values(
        "Date"
    )

    end_date = (
        start_date
        + pd.Timedelta(
            days=horizon - 1
        )
    )

    temp = temp[
        temp["Date"] <= end_date
    ].copy()

    return temp


def prepare_product_forecast(
    forecast_data,
    start_date,
    horizon,
    product=None
):

    temp = forecast_data.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    temp["Predicted_Units_Sold"] = (
        pd.to_numeric(
            temp["Predicted_Units_Sold"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )

    temp = temp[
        temp["Date"] >= start_date
    ].copy()

    if (
        product
        and product != "All Products"
        and "Product_Name" in temp.columns
    ):

        temp = temp[
            temp["Product_Name"] == product
        ].copy()

    end_date = (
        start_date
        + pd.Timedelta(
            days=horizon - 1
        )
    )

    temp = temp[
        temp["Date"] <= end_date
    ].copy()

    return temp.sort_values(
        "Date"
    )


def prepare_region_forecast(
    forecast_data,
    start_date,
    horizon,
    region=None
):

    temp = forecast_data.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    temp["Predicted_Units_Sold"] = (
        pd.to_numeric(
            temp["Predicted_Units_Sold"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )

    temp = temp[
        temp["Date"] >= start_date
    ].copy()

    if (
        region
        and region != "All Regions"
        and "Store_Location" in temp.columns
    ):

        temp = temp[
            temp["Store_Location"] == region
        ].copy()

    end_date = (
        start_date
        + pd.Timedelta(
            days=horizon - 1
        )
    )

    temp = temp[
        temp["Date"] <= end_date
    ].copy()

    return temp.sort_values(
        "Date"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📈 ForecastIQ"
)

st.sidebar.caption(
    "Daily Sales Forecasting Dashboard"
)

st.sidebar.markdown("---")

pages = [
    "📦 Product-Based Forecast",
    "📍 Region-Based Forecast",
    "📊 Executive Dashboard",
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

page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"""
**Historical Data**

06-Jan-2023 → 10-Sep-2026

**Forecast Data**

11-Sep-2026 → 31-Dec-2027

**Model**

Final Trial 84 Log-XGBoost
"""
)


# ============================================================
# PAGE 1 — PRODUCT-BASED FORECAST
# ============================================================

if page == "📦 Product-Based Forecast":

    st.title(
        "📦 Product-Based Forecast"
    )

    st.caption(
        "View the product-level forecast generated by the "
        "Final Trial 84 Log-XGBoost model."
    )

    if product_forecast_df.empty:

        st.error(
            "new_product_forecast.csv "
            "could not be loaded."
        )

    else:

        required = [
            "Forecast_Date",
            "Predicted_Units_Sold"
        ]

        missing = [
            c
            for c in required
            if c not in product_forecast_df.columns
        ]

        if missing:

            st.error(
                f"Missing columns in product forecast CSV: {missing}"
            )

            st.write(
                "Columns found:",
                list(product_forecast_df.columns)
            )

        else:

            product_data = (
                product_forecast_df.copy()
            )

            product_data["Date"] = pd.to_datetime(
                product_data["Date"],
                errors="coerce"
            )

            product_data["Predicted_Units_Sold"] = (
                pd.to_numeric(
                    product_data[
                        "Predicted_Units_Sold"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .clip(lower=0)
                .round()
                .astype(int)
            )

            products = sorted(
                product_data[
                    "Product_Name"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not products:

                st.error(
                    "No Product_Name values were found "
                    "in the product forecast CSV."
                )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    selected_product = st.selectbox(
                        "📦 Product",
                        products,
                        key="product_forecast_product"
                    )

                with c2:

                    current_date = st.date_input(
                        "📅 Current Date",
                        value=HISTORICAL_CUTOFF.date(),
                        min_value=HISTORICAL_CUTOFF.date(),
                        max_value=pd.Timestamp(
                            "2027-12-31"
                        ).date(),
                        key="product_forecast_date"
                    )

                horizon = st.slider(
                    "🔮 Forecast Horizon (days)",
                    min_value=1,
                    max_value=477,
                    value=30,
                    step=1,
                    key="product_forecast_horizon"
                )

                st.info(
                    "The product forecast below is read directly "
                    "from new_product_forecast.csv. "
                    "No store allocation or redistribution is applied."
                )

                if st.button(
                    "🚀 Generate Forecast",
                    type="primary",
                    use_container_width=True,
                    key="product_forecast_generate"
                ):

                    start_date = (
                        pd.Timestamp(
                            current_date
                        )
                        + pd.Timedelta(days=1)
                    )

                    end_date = (
                        start_date
                        + pd.Timedelta(
                            days=int(horizon) - 1
                        )
                    )

                    result = product_data[
                        (
                            product_data[
                                "Product_Name"
                            ].astype(str)
                            == str(selected_product)
                        )
                        &
                        (
                            product_data[
                                "Date"
                            ] >= start_date
                        )
                        &
                        (
                            product_data[
                                "Date"
                            ] <= end_date
                        )
                    ].copy()

                    result = result.sort_values(
                        "Date"
                    )

                    if result.empty:

                        st.warning(
                            "No product forecast is available "
                            "for the selected product and date range."
                        )

                    else:

                        total_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].sum()
                        )

                        average_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].mean()
                        )

                        minimum_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].min()
                        )

                        maximum_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].max()
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric(
                            "Total Forecast",
                            number(total_prediction)
                        )

                        c2.metric(
                            "Average Daily",
                            f"{average_prediction:,.1f}"
                        )

                        c3.metric(
                            "Minimum Daily",
                            number(minimum_prediction)
                        )

                        c4.metric(
                            "Maximum Daily",
                            number(maximum_prediction)
                        )

                        st.success(
                            "Product forecast generated successfully."
                        )

                        st.subheader(
                            f"Forecast for {selected_product}"
                        )

                        chart_data = (
                            result[
                                [
                                    "Date",
                                    "Predicted_Units_Sold"
                                ]
                            ]
                            .set_index("Date")
                        )

                        st.line_chart(
                            chart_data,
                            height=400
                        )

                        st.subheader(
                            "Product Forecast Data"
                        )

                        st.dataframe(
                            result[
                                [
                                    "Date",
                                    "Product_ID",
                                    "Product_Name",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            use_container_width=True,
                            height=450,
                            hide_index=True
                        )

                        download_csv(
                            result[
                                [
                                    "Date",
                                    "Product_ID",
                                    "Product_Name",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            "product_forecast.csv"
                        )

                        download_excel(
                            result[
                                [
                                    "Date",
                                    "Product_ID",
                                    "Product_Name",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            "product_forecast.xlsx"
                        )


# ============================================================
# PAGE 2 — REGION-BASED FORECAST
# ============================================================

elif page == "📍 Region-Based Forecast":

    st.title(
        "📍 Region-Based Forecast"
    )

    st.caption(
        "View the region-level forecast generated by the "
        "Final Trial 84 Log-XGBoost model."
    )

    if region_forecast_df.empty:

        st.error(
            "new_region_forecast.csv "
            "could not be loaded."
        )

    else:

        required = [
            "Forecast_Date",
            Product_ID,
            Product_Name,
            "Predicted_Units_Sold"
        ]

        missing = [
            c
            for c in required
            if c not in region_forecast_df.columns
        ]

        if missing:

            st.error(
                f"Missing columns in region forecast CSV: {missing}"
            )

            st.write(
                "Columns found:",
                list(region_forecast_df.columns)
            )

        else:

            region_data = (
                region_forecast_df.copy()
            )

            region_data["Date"] = pd.to_datetime(
                region_data["Date"],
                errors="coerce"
            )

            region_data["Predicted_Units_Sold"] = (
                pd.to_numeric(
                    region_data[
                        "Predicted_Units_Sold"
                    ],
                    errors="coerce"
                )
                .fillna(0)
                .clip(lower=0)
                .round()
                .astype(int)
            )

            regions = sorted(
                region_data[
                    "Store_Location"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not regions:

                st.error(
                    "No Store_Location values were found "
                    "in the region forecast CSV."
                )

            else:

                c1, c2 = st.columns(2)

                with c1:

                    selected_region = st.selectbox(
                        "📍 Region / Location",
                        regions,
                        key="region_forecast_region"
                    )

                with c2:

                    current_date = st.date_input(
                        "📅 Current Date",
                        value=HISTORICAL_CUTOFF.date(),
                        min_value=HISTORICAL_CUTOFF.date(),
                        max_value=pd.Timestamp(
                            "2027-12-31"
                        ).date(),
                        key="region_forecast_date"
                    )

                horizon = st.slider(
                    "🔮 Forecast Horizon (days)",
                    min_value=1,
                    max_value=477,
                    value=30,
                    step=1,
                    key="region_forecast_horizon"
                )

                st.info(
                    "The region forecast below is read directly "
                    "from new_region_forecast.csv. "
                    "No product redistribution or allocation is applied."
                )

                if st.button(
                    "🚀 Generate Forecast",
                    type="primary",
                    use_container_width=True,
                    key="region_forecast_generate"
                ):

                    start_date = (
                        pd.Timestamp(
                            current_date
                        )
                        + pd.Timedelta(days=1)
                    )

                    end_date = (
                        start_date
                        + pd.Timedelta(
                            days=int(horizon) - 1
                        )
                    )

                    result = region_data[
                        (
                            region_data[
                                "Store_Location"
                            ].astype(str)
                            == str(selected_region)
                        )
                        &
                        (
                            region_data[
                                "Date"
                            ] >= start_date
                        )
                        &
                        (
                            region_data[
                                "Date"
                            ] <= end_date
                        )
                    ].copy()

                    result = result.sort_values(
                        "Date"
                    )

                    if result.empty:

                        st.warning(
                            "No regional forecast is available "
                            "for the selected region and date range."
                        )

                    else:

                        total_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].sum()
                        )

                        average_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].mean()
                        )

                        minimum_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].min()
                        )

                        maximum_prediction = (
                            result[
                                "Predicted_Units_Sold"
                            ].max()
                        )

                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric(
                            "Total Forecast",
                            number(total_prediction)
                        )

                        c2.metric(
                            "Average Daily",
                            f"{average_prediction:,.1f}"
                        )

                        c3.metric(
                            "Minimum Daily",
                            number(minimum_prediction)
                        )

                        c4.metric(
                            "Maximum Daily",
                            number(maximum_prediction)
                        )

                        st.success(
                            "Region forecast generated successfully."
                        )

                        st.subheader(
                            f"Forecast for {selected_region}"
                        )

                        chart_data = (
                            result[
                                [
                                    "Date",
                                    "Predicted_Units_Sold"
                                ]
                            ]
                            .set_index("Date")
                        )

                        st.line_chart(
                            chart_data,
                            height=400
                        )

                        st.subheader(
                            "Region Forecast Data"
                        )

                        st.dataframe(
                            result[
                                [
                                    "Date",
                                    "Store_Location",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            use_container_width=True,
                            height=450,
                            hide_index=True
                        )

                        download_csv(
                            result[
                                [
                                    "Date",
                                    "Store_Location",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            "region_forecast.csv"
                        )

                        download_excel(
                            result[
                                [
                                    "Date",
                                    "Store_Location",
                                    "Predicted_Units_Sold"
                                ]
                            ],
                            "region_forecast.xlsx"
                        )


# ============================================================
# PAGE 3 — EXECUTIVE DASHBOARD
# ============================================================

elif page == "📊 Executive Dashboard":

    st.title(
        "📊 Executive Dashboard"
    )

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

    avg_daily = (
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
        "Avg Daily Sales",
        f"{avg_daily:,.1f}"
    )

    c4.metric(
        "Forecast Units",
        number(forecast_units)
    )

    st.markdown("---")

    st.subheader(
        "Historical Sales Trend"
    )

    if not daily.empty:

        st.line_chart(
            daily[
                "Units_Sold"
            ]
        )

    st.subheader(
        "Future Forecast"
    )

    fc = forecast_df.copy()

    if (
        "Date" in fc.columns
        and "Predicted_Units_Sold" in fc.columns
    ):

        fc = (
            fc.groupby(
                "Date",
                as_index=True
            )[
                "Predicted_Units_Sold"
            ]
            .sum()
        )

        st.line_chart(
            fc
        )


# ============================================================
# PAGE 4 — LOCATION INTELLIGENCE
# ============================================================

elif page == "Location Intelligence":

    st.title(
        "📍 Location Intelligence"
    )

    if "Store_Location" not in historical_df.columns:

        st.warning(
            "Store_Location column is not available."
        )

    else:

        location_units = group_units(
            historical_df,
            "Store_Location"
        )

        st.subheader(
            "Units Sold by Location"
        )

        st.bar_chart(
            location_units.set_index(
                "Store_Location"
            )
        )

        st.dataframe(
            location_units,
            use_container_width=True
        )


# ============================================================
# PAGE 5 — PRODUCT INTELLIGENCE
# ============================================================

elif page == "Product Intelligence":

    st.title(
        "📦 Product Intelligence"
    )

    if "Product_Name" not in historical_df.columns:

        st.warning(
            "Product_Name column is not available."
        )

    else:

        product_units = group_units(
            historical_df,
            "Product_Name"
        )

        st.subheader(
            "Product Sales"
        )

        st.bar_chart(
            product_units.set_index(
                "Product_Name"
            )
        )

        st.dataframe(
            product_units,
            use_container_width=True
        )


# ============================================================
# PAGE 6 — QUARTER INTELLIGENCE
# ============================================================

elif page == "Quarter Intelligence":

    st.title(
        "📅 Quarter Intelligence"
    )

    temp = historical_df.copy()

    temp["Year"] = temp[
        "Date"
    ].dt.year

    temp["Quarter_Label"] = (
        "Q"
        + temp[
            "Date"
        ].dt.quarter.astype(str)
    )

    quarterly = (
        temp.groupby(
            ["Year", "Quarter_Label"]
        )["Units_Sold"]
        .sum()
        .reset_index()
    )

    quarterly["Period"] = (
        quarterly[
            "Year"
        ].astype(str)
        + "-"
        + quarterly[
            "Quarter_Label"
        ]
    )

    st.bar_chart(
        quarterly.set_index(
            "Period"
        )[["Units_Sold"]]
    )

    st.dataframe(
        quarterly,
        use_container_width=True
    )


# ============================================================
# PAGE 7 — DEMAND DRIVERS
# ============================================================

elif page == "Demand Drivers":

    st.title(
        "📈 Demand Drivers"
    )

    numeric_candidates = [
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
        c
        for c in numeric_candidates
        if c in historical_df.columns
    ]

    corr = historical_df[
        available
    ].corr(
        numeric_only=True
    )

    if "Units_Sold" in corr.columns:

        demand_corr = (
            corr[
                "Units_Sold"
            ]
            .sort_values(
                ascending=False
            )
        )

        st.subheader(
            "Correlation with Units Sold"
        )

        st.dataframe(
            demand_corr.to_frame(
                "Correlation"
            ),
            use_container_width=True
        )

        st.bar_chart(
            demand_corr.drop(
                "Units_Sold"
            )
        )


# ============================================================
# PAGE 8 — PRICING INTELLIGENCE
# ============================================================

elif page == "Pricing Intelligence":

    st.title(
        "💰 Pricing Intelligence"
    )

    if (
        "Price" not in historical_df.columns
        or "Units_Sold"
        not in historical_df.columns
    ):

        st.warning(
            "Price or Units_Sold column unavailable."
        )

    else:

        temp = historical_df[
            ["Price", "Units_Sold"]
        ].dropna()

        st.subheader(
            "Price vs Units Sold"
        )

        st.scatter_chart(
            temp,
            x="Price",
            y="Units_Sold"
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


# ============================================================
# PAGE 9 — PROMOTION INTELLIGENCE
# ============================================================

elif page == "Promotion Intelligence":

    st.title(
        "🎯 Promotion Intelligence"
    )

    if "Promotion_Flag" not in historical_df.columns:

        st.warning(
            "Promotion_Flag column is not available."
        )

    else:

        promotion = (
            historical_df
            .groupby(
                "Promotion_Flag"
            )["Units_Sold"]
            .agg(
                ["sum", "mean", "count"]
            )
            .reset_index()
        )

        promotion["Promotion"] = (
            promotion[
                "Promotion_Flag"
            ]
            .map({
                0: "No Promotion",
                1: "Promotion"
            })
            .fillna(
                promotion[
                    "Promotion_Flag"
                ].astype(str)
            )
        )

        st.bar_chart(
            promotion.set_index(
                "Promotion"
            )[["mean"]]
        )

        st.dataframe(
            promotion,
            use_container_width=True
        )


# ============================================================
# PAGE 10 — CATEGORY INTELLIGENCE
# ============================================================

elif page == "Category Intelligence":

    st.title(
        "🏷️ Category Intelligence"
    )

    if "Category" not in historical_df.columns:

        st.warning(
            "Category column is not available."
        )

    else:

        category = group_units(
            historical_df,
            "Category"
        )

        st.bar_chart(
            category.set_index(
                "Category"
            )
        )

        st.dataframe(
            category,
            use_container_width=True
        )


# ============================================================
# PAGE 11 — SALES CHANNEL INTELLIGENCE
# ============================================================

elif page == "Sales Channel Intelligence":

    st.title(
        "🛒 Sales Channel Intelligence"
    )

    if "Sales_Channel" not in historical_df.columns:

        st.warning(
            "Sales_Channel column is not available."
        )

    else:

        channel = group_units(
            historical_df,
            "Sales_Channel"
        )

        st.bar_chart(
            channel.set_index(
                "Sales_Channel"
            )
        )

        st.dataframe(
            channel,
            use_container_width=True
        )


# ============================================================
# PAGE 12 — CUSTOMER INTELLIGENCE
# ============================================================

elif page == "Customer Intelligence":

    st.title(
        "👥 Customer Intelligence"
    )

    if (
        "Customer_Segment"
        not in historical_df.columns
    ):

        st.warning(
            "Customer_Segment column is not available."
        )

    else:

        segment = group_units(
            historical_df,
            "Customer_Segment"
        )

        st.bar_chart(
            segment.set_index(
                "Customer_Segment"
            )
        )

        st.dataframe(
            segment,
            use_container_width=True
        )


# ============================================================
# PAGE 13 — WEEKLY SALES INTELLIGENCE
# ============================================================

elif page == "Weekly Sales Intelligence":

    st.title(
        "📆 Weekly Sales Intelligence"
    )

    weekly = weekly_sales(
        historical_df
    )

    if weekly.empty:

        st.warning(
            "No weekly data available."
        )

    else:

        st.subheader(
            "Weekly Units Sold"
        )

        st.line_chart(
            weekly.set_index(
                "Date"
            )[["Units_Sold"]]
        )

        st.subheader(
            "Weekly Revenue"
        )

        st.line_chart(
            weekly.set_index(
                "Date"
            )[["Revenue"]]
        )

        st.dataframe(
            weekly,
            use_container_width=True
        )

# ============================================================
# PAGE 14 — FORECAST INTELLIGENCE
# ============================================================

elif page == "Forecast Intelligence":

    st.title("🔮 Forecast Intelligence")

    fc = forecast_df.copy()

    fc["Date"] = pd.to_datetime(
        fc["Date"]
    )

    fc["Predicted_Units_Sold"] = (
        pd.to_numeric(
            fc["Predicted_Units_Sold"],
            errors="coerce"
        )
        .fillna(0)
        .round()
        .astype(int)
    )

    total_forecast = (
        fc["Predicted_Units_Sold"]
        .sum()
    )

    average_forecast = (
        fc["Predicted_Units_Sold"]
        .mean()
    )

    peak_forecast = (
        fc["Predicted_Units_Sold"]
        .max()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Forecast",
        number(total_forecast)
    )

    c2.metric(
        "Average Daily",
        f"{average_forecast:,.1f}"
    )

    c3.metric(
        "Peak Daily",
        number(peak_forecast)
    )

    st.line_chart(
        fc.groupby(
            "Date"
        )["Predicted_Units_Sold"]
        .sum()
    )

    st.dataframe(
        fc,
        use_container_width=True
    )


# ============================================================
# PAGE 15 — MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.title("🤖 Model Intelligence")

    st.subheader(
        "Final Model"
    )

    model_info = pd.DataFrame({
        "Metric": [
            "Model",
            "Algorithm",
            "Target",
            "Transformation",
            "CV Method",
            "Selected Trial",
            "Forecast Frequency",
            "MAE",
            "MSE",
            "RMSE",
            "MAPE"
        ],
        "Value": [
            "Final Trial 84 Log-XGBoost",
            "XGBoost Regressor",
            "Units_Sold",
            "log1p → expm1",
            "TimeSeriesSplit",
            "Optuna Trial 84",
            "Daily",
            "2.280793",
            "25.524807",
            "5.052208",
            "11.136303%"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Model Features"
    )

    st.write(
        f"Total features: {len(MODEL_FEATURES)}"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": MODEL_FEATURES
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The final production model was trained using "
        "Optuna Trial 84 with log1p transformation of "
        "Units_Sold."
    )


# ============================================================
# PAGE 16 — CROSS-ANALYSIS EXPLORER
# ============================================================

elif page == "Cross-Analysis Explorer":

    st.title("🔍 Cross-Analysis Explorer")

    dimensions = [
        c for c in [
            "Product_Name",
            "Category",
            "Store_Location",
            "Sales_Channel",
            "Customer_Segment",
            "Season",
            "Holiday_Name"
        ]
        if c in historical_df.columns
    ]

    if not dimensions:

        st.warning(
            "No categorical dimensions available."
        )

    else:

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
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        st.bar_chart(
            result.set_index(selected)
        )

        st.dataframe(
            result,
            use_container_width=True
        )


# ============================================================
# PAGE 17 — DEMAND OPPORTUNITY FINDER
# ============================================================

elif page == "Demand Opportunity Finder":

    st.title("💡 Demand Opportunity Finder")

    if "Product_Name" not in historical_df.columns:

        st.warning(
            "Product_Name is unavailable."
        )

    else:

        product_summary = (
            historical_df
            .groupby("Product_Name")
            .agg(
                Units_Sold=(
                    "Units_Sold",
                    "sum"
                ),
                Revenue=(
                    "Revenue",
                    "sum"
                )
            )
            .reset_index()
        )

        product_summary[
            "Revenue_per_Unit"
        ] = (
            product_summary["Revenue"]
            /
            product_summary["Units_Sold"]
            .replace(0, np.nan)
        )

        product_summary = (
            product_summary
            .sort_values(
                "Units_Sold",
                ascending=False
            )
        )

        st.dataframe(
            product_summary,
            use_container_width=True
        )

        st.subheader(
            "Product Demand"
        )

        st.bar_chart(
            product_summary.set_index(
                "Product_Name"
            )[["Units_Sold"]]
        )


# ============================================================
# PAGE 18 — LEADERBOARDS
# ============================================================

elif page == "Leaderboards":

    st.title("🏆 Leaderboards")

    if "Product_Name" in historical_df.columns:

        st.subheader(
            "Top Products by Units Sold"
        )

        top_products = group_units(
            historical_df,
            "Product_Name"
        ).head(10)

        st.dataframe(
            top_products,
            use_container_width=True
        )

    if "Store_Location" in historical_df.columns:

        st.subheader(
            "Top Locations by Units Sold"
        )

        top_locations = group_units(
            historical_df,
            "Store_Location"
        ).head(10)

        st.dataframe(
            top_locations,
            use_container_width=True
        )

    if "Category" in historical_df.columns:

        st.subheader(
            "Top Categories"
        )

        top_categories = group_units(
            historical_df,
            "Category"
        ).head(10)

        st.dataframe(
            top_categories,
            use_container_width=True
        )


# ============================================================
# PAGE 19 — PRODUCT × LOCATION FINDER
# ============================================================

elif page == "Product × Location Finder":

    st.title("📦 × 📍 Product × Location Finder")

    if (
        "Product_Name" not in historical_df.columns
        or "Store_Location"
        not in historical_df.columns
    ):

        st.warning(
            "Product_Name or Store_Location unavailable."
        )

    else:

        products = sorted(
            historical_df[
                "Product_Name"
            ].dropna().unique()
        )

        locations = sorted(
            historical_df[
                "Store_Location"
            ].dropna().unique()
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

        temp = historical_df.copy()

        if selected_product != "All":

            temp = temp[
                temp["Product_Name"]
                == selected_product
            ]

        if selected_location != "All":

            temp = temp[
                temp["Store_Location"]
                == selected_location
            ]

        c1, c2 = st.columns(2)

        c1.metric(
            "Units Sold",
            number(
                safe_sum(
                    temp,
                    "Units_Sold"
                )
            )
        )

        c2.metric(
            "Revenue",
            money(
                safe_sum(
                    temp,
                    "Revenue"
                )
            )
        )

        daily = daily_sales(temp)

        if not daily.empty:

            st.line_chart(
                daily.set_index("Date")[
                    ["Units_Sold"]
                ]
            )

        st.dataframe(
            temp,
            use_container_width=True,
            height=400
        )


# ============================================================
# PAGE 20 — ASK FORECASTIQ
# ============================================================

elif page == "Ask ForecastIQ":

    st.title("💬 Ask ForecastIQ")

    question = st.text_input(
        "Ask a question about the sales data"
    )

    if question:

        q = question.lower()

        if "top product" in q:

            if "Product_Name" in historical_df.columns:

                result = (
                    historical_df
                    .groupby("Product_Name")
                    ["Units_Sold"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(5)
                )

                st.subheader(
                    "Top Products"
                )

                st.dataframe(
                    result.reset_index(),
                    use_container_width=True
                )

        elif (
            "region" in q
            or "location" in q
        ):

            if "Store_Location" in historical_df.columns:

                result = (
                    historical_df
                    .groupby("Store_Location")
                    ["Units_Sold"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.dataframe(
                    result.reset_index(),
                    use_container_width=True
                )

        elif "forecast" in q:

            total = safe_sum(
                forecast_df,
                "Predicted_Units_Sold"
            )

            st.success(
                f"Total available forecast: "
                f"{total:,.0f} units."
            )

        elif "revenue" in q:

            revenue = safe_sum(
                historical_df,
                "Revenue"
            )

            st.success(
                f"Historical revenue: "
                f"{revenue:,.0f}"
            )

        else:

            st.info(
                "Try questions such as:\n\n"
                "- What are the top products?\n"
                "- Show sales by region\n"
                "- What is the forecast?\n"
                "- What is the total revenue?"
            )


# ============================================================
# PAGE 21 — DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("🗃️ Data Explorer")

    st.write(
        f"Rows: {historical_df.shape[0]:,}"
    )

    st.write(
        f"Columns: {historical_df.shape[1]:,}"
    )

    st.subheader(
        "Historical Data"
    )

    st.dataframe(
        historical_df,
        use_container_width=True,
        height=500
    )

    st.subheader(
        "Column Information"
    )

    info_df = pd.DataFrame({
        "Column": historical_df.columns,
        "Data Type": [
            str(
                historical_df[c].dtype
            )
            for c in historical_df.columns
        ],
        "Missing Values": [
            historical_df[c].isna().sum()
            for c in historical_df.columns
        ],
        "Unique Values": [
            historical_df[c].nunique()
            for c in historical_df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


# ============================================================
# PAGE 22 — NEW PREDICTION
# ============================================================

elif page == "New Prediction":

    st.title("🚀 New Prediction")

    st.caption(
        "Generate a forecast scenario from the existing "
        "Trial-84 forecast using user-selected business inputs."
    )

    st.info(
        "The uploaded controls modify the existing "
        "Trial-84 baseline forecast. The product and region "
        "CSV files are allocation outputs based on historical "
        "sales shares."
    )

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    products = sorted(
        historical_df[
            "Product_Name"
        ].dropna().unique()
    ) if "Product_Name" in historical_df.columns else []

    regions = sorted(
        historical_df[
            "Store_Location"
        ].dropna().unique()
    ) if "Store_Location" in historical_df.columns else []

    c1, c2, c3 = st.columns(3)

    with c1:

        selected_product = st.selectbox(
            "Product",
            ["All Products"] + products,
            key="new_product"
        )

    with c2:

        selected_region = st.selectbox(
            "Store / Region",
            ["All Regions"] + regions,
            key="new_region"
        )

    with c3:

        current_date = st.date_input(
            "Current Date",
            value=HISTORICAL_CUTOFF.date(),
            min_value=HISTORICAL_CUTOFF.date(),
            key="new_current_date"
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        horizon = st.number_input(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=477,
            value=30,
            step=1,
            key="new_horizon"
        )

    with c2:

        price = st.number_input(
            "Price",
            min_value=0.01,
            value=float(
                historical_df["Price"].median()
                if "Price" in historical_df.columns
                else 100
            ),
            step=1.0,
            key="new_price"
        )

    with c3:

        discount = st.number_input(
            "Discount %",
            min_value=0.0,
            max_value=100.0,
            value=float(
                historical_df[
                    "Discount_Percentage"
                ].median()
                if "Discount_Percentage"
                in historical_df.columns
                else 0
            ),
            step=1.0,
            key="new_discount"
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        promotion = st.selectbox(
            "Promotion",
            [
                "No",
                "Yes"
            ],
            key="new_promotion"
        )

    with c2:

        stock = st.slider(
            "Stock Availability %",
            min_value=0,
            max_value=100,
            value=100,
            key="new_stock"
        )

    with c3:

        holiday = st.selectbox(
            "Holiday",
            [
                "No",
                "Yes"
            ],
            key="new_holiday"
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        local_event = st.selectbox(
            "Local Event",
            [
                "No",
                "Yes"
            ],
            key="new_event"
        )

    with c2:

        competitor_price = st.number_input(
            "Competitor Price",
            min_value=0.01,
            value=float(
                historical_df[
                    "Competitor_Price"
                ].median()
                if "Competitor_Price"
                in historical_df.columns
                else price
            ),
            step=1.0,
            key="new_competitor"
        )

    with c3:

        marketing_spend = st.number_input(
            "Marketing Spend",
            min_value=0.0,
            value=float(
                historical_df[
                    "Marketing_Spend"
                ].median()
                if "Marketing_Spend"
                in historical_df.columns
                else 1000
            ),
            step=100.0,
            key="new_marketing"
        )

    st.markdown("---")

    generate_prediction = st.button(
        "🚀 Generate New Prediction",
        type="primary",
        use_container_width=True
    )

    if generate_prediction:

        start_date = (
            pd.Timestamp(current_date)
            + pd.Timedelta(days=1)
        )

        base_forecast = prepare_overall_forecast(
            forecast_df,
            start_date,
            int(horizon)
        )

        if base_forecast.empty:

            st.error(
                "No baseline forecast is available for "
                "the selected Current Date and Horizon."
            )

        else:

            result = base_forecast.copy()

            # ------------------------------------------------
            # PRODUCT SHARE
            # ------------------------------------------------

            product_share = 1.0

            if (
                selected_product != "All Products"
                and "Product_Name"
                in historical_df.columns
            ):

                product_totals = (
                    historical_df
                    .groupby("Product_Name")
                    ["Units_Sold"]
                    .sum()
                )

                total_product_units = (
                    product_totals.sum()
                )

                if total_product_units > 0:

                    product_share = (
                        product_totals.get(
                            selected_product,
                            0
                        )
                        / total_product_units
                    )

            # ------------------------------------------------
            # REGION SHARE
            # ------------------------------------------------

            region_share = 1.0

            if (
                selected_region != "All Regions"
                and "Store_Location"
                in historical_df.columns
            ):

                region_totals = (
                    historical_df
                    .groupby("Store_Location")
                    ["Units_Sold"]
                    .sum()
                )

                total_region_units = (
                    region_totals.sum()
                )

                if total_region_units > 0:

                    region_share = (
                        region_totals.get(
                            selected_region,
                            0
                        )
                        / total_region_units
                    )

            # ------------------------------------------------
            # SCENARIO MULTIPLIER
            # ------------------------------------------------

            multiplier = scenario_multiplier(
                historical_df,
                (
                    None
                    if selected_product
                    == "All Products"
                    else selected_product
                ),
                safe_float(price),
                safe_float(discount),
                1 if promotion == "Yes" else 0,
                safe_float(stock),
                1 if holiday == "Yes" else 0,
                1 if local_event == "Yes" else 0,
                safe_float(competitor_price),
                safe_float(marketing_spend)
            )

            # ------------------------------------------------
            # APPLY FORECAST
            # ------------------------------------------------

            result[
                "Baseline_Forecast"
            ] = pd.to_numeric(
                result[
                    "Predicted_Units_Sold"
                ],
                errors="coerce"
            ).fillna(0)

            result[
                "Scenario_Multiplier"
            ] = multiplier

            result[
                "Product_Share"
            ] = product_share

            result[
                "Region_Share"
            ] = region_share

            result[
                "Predicted_Units_Sold"
            ] = (
                result[
                    "Baseline_Forecast"
                ]
                * multiplier
                * product_share
                * region_share
            )

            result[
                "Predicted_Units_Sold"
            ] = (
                result[
                    "Predicted_Units_Sold"
                ]
                .clip(lower=0)
                .round()
                .astype(int)
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            total_prediction = (
                result[
                    "Predicted_Units_Sold"
                ].sum()
            )

            average_prediction = (
                result[
                    "Predicted_Units_Sold"
                ].mean()
            )

            peak_prediction = (
                result[
                    "Predicted_Units_Sold"
                ].max()
            )

            baseline_total = (
                result[
                    "Baseline_Forecast"
                ].sum()
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Baseline Units",
                number(baseline_total)
            )

            c2.metric(
                "Scenario Units",
                number(total_prediction)
            )

            c3.metric(
                "Average Daily",
                f"{average_prediction:,.1f}"
            )

            c4.metric(
                "Scenario Multiplier",
                f"{multiplier:.3f}x"
            )

            st.success(
                "New prediction generated successfully."
            )

            st.subheader(
                "Prediction Trend"
            )

            chart = result[
                [
                    "Date",
                    "Baseline_Forecast",
                    "Predicted_Units_Sold"
                ]
            ].copy()

            chart = chart.set_index(
                "Date"
            )

            st.line_chart(
                chart
            )

            st.subheader(
                "Prediction Details"
            )

            st.dataframe(
                result,
                use_container_width=True,
                height=450
            )

            d1, d2 = st.columns(2)

            with d1:

                download_csv(
                    result,
                    "new_prediction.csv"
                )

            with d2:

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


