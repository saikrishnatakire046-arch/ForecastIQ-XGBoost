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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = Path(r"C:\Users\lenovo\ForecastIQ\sales_data.csv")
FORECAST_PATH = BASE_DIR / "future_sales_forecast_2026_2027.csv"

PRODUCT_FORECAST_PATH = BASE_DIR / "product_based_forecast_2026_2027.csv"
REGION_FORECAST_PATH = BASE_DIR / "region_based_forecast_2026_2027.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data


@st.cache_data
def load_forecast():
    data = pd.read_csv(FORECAST_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data


@st.cache_data
def load_optional_forecast(path):
    data = pd.read_csv(path)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data


# ============================================================
# LOAD
# ============================================================

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load historical dataset: {e}")
    st.stop()

try:
    forecast_df = load_forecast()
except Exception as e:
    st.error(f"Could not load forecast file: {e}")
    st.stop()

# Optional files. The app will still open if they are not present.
product_forecast_df = None
region_forecast_df = None

if PRODUCT_FORECAST_PATH.exists():
    try:
        product_forecast_df = load_optional_forecast(PRODUCT_FORECAST_PATH)
    except Exception as e:
        st.warning(f"Product forecast file could not be loaded: {e}")

if REGION_FORECAST_PATH.exists():
    try:
        region_forecast_df = load_optional_forecast(REGION_FORECAST_PATH)
    except Exception as e:
        st.warning(f"Region forecast file could not be loaded: {e}")


# ============================================================
# HISTORICAL CUTOFF
# ============================================================

HISTORICAL_CUTOFF = pd.Timestamp("2026-09-10")

historical_df = df[
    df["Date"] <= HISTORICAL_CUTOFF
].copy()


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "Row_ID",
    "Date",
    "Product_ID",
    "Product_Name",
    "Category",
    "Store_Location",
    "Units_Sold",
    "Price",
    "Discount_Percentage",
    "Revenue",
    "Promotion_Flag",
    "Stock_Availability",
    "Day_of_Week",
    "Month",
    "Quarter",
    "Is_Weekend",
    "Holiday_Flag",
    "Holiday_Name",
    "Local_Event_Flag",
    "Competitor_Price",
    "Economic_Indicator",
    "Marketing_Spend",
    "Sales_Channel",
    "Customer_Segment",
    "Season",
    "Weather"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns in historical dataset: {missing_columns}"
    )
    st.stop()


# ============================================================
# COMMON HELPER FUNCTIONS
# ============================================================

def weekly_sales(data):
    return (
        data.set_index("Date")["Units_Sold"]
        .resample("W-SUN")
        .sum()
    )


def daily_sales(data):
    return (
        data.set_index("Date")["Units_Sold"]
        .resample("D")
        .sum()
    )


def group_units(data, column):
    return (
        data.groupby(column)["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )


def group_revenue(data, column):
    return (
        data.groupby(column)["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def top_group(data, group_columns, metric="Units_Sold"):
    result = (
        data.groupby(group_columns)[metric]
        .sum()
        .reset_index()
    )

    return result.sort_values(
        metric,
        ascending=False
    )


def download_csv(data, filename):
    return st.download_button(
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
            index=False
        )

    return st.download_button(
        label="⬇️ Download Excel",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# APP HEADER
# ============================================================

st.title("📈 ForecastIQ")

st.caption(
    "AI-Powered Daily Sales Intelligence & Forecasting"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 ForecastIQ")

st.sidebar.caption(
    "Sales Intelligence Platform"
)

st.sidebar.divider()


# ============================================================
# TOP FORECAST FEATURES IN SIDEBAR
# ============================================================

st.sidebar.subheader("🔮 Forecast Tools")

product_options = sorted(
    historical_df["Product_Name"]
    .dropna()
    .unique()
)

location_options = sorted(
    historical_df["Store_Location"]
    .dropna()
    .unique()
)

if product_forecast_df is not None and "Product_Name" in product_forecast_df.columns:
    product_options = sorted(
        product_forecast_df["Product_Name"]
        .dropna()
        .unique()
    )

if region_forecast_df is not None and "Store_Location" in region_forecast_df.columns:
    location_options = sorted(
        region_forecast_df["Store_Location"]
        .dropna()
        .unique()
    )

selected_forecast_product = st.sidebar.selectbox(
    "📦 Product-Based Forecast",
    product_options,
    key="forecast_product_selector"
)

selected_forecast_region = st.sidebar.selectbox(
    "🌍 Region-Based Forecast",
    location_options,
    key="forecast_region_selector"
)

st.sidebar.divider()


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "📦 Product-Based Forecast",
        "📍 Region-Based Forecast",
        "Executive Dashboard",
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
)


st.sidebar.divider()

st.sidebar.metric(
    "Historical Records",
    f"{len(historical_df):,}"
)

st.sidebar.metric(
    "Forecast Days",
    f"{len(forecast_df):,}"
)

st.sidebar.caption(
    "ForecastIQ • Daily Sales Forecasting"
)


# ============================================================
# MAIN-AREA FORECAST TOOL HELPER
# ============================================================

def show_selected_product_forecast():
    st.header("📦 Product-Based Forecast")

    if product_forecast_df is None:
        st.info(
            "Product forecast file is not available yet. "
            "Place product_based_forecast_2026_2027.csv in the same folder as app.py."
        )
        return

    if "Product_Name" not in product_forecast_df.columns:
        st.error(
            "The product forecast CSV must contain a Product_Name column."
        )
        st.dataframe(
            product_forecast_df.head(),
            use_container_width=True,
            hide_index=True
        )
        return

    if "Predicted_Units_Sold" not in product_forecast_df.columns:
        st.error(
            "The product forecast CSV must contain Predicted_Units_Sold."
        )
        st.dataframe(
            product_forecast_df.head(),
            use_container_width=True,
            hide_index=True
        )
        return

    result = product_forecast_df[
        product_forecast_df["Product_Name"] == selected_forecast_product
    ].copy()

    if "Date" in result.columns:
        result = result.sort_values("Date")

    if len(result) == 0:
        st.warning(
            f"No forecast data found for {selected_forecast_product}."
        )
        return

    total_forecast = result["Predicted_Units_Sold"].sum()
    average_forecast = result["Predicted_Units_Sold"].mean()
    peak_forecast = result["Predicted_Units_Sold"].max()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:,.0f} units"
        )

    with c2:
        st.metric(
            "Average Daily Forecast",
            f"{average_forecast:,.2f}"
        )

    with c3:
        st.metric(
            "Peak Daily Forecast",
            f"{peak_forecast:,.0f} units"
        )

    st.divider()

    if "Date" in result.columns:
        st.subheader("Daily Product Forecast")
        st.line_chart(
            result.set_index("Date")["Predicted_Units_Sold"],
            height=400
        )

    st.subheader("Product Forecast Data")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    download_csv(
        result,
        f"ForecastIQ_{selected_forecast_product}_Forecast.csv"
    )


def show_selected_region_forecast():
    st.header("🌍 Region-Based Forecast")

    if region_forecast_df is None:
        st.info(
            "Region forecast file is not available yet. "
            "Place region_based_forecast_2026_2027.csv in the same folder as app.py."
        )
        return

    if "Store_Location" not in region_forecast_df.columns:
        st.error(
            "The region forecast CSV must contain a Store_Location column."
        )
        st.dataframe(
            region_forecast_df.head(),
            use_container_width=True,
            hide_index=True
        )
        return

    if "Predicted_Units_Sold" not in region_forecast_df.columns:
        st.error(
            "The region forecast CSV must contain Predicted_Units_Sold."
        )
        st.dataframe(
            region_forecast_df.head(),
            use_container_width=True,
            hide_index=True
        )
        return

    result = region_forecast_df[
        region_forecast_df["Store_Location"] == selected_forecast_region
    ].copy()

    if "Date" in result.columns:
        result = result.sort_values("Date")

    if len(result) == 0:
        st.warning(
            f"No forecast data found for {selected_forecast_region}."
        )
        return

    total_forecast = result["Predicted_Units_Sold"].sum()
    average_forecast = result["Predicted_Units_Sold"].mean()
    peak_forecast = result["Predicted_Units_Sold"].max()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:,.0f} units"
        )

    with c2:
        st.metric(
            "Average Daily Forecast",
            f"{average_forecast:,.2f}"
        )

    with c3:
        st.metric(
            "Peak Daily Forecast",
            f"{peak_forecast:,.0f} units"
        )

    st.divider()

    if "Date" in result.columns:
        st.subheader("Daily Region Forecast")
        st.line_chart(
            result.set_index("Date")["Predicted_Units_Sold"],
            height=400
        )

    st.subheader("Region Forecast Data")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    download_csv(
        result,
        f"ForecastIQ_{selected_forecast_region}_Forecast.csv"
    )


# ============================================================
# COMMON FORECAST INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)

with info1:
    st.caption("HISTORICAL DATA")
    st.write(
        f"{historical_df['Date'].min().strftime('%d %b %Y')} → "
        f"{historical_df['Date'].max().strftime('%d %b %Y')}"
    )

with info2:
    st.caption("FORECAST HORIZON")
    st.write("11 Sep 2026 → 31 Dec 2027")

with info3:
    st.caption("FORECAST FREQUENCY")
    st.write("Daily")

st.divider()


# ============================================================
# 0. PRODUCT-BASED FORECAST
# ============================================================

if page == "📦 Product-Based Forecast":
    st.header("📦 Product-Based Forecast")
    st.caption("Generate a product-level forecast with adjustable business scenario inputs and download the generated result as CSV.")

    if product_forecast_df.empty:
        st.error(
            "Product forecast data is not available. Place "
            "product_based_forecast_2026_2027.csv in the same folder as app.py."
        )
    else:
        products = sorted(product_forecast_df["Product_Name"].dropna().astype(str).unique().tolist())
        selected_product = st.selectbox("📦 Product", products)

        # Product-level forecast files do not contain Store_Location. We therefore
        # show the store selector from historical data as requested, while the
        # product forecast itself remains product-level.
        product_hist = historical_df[
            historical_df["Product_Name"].astype(str) == str(selected_product)
        ].copy()
        stores = sorted(product_hist["Store_Location"].dropna().astype(str).unique().tolist()) if "Store_Location" in product_hist.columns else []
        selected_store = st.selectbox(
            "🏪 Store",
            ["All Stores"] + stores,
            help="The supplied product forecast is product-level, so Store is a scenario/context selection and does not filter the CSV forecast."
        )

        forecast_min = product_forecast_df["Date"].min().date()
        forecast_max = product_forecast_df["Date"].max().date()
        default_date = max(forecast_min, min(pd.Timestamp("2026-09-15").date(), forecast_max))
        current_date = st.date_input(
            "📅 Current Date",
            value=default_date,
            min_value=forecast_min,
            max_value=forecast_max
        )

        # Historical defaults for the selected product.
        if not product_hist.empty:
            default_price = float(pd.to_numeric(product_hist["Price"], errors="coerce").dropna().mean()) if "Price" in product_hist.columns and pd.to_numeric(product_hist["Price"], errors="coerce").notna().any() else 100.0
            default_competitor = float(pd.to_numeric(product_hist["Competitor_Price"], errors="coerce").dropna().mean()) if "Competitor_Price" in product_hist.columns and pd.to_numeric(product_hist["Competitor_Price"], errors="coerce").notna().any() else default_price
            default_stock = float(pd.to_numeric(product_hist["Stock_Availability"], errors="coerce").dropna().mean()) if "Stock_Availability" in product_hist.columns and pd.to_numeric(product_hist["Stock_Availability"], errors="coerce").notna().any() else 100.0
        else:
            default_price = 100.0
            default_competitor = 100.0
            default_stock = 100.0

        c1, c2 = st.columns(2)
        with c1:
            base_price = st.number_input(
                "💰 Price",
                min_value=0.0,
                value=round(default_price, 2),
                step=1.0,
                format="%.2f",
                key="product_base_price"
            )
        with c2:
            price_change_pct = st.slider(
                "💰 Price Adjustment (%)",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                help="Negative values decrease price; positive values increase price."
            )

        adjusted_price = max(0.0, base_price * (1 + price_change_pct / 100.0))
        st.metric("Adjusted Price", f"₹{adjusted_price:,.2f}", f"{price_change_pct:+.0f}%")

        c1, c2, c3 = st.columns(3)
        with c1:
            discount_pct = st.number_input("🏷️ Discount %", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        with c2:
            promotion = st.selectbox("📢 Promotion", ["No", "Yes"])
        with c3:
            stock_availability = st.number_input(
                "📦 Stock Availability",
                min_value=0.0,
                value=round(default_stock, 2),
                step=1.0,
                format="%.2f"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            holiday = st.selectbox("🎉 Holiday", ["No", "Yes"])
        with c2:
            local_event = st.selectbox("📍 Local Event", ["No", "Yes"])
        with c3:
            competitor_price = st.number_input(
                "💰 Competitor Price",
                min_value=0.0,
                value=round(default_competitor, 2),
                step=1.0,
                format="%.2f"
            )

        forecast_horizon = st.slider(
            "🔮 Forecast Horizon (days)",
            min_value=1,
            max_value=90,
            value=30,
            step=1
        )

        st.divider()
        st.info(
            "Scenario inputs are applied to the existing product forecast as an adjustment layer. "
            "The supplied product forecast CSV is product-level and does not contain store-level or all scenario feature columns."
        )

        if st.button("🔮 Generate Forecast", type="primary", key="generate_product_forecast"):
            end_date = min(
                pd.Timestamp(current_date) + pd.Timedelta(days=forecast_horizon - 1),
                product_forecast_df["Date"].max()
            )

            result = product_forecast_df[
                (product_forecast_df["Product_Name"].astype(str) == str(selected_product))
                & (product_forecast_df["Date"] >= pd.Timestamp(current_date))
                & (product_forecast_df["Date"] <= end_date)
            ].copy()

            if result.empty:
                st.warning("No product forecast is available for the selected product/date range.")
            else:
                # Transparent scenario adjustment layer.
                # Price elasticity assumption: +1% price -> approximately -0.5% demand.
                price_factor = max(0.0, 1 - 0.50 * (price_change_pct / 100.0))
                discount_factor = 1 + 0.30 * (discount_pct / 100.0)
                promotion_factor = 1.10 if promotion == "Yes" else 1.00
                holiday_factor = 1.05 if holiday == "Yes" else 1.00
                event_factor = 1.05 if local_event == "Yes" else 1.00
                competitor_factor = 1.00
                if competitor_price > 0:
                    competitor_factor = np.clip(1 + 0.20 * ((adjusted_price - competitor_price) / competitor_price), 0.75, 1.25)
                stock_factor = np.clip(stock_availability / 100.0, 0.0, 1.0)

                scenario_factor = (
                    price_factor
                    * discount_factor
                    * promotion_factor
                    * holiday_factor
                    * event_factor
                    * competitor_factor
                    * stock_factor
                )

                result["Base_Predicted_Units_Sold"] = result["Predicted_Units_Sold"]
                result["Predicted_Units_Sold"] = (
                    result["Predicted_Units_Sold"] * scenario_factor
                ).clip(lower=0)
                result["Product"] = selected_product
                result["Store"] = selected_store
                result["Adjusted_Price"] = adjusted_price
                result["Price_Adjustment_%"] = price_change_pct
                result["Discount_%"] = discount_pct
                result["Promotion"] = promotion
                result["Stock_Availability"] = stock_availability
                result["Holiday"] = holiday
                result["Local_Event"] = local_event
                result["Competitor_Price"] = competitor_price

                st.success("Product forecast generated successfully!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Forecast Units", f"{result['Predicted_Units_Sold'].sum():,.0f}")
                m2.metric("Average Daily Forecast", f"{result['Predicted_Units_Sold'].mean():,.2f}")
                m3.metric("Peak Daily Forecast", f"{result['Predicted_Units_Sold'].max():,.0f}")

                st.subheader("📈 Forecast")
                st.line_chart(
                    result.set_index("Date")["Predicted_Units_Sold"],
                    use_container_width=True
                )

                st.subheader("📋 Forecast Details")
                display_cols = [
                    "Date", "Product", "Store", "Base_Predicted_Units_Sold",
                    "Predicted_Units_Sold", "Adjusted_Price", "Price_Adjustment_%",
                    "Discount_%", "Promotion", "Stock_Availability", "Holiday",
                    "Local_Event", "Competitor_Price"
                ]
                display_cols = [c for c in display_cols if c in result.columns]
                st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

                csv_data = result[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"{selected_product}_product_forecast.csv",
                    mime="text/csv",
                    key="download_product_forecast_csv"
                )


elif page == "📍 Region-Based Forecast":
    st.header("📍 Region-Based Forecast")
    st.caption("Generate a region/location-level forecast and download the generated result as CSV.")

    if region_forecast_df.empty:
        st.error(
            "Region forecast data is not available. Place "
            "region_based_forecast_2026_2027.csv in the same folder as app.py."
        )
    else:
        regions = sorted(region_forecast_df["Store_Location"].dropna().astype(str).unique().tolist())
        selected_region = st.selectbox("📍 Region / Location", regions)

        forecast_min = region_forecast_df["Date"].min().date()
        forecast_max = region_forecast_df["Date"].max().date()
        default_date = max(forecast_min, min(pd.Timestamp("2026-09-15").date(), forecast_max))
        current_date = st.date_input(
            "📅 Current Date",
            value=default_date,
            min_value=forecast_min,
            max_value=forecast_max,
            key="region_current_date"
        )

        forecast_horizon = st.slider(
            "🔮 Forecast Horizon (days)",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
            key="region_forecast_horizon"
        )

        if st.button("🔮 Generate Forecast", type="primary", key="generate_region_forecast"):
            end_date = min(
                pd.Timestamp(current_date) + pd.Timedelta(days=forecast_horizon - 1),
                region_forecast_df["Date"].max()
            )

            result = region_forecast_df[
                (region_forecast_df["Store_Location"].astype(str) == str(selected_region))
                & (region_forecast_df["Date"] >= pd.Timestamp(current_date))
                & (region_forecast_df["Date"] <= end_date)
            ].copy()

            if result.empty:
                st.warning("No regional forecast is available for the selected location/date range.")
            else:
                result["Region"] = selected_region

                st.success("Region forecast generated successfully!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Forecast Units", f"{result['Predicted_Units_Sold'].sum():,.0f}")
                m2.metric("Average Daily Forecast", f"{result['Predicted_Units_Sold'].mean():,.2f}")
                m3.metric("Peak Daily Forecast", f"{result['Predicted_Units_Sold'].max():,.0f}")

                st.subheader("📈 Forecast")
                st.line_chart(
                    result.set_index("Date")["Predicted_Units_Sold"],
                    use_container_width=True
                )

                st.subheader("📋 Forecast Details")
                display_cols = ["Date", "Region", "Store_Location", "Predicted_Units_Sold"]
                display_cols = [c for c in display_cols if c in result.columns]
                st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

                csv_data = result[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"{selected_region}_region_forecast.csv",
                    mime="text/csv",
                    key="download_region_forecast_csv"
                )


if page == "Executive Dashboard":
    # Product and region forecasts are available from the sidebar.
    # The normal default page remains Executive Dashboard.

    st.header("📦 Product-Based Forecast")
    st.caption(
        f"Forecast for selected product: {selected_forecast_product}"
    )

    if product_forecast_df is not None and "Product_Name" in product_forecast_df.columns:
        product_result = product_forecast_df[
            product_forecast_df["Product_Name"] == selected_forecast_product
        ].copy()

        if "Date" in product_result.columns:
            product_result = product_result.sort_values("Date")

        if "Predicted_Units_Sold" in product_result.columns and len(product_result) > 0:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Total Product Forecast",
                    f"{product_result['Predicted_Units_Sold'].sum():,.0f} units"
                )

            with c2:
                st.metric(
                    "Average Daily Forecast",
                    f"{product_result['Predicted_Units_Sold'].mean():,.2f}"
                )

            with c3:
                st.metric(
                    "Peak Daily Forecast",
                    f"{product_result['Predicted_Units_Sold'].max():,.0f} units"
                )

            if "Date" in product_result.columns:
                st.line_chart(
                    product_result.set_index("Date")[
                        "Predicted_Units_Sold"
                    ],
                    height=300
                )

            st.dataframe(
                product_result,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No product-specific forecast data is available for the selected product."
            )
    else:
        st.info(
            "Product forecast CSV is not available or does not contain Product_Name yet."
        )

    st.divider()

    # ========================================================
    # 1. REGION-BASED FORECAST
    # ========================================================

    st.header("🌍 Region-Based Forecast")
    st.caption(
        f"Forecast for selected region: {selected_forecast_region}"
    )

    if region_forecast_df is not None and "Store_Location" in region_forecast_df.columns:
        region_result = region_forecast_df[
            region_forecast_df["Store_Location"] == selected_forecast_region
        ].copy()

        if "Date" in region_result.columns:
            region_result = region_result.sort_values("Date")

        if "Predicted_Units_Sold" in region_result.columns and len(region_result) > 0:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Total Region Forecast",
                    f"{region_result['Predicted_Units_Sold'].sum():,.0f} units"
                )

            with c2:
                st.metric(
                    "Average Daily Forecast",
                    f"{region_result['Predicted_Units_Sold'].mean():,.2f}"
                )

            with c3:
                st.metric(
                    "Peak Daily Forecast",
                    f"{region_result['Predicted_Units_Sold'].max():,.0f} units"
                )

            if "Date" in region_result.columns:
                st.line_chart(
                    region_result.set_index("Date")[
                        "Predicted_Units_Sold"
                    ],
                    height=300
                )

            st.dataframe(
                region_result,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No region-specific forecast data is available for the selected region."
            )
    else:
        st.info(
            "Region forecast CSV is not available or does not contain Store_Location yet."
        )

    st.divider()


    # ========================================================
    # 2. EXECUTIVE DASHBOARD
    # ========================================================

    st.header("Executive Dashboard")

    st.caption(
        "High-level view of historical sales performance and business trends."
    )

    total_units = historical_df["Units_Sold"].sum()
    total_revenue = historical_df["Revenue"].sum()
    daily = daily_sales(historical_df)
    avg_daily = daily.mean()

    best_product = (
        historical_df.groupby("Product_Name")["Units_Sold"]
        .sum()
        .idxmax()
    )

    best_location = (
        historical_df.groupby("Store_Location")["Units_Sold"]
        .sum()
        .idxmax()
    )

    best_category = (
        historical_df.groupby("Category")["Units_Sold"]
        .sum()
        .idxmax()
    )

    best_channel = (
        historical_df.groupby("Sales_Channel")["Units_Sold"]
        .sum()
        .idxmax()
    )

    best_segment = (
        historical_df.groupby("Customer_Segment")["Units_Sold"]
        .sum()
        .idxmax()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Units Sold", f"{total_units:,.0f}")

    with c2:
        st.metric("Total Revenue", f"{total_revenue:,.2f}")

    with c3:
        st.metric("Average Daily Demand", f"{avg_daily:,.2f}")

    with c4:
        st.metric("Best Product", best_product)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Best Location", best_location)

    with c2:
        st.metric("Best Category", best_category)

    with c3:
        st.metric("Best Channel", best_channel)

    with c4:
        st.metric("Best Customer Segment", best_segment)

    st.divider()

    st.subheader("📈 Overall Daily Sales Trend")

    st.line_chart(daily, height=400)

    peak_day = daily.idxmax()
    lowest_day = daily.idxmin()

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            f"📈 Highest Sales Day\n\n"
            f"{peak_day.strftime('%d %b %Y')} — "
            f"{daily.max():,.0f} units"
        )

    with c2:
        st.warning(
            f"📉 Lowest Sales Day\n\n"
            f"{lowest_day.strftime('%d %b %Y')} — "
            f"{daily.min():,.0f} units"
        )

    st.divider()

    st.subheader("🚨 Quick Business Alerts")

    stock_rate = historical_df["Stock_Availability"].mean()

    if stock_rate < 0.80:
        st.warning("Stock availability requires attention.")

    if historical_df["Promotion_Flag"].mean() > 0.50:
        st.info(
            "Promotions are present in more than 50% of historical records."
        )

    if daily.max() > avg_daily * 1.5:
        st.success("Peak demand is significantly above average.")


# ============================================================
# 3. LOCATION INTELLIGENCE
# ============================================================

elif page == "Location Intelligence":

    st.header("📍 Location Intelligence")
    st.caption("Compare performance across all locations.")

    location_summary = (
        historical_df.groupby("Store_Location")
        .agg(
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Revenue=("Revenue", "sum"),
            Avg_Price=("Price", "mean")
        )
        .reset_index()
    )

    location_weekly = (
        historical_df
        .set_index("Date")
        .groupby("Store_Location")["Units_Sold"]
        .resample("W-SUN")
        .sum()
        .reset_index()
    )

    avg_weekly_location = (
        location_weekly
        .groupby("Store_Location")["Units_Sold"]
        .mean()
        .reset_index(name="Avg_Weekly_Demand")
    )

    location_summary = location_summary.merge(
        avg_weekly_location,
        on="Store_Location"
    )

    location_summary["Rank"] = (
        location_summary["Total_Units_Sold"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    location_summary = location_summary.sort_values("Rank")

    cols = st.columns(len(location_summary))

    for col, (_, row) in zip(cols, location_summary.iterrows()):
        with col:
            st.metric(
                row["Store_Location"],
                f"{row['Total_Units_Sold']:,.0f}"
            )
            st.caption(
                f"Rank #{row['Rank']} | "
                f"Avg weekly {row['Avg_Weekly_Demand']:,.0f}"
            )

    st.divider()

    st.subheader("Location Performance")
    st.dataframe(
        location_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🏆 Which Product Sells Most in Each Location?")

    location_product = (
        historical_df.groupby(
            ["Store_Location", "Product_Name"]
        )["Units_Sold"]
        .sum()
        .reset_index()
    )

    top_location_product = location_product.loc[
        location_product.groupby("Store_Location")["Units_Sold"].idxmax()
    ]

    st.dataframe(
        top_location_product,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Weekly Demand by Location")

    location_weekly_pivot = (
        location_weekly
        .pivot(
            index="Date",
            columns="Store_Location",
            values="Units_Sold"
        )
        .fillna(0)
    )

    st.line_chart(location_weekly_pivot, height=400)

    st.divider()

    st.subheader("🔥 Location × Product")

    location_product_heatmap = pd.pivot_table(
        historical_df,
        index="Store_Location",
        columns="Product_Name",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(location_product_heatmap, use_container_width=True)

    st.divider()

    st.subheader("🔥 Location × Category")

    location_category_heatmap = pd.pivot_table(
        historical_df,
        index="Store_Location",
        columns="Category",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(location_category_heatmap, use_container_width=True)


# ============================================================
# 4. PRODUCT INTELLIGENCE
# ============================================================

elif page == "Product Intelligence":

    st.header("📦 Product Intelligence")

    product_summary = (
        historical_df.groupby("Product_Name")
        .agg(
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Revenue=("Revenue", "sum"),
            Avg_Price=("Price", "mean"),
            Avg_Discount=("Discount_Percentage", "mean")
        )
        .reset_index()
        .sort_values("Total_Units_Sold", ascending=False)
    )

    st.subheader("Product Ranking")
    st.dataframe(
        product_summary,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        product_summary.set_index("Product_Name")["Total_Units_Sold"]
    )

    st.divider()

    selected_product = st.selectbox(
        "Select Product",
        sorted(historical_df["Product_Name"].dropna().unique())
    )

    product_data = historical_df[
        historical_df["Product_Name"] == selected_product
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Units Sold", f"{product_data['Units_Sold'].sum():,.0f}")

    with c2:
        st.metric("Revenue", f"{product_data['Revenue'].sum():,.2f}")

    with c3:
        st.metric("Average Price", f"{product_data['Price'].mean():,.2f}")

    with c4:
        st.metric(
            "Average Discount",
            f"{product_data['Discount_Percentage'].mean():,.2f}%"
        )

    st.divider()

    product_location = (
        product_data
        .groupby("Store_Location")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.subheader("Product × Location Performance")
    st.bar_chart(product_location)

    st.subheader("Weekly Product Sales")

    product_weekly = (
        product_data
        .set_index("Date")["Units_Sold"]
        .resample("W-SUN")
        .sum()
    )

    st.line_chart(product_weekly)


# ============================================================
# 5. QUARTER INTELLIGENCE
# ============================================================

elif page == "Quarter Intelligence":

    st.header("📅 Quarter Intelligence")

    quarter_product = (
        historical_df.groupby(["Quarter", "Product_Name"])["Units_Sold"]
        .sum()
        .reset_index()
    )

    st.subheader("🏆 Which Product Sold Most in Each Quarter?")

    top_quarter_product = quarter_product.loc[
        quarter_product.groupby("Quarter")["Units_Sold"].idxmax()
    ]

    st.dataframe(
        top_quarter_product,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Quarter × Product")

    quarter_product_pivot = pd.pivot_table(
        historical_df,
        index="Quarter",
        columns="Product_Name",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(quarter_product_pivot, use_container_width=True)

    st.divider()

    st.subheader("Quarter × Location")

    quarter_location = pd.pivot_table(
        historical_df,
        index="Quarter",
        columns="Store_Location",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(quarter_location, use_container_width=True)

    st.divider()

    st.subheader("Quarter × Category")

    quarter_category = pd.pivot_table(
        historical_df,
        index="Quarter",
        columns="Category",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(quarter_category, use_container_width=True)

    st.divider()

    st.subheader("Quarter × Channel")

    quarter_channel = pd.pivot_table(
        historical_df,
        index="Quarter",
        columns="Sales_Channel",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(quarter_channel, use_container_width=True)


# ============================================================
# 6. DEMAND DRIVERS
# ============================================================

elif page == "Demand Drivers":

    st.header("🎯 Demand Drivers")

    numeric_driver_columns = [
        "Price",
        "Discount_Percentage",
        "Marketing_Spend",
        "Competitor_Price",
        "Revenue",
        "Economic_Indicator",
        "Units_Sold"
    ]

    available_drivers = [
        col for col in numeric_driver_columns
        if col in historical_df.columns
    ]

    corr = historical_df[available_drivers].corr()

    st.subheader("Overall Correlation Matrix")
    st.dataframe(corr.round(3), use_container_width=True)

    st.divider()

    target_corr = (
        corr["Units_Sold"]
        .drop("Units_Sold")
        .sort_values()
    )

    st.subheader("Relationships with Units Sold")
    st.dataframe(
        target_corr.to_frame("Correlation"),
        use_container_width=True
    )

    st.divider()

    driver = st.selectbox(
        "Select Demand Driver",
        [
            "Price",
            "Discount_Percentage",
            "Marketing_Spend",
            "Competitor_Price",
            "Revenue",
            "Economic_Indicator"
        ]
    )

    scatter_data = historical_df[[driver, "Units_Sold"]].dropna()

    st.scatter_chart(
        scatter_data,
        x=driver,
        y="Units_Sold"
    )

    st.info("⚠️ Correlation shows association, not causation.")

    st.divider()

    st.subheader("Promotion vs Demand")

    promotion_sales = (
        historical_df.groupby("Promotion_Flag")["Units_Sold"]
        .mean()
    )

    st.bar_chart(promotion_sales)

    st.subheader("Stock Availability vs Demand")

    stock_sales = (
        historical_df.groupby("Stock_Availability")["Units_Sold"]
        .mean()
    )

    st.bar_chart(stock_sales)


# ============================================================
# 7. PRICING INTELLIGENCE
# ============================================================

elif page == "Pricing Intelligence":

    st.header("💰 Pricing Intelligence")

    yearly_price = (
        historical_df
        .groupby(historical_df["Date"].dt.year)["Price"]
        .mean()
    )

    st.subheader("Average Price by Year")
    st.line_chart(yearly_price)

    st.divider()

    st.subheader("Product Price Trends")

    product_price = (
        historical_df
        .assign(Year=historical_df["Date"].dt.year)
        .groupby(["Year", "Product_Name"])["Price"]
        .mean()
        .unstack()
    )

    st.line_chart(product_price)

    st.divider()

    st.subheader("Location Price Trends")

    location_price = (
        historical_df
        .assign(Year=historical_df["Date"].dt.year)
        .groupby(["Year", "Store_Location"])["Price"]
        .mean()
        .unstack()
    )

    st.line_chart(location_price)

    st.divider()

    st.subheader("Our Price vs Competitor Price")

    price_compare = historical_df[
        ["Date", "Price", "Competitor_Price"]
    ].set_index("Date")

    st.line_chart(price_compare)

    st.divider()

    st.subheader("Discount Trend")

    discount_trend = (
        historical_df
        .set_index("Date")["Discount_Percentage"]
        .resample("W-SUN")
        .mean()
    )

    st.line_chart(discount_trend)


# ============================================================
# 8. PROMOTION INTELLIGENCE
# ============================================================

elif page == "Promotion Intelligence":

    st.header("📢 Promotion Intelligence")

    promo_summary = (
        historical_df
        .groupby("Promotion_Flag")
        .agg(
            Total_Units=("Units_Sold", "sum"),
            Average_Units=("Units_Sold", "mean"),
            Total_Revenue=("Revenue", "sum"),
            Average_Discount=("Discount_Percentage", "mean")
        )
        .reset_index()
    )

    promo_summary["Promotion_Status"] = np.where(
        promo_summary["Promotion_Flag"] == 1,
        "Promotion",
        "No Promotion"
    )

    st.dataframe(
        promo_summary,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Promotion Impact by Location")

    promo_location = pd.pivot_table(
        historical_df,
        index="Store_Location",
        columns="Promotion_Flag",
        values="Units_Sold",
        aggfunc="mean",
        fill_value=0
    )

    st.dataframe(promo_location, use_container_width=True)

    st.divider()

    st.subheader("Promotion Impact by Product")

    promo_product = pd.pivot_table(
        historical_df,
        index="Product_Name",
        columns="Promotion_Flag",
        values="Units_Sold",
        aggfunc="mean",
        fill_value=0
    )

    st.dataframe(promo_product, use_container_width=True)

    st.divider()

    st.subheader("Discount vs Demand")

    st.scatter_chart(
        historical_df[
            ["Discount_Percentage", "Units_Sold"]
        ].dropna(),
        x="Discount_Percentage",
        y="Units_Sold"
    )


# ============================================================
# 9. CATEGORY INTELLIGENCE
# ============================================================

elif page == "Category Intelligence":

    st.header("🗂️ Category Intelligence")

    category_summary = (
        historical_df
        .groupby("Category")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .sort_values("Units_Sold", ascending=False)
    )

    st.subheader("Category Ranking")
    st.dataframe(category_summary, use_container_width=True)
    st.bar_chart(category_summary["Units_Sold"])

    st.divider()

    st.subheader("Category × Location")

    category_location = pd.pivot_table(
        historical_df,
        index="Category",
        columns="Store_Location",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(category_location, use_container_width=True)

    st.divider()

    st.subheader("Category × Product")

    category_product = pd.pivot_table(
        historical_df,
        index="Category",
        columns="Product_Name",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(category_product, use_container_width=True)

    st.divider()

    st.subheader("Category × Quarter")

    category_quarter = pd.pivot_table(
        historical_df,
        index="Category",
        columns="Quarter",
        values="Units_Sold",
        aggfunc="sum",
        fill_value=0
    )

    st.dataframe(category_quarter, use_container_width=True)


# ============================================================
# 10. SALES CHANNEL INTELLIGENCE
# ============================================================

elif page == "Sales Channel Intelligence":

    st.header("🛒 Sales Channel Intelligence")

    channel_summary = (
        historical_df
        .groupby("Sales_Channel")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .sort_values("Units_Sold", ascending=False)
    )

    st.subheader("Channel Performance")
    st.dataframe(channel_summary, use_container_width=True)
    st.bar_chart(channel_summary["Units_Sold"])

    st.divider()

    for dimension in [
        "Store_Location",
        "Product_Name",
        "Category",
        "Customer_Segment"
    ]:

        st.subheader(f"Channel × {dimension}")

        table = pd.pivot_table(
            historical_df,
            index=dimension,
            columns="Sales_Channel",
            values="Units_Sold",
            aggfunc="sum",
            fill_value=0
        )

        st.dataframe(table, use_container_width=True)


# ============================================================
# 11. CUSTOMER INTELLIGENCE
# ============================================================

elif page == "Customer Intelligence":

    st.header("👥 Customer Intelligence")

    segment_summary = (
        historical_df
        .groupby("Customer_Segment")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum")
        )
        .sort_values("Units_Sold", ascending=False)
    )

    st.subheader("Customer Segment Performance")
    st.dataframe(segment_summary, use_container_width=True)
    st.bar_chart(segment_summary["Units_Sold"])

    st.divider()

    for dimension in [
        "Store_Location",
        "Product_Name",
        "Category",
        "Sales_Channel"
    ]:

        st.subheader(f"Segment × {dimension}")

        table = pd.pivot_table(
            historical_df,
            index=dimension,
            columns="Customer_Segment",
            values="Units_Sold",
            aggfunc="sum",
            fill_value=0
        )

        st.dataframe(table, use_container_width=True)


# ============================================================
# 12. WEEKLY SALES INTELLIGENCE
# ============================================================

elif page == "Weekly Sales Intelligence":

    st.header("📅 Weekly Sales Intelligence")

    weekly = weekly_sales(historical_df)

    weekly_df = weekly.to_frame("Units_Sold")

    weekly_df["WoW_Growth_%"] = (
        weekly_df["Units_Sold"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )

    weekly_df["Rolling_Mean_4"] = (
        weekly_df["Units_Sold"].rolling(4).mean()
    )

    weekly_df["Rolling_Std_4"] = (
        weekly_df["Units_Sold"].rolling(4).std()
    )

    st.subheader("Weekly Sales")

    st.line_chart(
        weekly_df[["Units_Sold", "Rolling_Mean_4"]],
        height=400
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Average Weekly Demand", f"{weekly.mean():,.0f}")

    with c2:
        st.metric("Peak Week", f"{weekly.max():,.0f}")

    with c3:
        st.metric("Demand Volatility", f"{weekly.std():,.0f}")

    st.divider()

    st.subheader("Week-over-Week Growth")
    st.line_chart(weekly_df["WoW_Growth_%"], height=350)

    st.divider()

    st.subheader("Peak Weeks")

    st.dataframe(
        weekly_df.sort_values(
            "Units_Sold",
            ascending=False
        ).head(10),
        use_container_width=True
    )

    st.subheader("Low-Demand Weeks")

    st.dataframe(
        weekly_df.sort_values("Units_Sold").head(10),
        use_container_width=True
    )


# ============================================================
# 13. FORECAST INTELLIGENCE
# ============================================================

elif page == "Forecast Intelligence":

    st.header("🔮 Forecast Intelligence")

    forecast_display = forecast_df.copy()

    forecast_display["Date"] = pd.to_datetime(
        forecast_display["Date"]
    )

    st.subheader("Daily Forecast — Sep 2026 to Dec 2027")

    st.line_chart(
        forecast_display.set_index("Date")[
            "Predicted_Units_Sold"
        ],
        height=450
    )

    st.divider()

    total_forecast = forecast_display[
        "Predicted_Units_Sold"
    ].sum()

    peak_forecast = forecast_display.loc[
        forecast_display["Predicted_Units_Sold"].idxmax()
    ]

    lowest_forecast = forecast_display.loc[
        forecast_display["Predicted_Units_Sold"].idxmin()
    ]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:,.0f} units"
        )

    with c2:
        st.metric(
            "Peak Forecast Day",
            peak_forecast["Date"].strftime("%d %b %Y")
        )

    with c3:
        st.metric(
            "Lowest Forecast Day",
            lowest_forecast["Date"].strftime("%d %b %Y")
        )

    st.divider()

    st.subheader("Monthly Forecast")

    monthly_forecast = (
        forecast_display
        .set_index("Date")["Predicted_Units_Sold"]
        .resample("M")
        .sum()
    )

    st.bar_chart(monthly_forecast)

    st.divider()

    st.subheader("Quarterly Forecast")

    quarterly_forecast = (
        forecast_display
        .set_index("Date")["Predicted_Units_Sold"]
        .resample("Q")
        .sum()
    )

    st.bar_chart(quarterly_forecast)

    st.divider()

    st.subheader("2027 Forecast")

    forecast_2027 = forecast_display[
        forecast_display["Date"].dt.year == 2027
    ]

    st.metric(
        "Total 2027 Forecast",
        f"{forecast_2027['Predicted_Units_Sold'].sum():,.0f} units"
    )

    st.divider()

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        download_csv(
            forecast_display,
            "ForecastIQ_Daily_Forecast.csv"
        )

    with c2:
        download_excel(
            forecast_display,
            "ForecastIQ_Daily_Forecast.xlsx"
        )


# ============================================================
# 14. MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.header("🤖 Model Intelligence")

    st.caption(
        "Final model evaluation from the completed XGBoost forecasting project."
    )

    model_results = pd.DataFrame({
        "Model": ["Final Trial 84 Log-XGBoost"],
        "MAE": [2.280793],
        "MSE": [25.524807],
        "RMSE": [5.052208],
        "MAPE": [11.136303]
    })

    model_results["Rank"] = [1]

    st.subheader("Final Model Performance")

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("🏆 Best Model")

    st.success(
        "Trial 84 Log-XGBoost is the final selected model."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("MAE", "2.280793")

    with c2:
        st.metric("MSE", "25.524807")

    with c3:
        st.metric("RMSE", "5.052208")

    with c4:
        st.metric("MAPE", "11.136303%")

    st.divider()

    st.subheader("Model Configuration")

    st.write("Algorithm: XGBoost Regressor")
    st.write("Target transformation: log1p(Unit_Sold)")
    st.write("Prediction transformation: expm1()")
    st.write("Cross-validation: TimeSeriesSplit")
    st.write("Final selected model: Optuna Trial 84")
    st.write("Forecast frequency: Daily")


# ============================================================
# 15. CROSS-ANALYSIS EXPLORER
# ============================================================

elif page == "Cross-Analysis Explorer":

    st.header("🔍 Cross-Analysis Explorer")

    st.caption(
        "Select dimensions and metric to explore business relationships."
    )

    dimensions = [
        "Store_Location",
        "Product_Name",
        "Category",
        "Sales_Channel",
        "Customer_Segment",
        "Quarter"
    ]

    x_dimension = st.selectbox(
        "Primary Dimension",
        dimensions
    )

    breakdown_dimension = st.selectbox(
        "Breakdown Dimension",
        ["None"] + [
            x for x in dimensions
            if x != x_dimension
        ]
    )

    metric = st.selectbox(
        "Metric",
        [
            "Units_Sold",
            "Revenue",
            "Price",
            "Discount_Percentage",
            "Marketing_Spend"
        ]
    )

    if breakdown_dimension == "None":

        result = (
            historical_df
            .groupby(x_dimension)[metric]
            .sum()
            .sort_values(ascending=False)
        )

        st.subheader(f"{metric} by {x_dimension}")

        st.dataframe(
            result.to_frame(),
            use_container_width=True
        )

        st.bar_chart(result)

    else:

        result = pd.pivot_table(
            historical_df,
            index=x_dimension,
            columns=breakdown_dimension,
            values=metric,
            aggfunc="sum",
            fill_value=0
        )

        st.subheader(
            f"{x_dimension} × {breakdown_dimension}"
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.bar_chart(result)


# ============================================================
# 16. DEMAND OPPORTUNITY FINDER
# ============================================================

elif page == "Demand Opportunity Finder":

    st.header("💡 Demand Opportunity Finder")

    product_demand = (
        historical_df
        .groupby("Product_Name")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Avg_Stock_Availability=("Stock_Availability", "mean"),
            Avg_Marketing_Spend=("Marketing_Spend", "mean"),
            Avg_Price=("Price", "mean")
        )
        .reset_index()
    )

    st.subheader("📦 High-Demand Products")

    high_demand = (
        product_demand
        .sort_values("Units_Sold", ascending=False)
        .head(10)
    )

    st.dataframe(
        high_demand,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⚠️ High Demand + Low Stock")

    high_demand_low_stock = product_demand[
        (
            product_demand["Units_Sold"]
            >= product_demand["Units_Sold"].median()
        )
        &
        (
            product_demand["Avg_Stock_Availability"] < 0.80
        )
    ]

    if len(high_demand_low_stock) > 0:
        st.dataframe(
            high_demand_low_stock,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success(
            "No products currently meet the high-demand + low-stock condition."
        )

    st.divider()

    st.subheader("💰 High Marketing + Low Sales")

    marketing_opportunity = product_demand[
        (
            product_demand["Avg_Marketing_Spend"]
            >= product_demand["Avg_Marketing_Spend"].median()
        )
        &
        (
            product_demand["Units_Sold"]
            < product_demand["Units_Sold"].median()
        )
    ]

    st.dataframe(
        marketing_opportunity,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Fast-Growing Products")

    product_weekly = (
        historical_df
        .assign(
            Week=historical_df["Date"]
            .dt.to_period("W")
            .apply(lambda x: x.start_time)
        )
        .groupby(["Week", "Product_Name"])["Units_Sold"]
        .sum()
        .reset_index()
    )

    growth_results = []

    for product in product_weekly["Product_Name"].unique():

        p = product_weekly[
            product_weekly["Product_Name"] == product
        ].sort_values("Week")

        if len(p) >= 4:

            first = p["Units_Sold"].head(
                max(1, len(p) // 4)
            ).mean()

            last = p["Units_Sold"].tail(
                max(1, len(p) // 4)
            ).mean()

            growth = (
                ((last - first) / first) * 100
                if first != 0
                else np.nan
            )

            growth_results.append(
                [product, growth]
            )

    growth_df = pd.DataFrame(
        growth_results,
        columns=["Product_Name", "Growth_%"]
    ).sort_values("Growth_%", ascending=False)

    st.dataframe(
        growth_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("⚠️ Price Rising While Demand Falls")

    product_year = (
        historical_df
        .assign(Year=historical_df["Date"].dt.year)
        .groupby(["Product_Name", "Year"])
        .agg(
            Price=("Price", "mean"),
            Units=("Units_Sold", "sum")
        )
        .reset_index()
    )

    price_demand_alerts = []

    for product in product_year["Product_Name"].unique():

        p = product_year[
            product_year["Product_Name"] == product
        ].sort_values("Year")

        if len(p) >= 2:

            price_change = (
                p.iloc[-1]["Price"]
                - p.iloc[0]["Price"]
            )

            demand_change = (
                p.iloc[-1]["Units"]
                - p.iloc[0]["Units"]
            )

            if price_change > 0 and demand_change < 0:
                price_demand_alerts.append(
                    [
                        product,
                        price_change,
                        demand_change
                    ]
                )

    alerts_df = pd.DataFrame(
        price_demand_alerts,
        columns=[
            "Product_Name",
            "Price_Change",
            "Demand_Change"
        ]
    )

    if len(alerts_df) > 0:
        st.dataframe(
            alerts_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success(
            "No product currently shows both rising price and falling demand."
        )


# ============================================================
# 17. LEADERBOARDS
# ============================================================

elif page == "Leaderboards":

    st.header("🏆 Leaderboards")

    st.subheader("Top 10 Products")

    top_products = (
        historical_df
        .groupby("Product_Name")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_products)

    st.divider()

    st.subheader("Top Locations")

    top_locations = (
        historical_df
        .groupby("Store_Location")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(top_locations)

    st.divider()

    st.subheader("Top Categories")

    top_categories = (
        historical_df
        .groupby("Category")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(top_categories)

    st.divider()

    st.subheader("Top Sales Channels")

    top_channels = (
        historical_df
        .groupby("Sales_Channel")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(top_channels)

    st.divider()

    st.subheader("Top Customer Segments")

    top_segments = (
        historical_df
        .groupby("Customer_Segment")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(top_segments)

    st.divider()

    st.subheader("Top Product × Location Combinations")

    top_combinations = (
        historical_df
        .groupby(["Product_Name", "Store_Location"])["Units_Sold"]
        .sum()
        .reset_index()
        .sort_values("Units_Sold", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_combinations,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Top Quarter × Product")

    top_quarter_products = (
        historical_df
        .groupby(["Quarter", "Product_Name"])["Units_Sold"]
        .sum()
        .reset_index()
        .sort_values("Units_Sold", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_quarter_products,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 18. PRODUCT × LOCATION FINDER
# ============================================================

elif page == "Product × Location Finder":

    st.header("📍 Product × Location Finder")

    selected_product = st.selectbox(
        "Choose Product",
        sorted(
            historical_df["Product_Name"]
            .dropna()
            .unique()
        )
    )

    product_location = (
        historical_df[
            historical_df["Product_Name"] == selected_product
        ]
        .groupby("Store_Location")
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum"),
            Avg_Price=("Price", "mean")
        )
        .reset_index()
        .sort_values("Units_Sold", ascending=False)
    )

    product_location["Rank"] = (
        product_location["Units_Sold"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    st.subheader(
        f"{selected_product} — Location Comparison"
    )

    st.dataframe(
        product_location,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        product_location.set_index("Store_Location")["Units_Sold"]
    )

    best = product_location.iloc[0]

    st.success(
        f"{selected_product} performs best in "
        f"{best['Store_Location']} with "
        f"{best['Units_Sold']:,.0f} units sold."
    )


# ============================================================
# 19. ASK FORECASTIQ
# ============================================================

elif page == "Ask ForecastIQ":

    st.header("💬 Ask ForecastIQ")

    st.caption(
        "Choose a business question to get an answer from the historical data."
    )

    question = st.selectbox(
        "Business Question",
        [
            "Which product sells most in each location?",
            "Which location has the highest demand?",
            "Which product sells the most overall?",
            "Which product is growing fastest?",
            "Does discount increase sales?",
            "Which quarter performs best?",
            "What are the strongest demand drivers?",
            "Which products may need more inventory?"
        ]
    )

    st.divider()

    if question == "Which product sells most in each location?":

        result = (
            historical_df
            .groupby(["Store_Location", "Product_Name"])["Units_Sold"]
            .sum()
            .reset_index()
        )

        result = result.loc[
            result.groupby("Store_Location")["Units_Sold"].idxmax()
        ]

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

    elif question == "Which location has the highest demand?":

        result = (
            historical_df
            .groupby("Store_Location")["Units_Sold"]
            .sum()
            .sort_values(ascending=False)
        )

        st.success(
            f"Highest-demand location: **{result.index[0]}** "
            f"with {result.iloc[0]:,.0f} units."
        )

        st.bar_chart(result)

    elif question == "Which product sells the most overall?":

        result = (
            historical_df
            .groupby("Product_Name")["Units_Sold"]
            .sum()
            .sort_values(ascending=False)
        )

        st.success(
            f"Top product: **{result.index[0]}** "
            f"with {result.iloc[0]:,.0f} units."
        )

        st.bar_chart(result)

    elif question == "Which product is growing fastest?":

        # Recalculate here so this page works independently.
        product_weekly = (
            historical_df
            .assign(
                Week=historical_df["Date"]
                .dt.to_period("W")
                .apply(lambda x: x.start_time)
            )
            .groupby(["Week", "Product_Name"])["Units_Sold"]
            .sum()
            .reset_index()
        )

        growth_results = []

        for product in product_weekly["Product_Name"].unique():

            p = product_weekly[
                product_weekly["Product_Name"] == product
            ].sort_values("Week")

            if len(p) >= 4:

                first = p["Units_Sold"].head(
                    max(1, len(p) // 4)
                ).mean()

                last = p["Units_Sold"].tail(
                    max(1, len(p) // 4)
                ).mean()

                growth = (
                    ((last - first) / first) * 100
                    if first != 0
                    else np.nan
                )

                growth_results.append([product, growth])

        growth_df_ask = pd.DataFrame(
            growth_results,
            columns=["Product_Name", "Growth_%"]
        ).sort_values("Growth_%", ascending=False)

        if len(growth_df_ask) > 0:

            top_growth = growth_df_ask.iloc[0]

            st.success(
                f"Fastest-growing product: **{top_growth['Product_Name']}** "
                f"with estimated growth of "
                f"{top_growth['Growth_%']:.2f}%."
            )

            st.dataframe(
                growth_df_ask,
                use_container_width=True,
                hide_index=True
            )

    elif question == "Does discount increase sales?":

        discount_corr = historical_df[
            ["Discount_Percentage", "Units_Sold"]
        ].corr().iloc[0, 1]

        st.metric(
            "Discount vs Units Sold Correlation",
            f"{discount_corr:.3f}"
        )

        if discount_corr > 0:
            st.info(
                "The historical data shows a positive association "
                "between discount and units sold."
            )
        elif discount_corr < 0:
            st.info(
                "The historical data shows a negative association "
                "between discount and units sold."
            )
        else:
            st.info(
                "The historical data shows almost no linear association."
            )

        st.info(
            "Correlation does not prove that discounts cause higher sales."
        )

    elif question == "Which quarter performs best?":

        result = (
            historical_df
            .groupby("Quarter")["Units_Sold"]
            .sum()
            .sort_values(ascending=False)
        )

        st.success(
            f"Best quarter: **{result.index[0]}** "
            f"with {result.iloc[0]:,.0f} units."
        )

        st.bar_chart(result)

    elif question == "What are the strongest demand drivers?":

        drivers = [
            "Price",
            "Discount_Percentage",
            "Marketing_Spend",
            "Competitor_Price",
            "Revenue",
            "Economic_Indicator"
        ]

        corr = (
            historical_df[drivers + ["Units_Sold"]]
            .corr()["Units_Sold"]
            .drop("Units_Sold")
            .sort_values(
                key=lambda x: x.abs(),
                ascending=False
            )
        )

        st.dataframe(
            corr.to_frame("Correlation with Units Sold"),
            use_container_width=True
        )

        st.info("These are associations, not causal effects.")

    elif question == "Which products may need more inventory?":

        inventory = (
            historical_df
            .groupby("Product_Name")
            .agg(
                Units_Sold=("Units_Sold", "sum"),
                Avg_Stock_Availability=(
                    "Stock_Availability",
                    "mean"
                )
            )
            .reset_index()
        )

        inventory_opportunity = inventory[
            inventory["Avg_Stock_Availability"] < 0.80
        ].sort_values(
            "Units_Sold",
            ascending=False
        )

        if len(inventory_opportunity) > 0:

            st.dataframe(
                inventory_opportunity,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No product meets the current low-stock threshold."
            )


# ============================================================
# 20. DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("🗃️ Data Explorer")

    st.caption(
        "Filter historical sales data and download the result."
    )

    filtered_df = historical_df.copy()

    locations = st.multiselect(
        "Location",
        sorted(
            historical_df["Store_Location"]
            .dropna()
            .unique()
        )
    )

    if locations:
        filtered_df = filtered_df[
            filtered_df["Store_Location"].isin(locations)
        ]

    products = st.multiselect(
        "Product",
        sorted(
            historical_df["Product_Name"]
            .dropna()
            .unique()
        )
    )

    if products:
        filtered_df = filtered_df[
            filtered_df["Product_Name"].isin(products)
        ]

    categories = st.multiselect(
        "Category",
        sorted(
            historical_df["Category"]
            .dropna()
            .unique()
        )
    )

    if categories:
        filtered_df = filtered_df[
            filtered_df["Category"].isin(categories)
        ]

    channels = st.multiselect(
        "Sales Channel",
        sorted(
            historical_df["Sales_Channel"]
            .dropna()
            .unique()
        )
    )

    if channels:
        filtered_df = filtered_df[
            filtered_df["Sales_Channel"].isin(channels)
        ]

    segments = st.multiselect(
        "Customer Segment",
        sorted(
            historical_df["Customer_Segment"]
            .dropna()
            .unique()
        )
    )

    if segments:
        filtered_df = filtered_df[
            filtered_df["Customer_Segment"].isin(segments)
        ]

    st.divider()

    st.subheader(
        f"Filtered Records: {len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )

    st.divider()

    st.subheader("Downloads")

    c1, c2 = st.columns(2)

    with c1:
        download_csv(
            filtered_df,
            "ForecastIQ_Filtered_Data.csv"
        )

    with c2:
        download_excel(
            filtered_df,
            "ForecastIQ_Filtered_Data.xlsx"
        )


# ============================================================
# 21. NEW PREDICTION
# ============================================================

elif page == "New Prediction":

    st.header("🔮 New Prediction")

    st.caption(
        "Upload a new dataset for validation and inspection "
        "before using the existing forecasting pipeline."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload New Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("📊 Dataset Overview")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", f"{new_data.shape[0]:,}")

        with c2:
            st.metric("Columns", new_data.shape[1])

        with c3:
            st.metric(
                "Missing Values",
                int(new_data.isnull().sum().sum())
            )

        st.dataframe(
            new_data.head(10),
            use_container_width=True
        )

        st.subheader("🔍 Data Quality Check")

        q1, q2 = st.columns(2)

        with q1:
            st.write(
                "Duplicate Rows:",
                int(new_data.duplicated().sum())
            )

        with q2:
            st.write(
                "Missing Values:",
                int(new_data.isnull().sum().sum())
            )

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column": new_data.columns,
            "Data Type": new_data.dtypes.astype(str).values,
            "Missing Values": new_data.isnull().sum().values,
            "Unique Values": [
                new_data[col].nunique()
                for col in new_data.columns
            ]
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        st.subheader("⚙️ Preprocessing Status")

        st.info(
            "Dataset validation completed. "
            "The uploaded data has been inspected. "
            "The production forecast shown in Forecast Intelligence "
            "uses the already completed Trial 84 XGBoost forecasting pipeline."
        )


# ============================================================
# END OF APPLICATION
# ============================================================

st.markdown("---")

st.caption(
    "ForecastIQ • Daily Sales Forecasting using Trial 84 Log-XGBoost | "
    "Historical: 06-Jan-2023 to 10-Sep-2026 | "
    "Forecast: 11-Sep-2026 to 31-Dec-2027"
)
