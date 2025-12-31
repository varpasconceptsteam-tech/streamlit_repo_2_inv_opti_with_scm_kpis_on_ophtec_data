import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Inventory Optimization Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        div[data-testid="metric-container"] {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .stMetricLabel {font-weight: bold; color: #555;}
        .stMetricValue {font-size: 1.5rem !important; color: #000;}
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    try:
        # Load SKU Level Analysis
        df_elw = pd.read_csv('ELW_stock_vs_target_analysis.csv')
        df_ocnh = pd.read_csv('OCNH_stock_vs_target_analysis.csv')
        df_sku = pd.concat([df_elw, df_ocnh], ignore_index=True)
        
        # Ensure Cylinder column exists
        if 'cylinder' not in df_sku.columns:
            df_sku['cylinder'] = 0.0

        # Load Layer 1 Forecast (Center Level)
        df_l1 = pd.read_csv('v2_layer1_center_monthly_forecast.csv')
        
        return df_sku, df_l1
    except FileNotFoundError:
        st.error("Critical Error: One or more data files not found. Please ensure 'ELW_stock_vs_target_analysis.csv', 'OCNH_stock_vs_target_analysis.csv', and 'v2_layer1_center_monthly_forecast.csv' are in the directory.")
        return pd.DataFrame(), pd.DataFrame()

df_sku, df_l1 = load_data()

if df_sku.empty:
    st.stop()

# --- 3. PRICING & FINANCIAL LOGIC ---
st.sidebar.markdown("## 💰 Pricing Assumptions")

# Default values for known models (used to pre-fill the inputs)
default_pricing = {
    'PC580': {'ASP': 140, 'COGS': 50},
    'PC545': {'ASP': 90, 'COGS': 30}
}

pricing_map = {}
# Get all unique models dynamically from the loaded data
unique_models = sorted(df_sku['model'].unique())

for model in unique_models:
    # Get defaults if available, otherwise set to 0
    defaults = default_pricing.get(model, {'ASP': 0, 'COGS': 0})
    
    st.sidebar.markdown(f"**{model} Pricing**")
    col_p1, col_p2 = st.sidebar.columns(2)
    
    with col_p1:
        asp = st.sidebar.number_input(
            f"Selling Price (€)", 
            value=int(defaults['ASP']), 
            key=f"asp_{model}",
            help=f"Average Selling Price for {model}"
        )
    with col_p2:
        cogs = st.sidebar.number_input(
            f"Cost Price (€)", 
            value=int(defaults['COGS']), 
            key=f"cogs_{model}",
            help=f"Cost of Goods Sold for {model}"
        )
    
    pricing_map[model] = {'ASP': asp, 'COGS': cogs}

def get_price(model, price_type):
    # Default fallback if model not found (0)
    return pricing_map.get(model, {'ASP': 0, 'COGS': 0}).get(price_type, 0)

# Apply Financials to SKU Dataframe (Recalculates instantly when inputs change)
df_sku['ASP'] = df_sku['model'].apply(lambda x: get_price(x, 'ASP'))
df_sku['COGS'] = df_sku['model'].apply(lambda x: get_price(x, 'COGS'))

df_sku['current_val'] = df_sku['current_stock'] * df_sku['COGS']
df_sku['optimized_val'] = df_sku['target_units'] * df_sku['COGS']
df_sku['lost_revenue'] = df_sku['less_stock'] * df_sku['ASP']


# --- 4. CUSTOM FILTER LOGIC (From presentation_app.py) ---

# --- HELPER: CALLBACK FOR STATE SYNC ---
def update_selection(ss_key, option):
    """Callback to update session state before the script reruns"""
    if option in st.session_state[ss_key]:
        st.session_state[ss_key].remove(option)
    else:
        st.session_state[ss_key].append(option)

# --- CUSTOM FILTER COMPONENT ---
def multiselect_dropdown(label, options, key_prefix, default_all=True):
    ss_key = f"{key_prefix}_selected"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = list(options) if default_all else []

    # Filter "Ghost" Selections (items not in current options)
    valid_selection = [x for x in st.session_state[ss_key] if x in options]
    
    selected_count = len(valid_selection)
    total_count = len(options)
    
    # Label Logic
    if selected_count == total_count:
        display_label = f"{label} (All)"
    elif selected_count == 0:
        display_label = f"{label} (None)"
    else:
        display_label = f"{label} ({selected_count})"

    with st.sidebar.popover(display_label, use_container_width=True):
        
        # Search
        search_query = st.text_input("Search", placeholder="Type to find...", key=f"{key_prefix}_search")

        # Select/Clear All
        c1, c2 = st.columns(2)
        if c1.button("Select All", key=f"{key_prefix}_all"):
            st.session_state[ss_key] = list(options)
            st.rerun()
        if c2.button("Clear All", key=f"{key_prefix}_clear"):
            st.session_state[ss_key] = []
            st.rerun()

        st.markdown("---")

        # List
        filtered_options = [opt for opt in options if str(search_query).lower() in str(opt).lower()]
        
        with st.container(height=200):
            for opt in filtered_options:
                is_checked = opt in st.session_state[ss_key]
                
                # Checkbox with callback for instant sync
                st.checkbox(
                    str(opt), 
                    value=is_checked, 
                    key=f"{key_prefix}_{opt}",
                    on_change=update_selection,
                    args=(ss_key, opt)
                )

    return st.session_state[ss_key]


# --- 5. SIDEBAR FILTER EXECUTION ---
st.sidebar.header("Filter Configuration")

# 1. CENTER (The Root Filter - Single Select as per uploaded code)
unique_centers = sorted(df_sku['center_id'].unique())
selected_center = st.sidebar.selectbox("Select Center", unique_centers, index=0)

# Apply Filter 1 immediately
df_step1 = df_sku[df_sku['center_id'] == selected_center]
df_l1_filtered = df_l1[df_l1['center_id'] == selected_center] # Filter Forecast file too

# Create a unique prefix based on the center
# This ensures ELW selections stay in ELW, and OCNH selections stay in OCNH
prefix = selected_center

# 2. STOCK CLASS (Dependent on Center)
available_classes = sorted(df_step1['stock_class'].unique())
selected_classes = multiselect_dropdown("Stock Class", available_classes, f"{prefix}_class")

if not selected_classes:
    st.warning("Please select at least one Stock Class to continue.")
    st.stop()

# Apply Filter 2
df_step2 = df_step1[df_step1['stock_class'].isin(selected_classes)]

# 3. IOL MODEL (Dependent on Center + Class)
available_models = sorted(df_step2['model'].unique())
selected_models = multiselect_dropdown("IOL Model", available_models, f"{prefix}_model")

if not selected_models:
    st.warning("Please select at least one Model.")
    st.stop()

# Apply Filter 3
df_step3 = df_step2[df_step2['model'].isin(selected_models)]

# 4. DIOPTER (Dependent on Center + Class + Model)
available_diopters = sorted(df_step3['diopter'].unique())
selected_diopters = multiselect_dropdown("Diopter", available_diopters, f"{prefix}_diop")

if not selected_diopters:
    st.warning("Please select at least one Diopter.")
    st.stop()

# Apply Filter 4
df_step4 = df_step3[df_step3['diopter'].isin(selected_diopters)]

# 5. CYLINDER (Dependent on everything above)
available_cylinders = sorted(df_step4['cylinder'].unique())
selected_cylinders = multiselect_dropdown("Cylinder", available_cylinders, f"{prefix}_cyl")

if not selected_cylinders:
    st.warning("Please select at least one Cylinder.")
    st.stop()

# Final Filter Application
df_final = df_step4[df_step4['cylinder'].isin(selected_cylinders)]


# --- 6. METRIC CALCULATIONS ---

# Basic Aggregations
total_current_stock = df_final['current_stock'].sum()
total_target_stock = df_final['target_units'].sum()
total_safety_stock = df_final['safety_stock'].sum()
deviation = total_current_stock - total_target_stock
total_excess = df_final['excess_stock'].sum()
total_shortage = df_final['less_stock'].sum()
forecast_sku_sum = df_final['forecasted_units'].sum()

# Active Diopters: Current > 0 OR Target > 0
df_final['is_active'] = (df_final['current_stock'] > 0) | (df_final['target_units'] > 0)
total_active_diopters = df_final['is_active'].sum()

# Healthy Diopters: Excess <= 1 AND Shortage <= 1
df_final['is_healthy'] = (df_final['excess_stock'] <= 1) & (df_final['less_stock'] <= 1) & df_final['is_active']
healthy_diopters_count = df_final['is_healthy'].sum()

# Health Percentage
health_pct = (healthy_diopters_count / total_active_diopters * 100) if total_active_diopters > 0 else 0

# Financial Aggregations
val_on_hand = df_final['current_val'].sum()
val_optimized = df_final['optimized_val'].sum()
val_lost_revenue = df_final['lost_revenue'].sum()

# Layer 1 Forecast Logic
# We only show the official Layer 1 Bounds if the user has NOT filtered out specific items
# Logic: If all options are selected in all steps (except center which is single select)
is_full_center_view = (
    len(selected_classes) == len(available_classes) and
    len(selected_models) == len(available_models) and
    len(selected_diopters) == len(available_diopters) and
    len(selected_cylinders) == len(available_cylinders)
)

if is_full_center_view:
    l1_mean = df_l1_filtered['forecast_mean'].sum()
    l1_low = df_l1_filtered['forecast_p2_5'].sum()
    l1_high = df_l1_filtered['forecast_p97_5'].sum()
    forecast_display = f"{forecast_sku_sum:,.0f}"
    forecast_delta = f"Bounds: {l1_low:.0f} - {l1_high:.0f}"
else:
    # If filtered, we just sum the SKU level forecasts
    forecast_display = f"{forecast_sku_sum:,.0f}"
    forecast_delta = "Aggregated SKU Forecast"


# --- 7. DASHBOARD LAYOUT ---

st.title("🏥 Inventory Optimization Dashboard")
st.markdown("Overview of stock health, financial efficiency, and forecast alignment.")

# Create the container
with st.expander("ℹ️ Data Parameters & Assumptions", expanded=True):
    p1, p2= st.columns(2)
    with p1:
        st.markdown(f"""
        **🗓️ Forecast Month** -> December 2025
        """)
    with p2:
        st.markdown("""
        **💶 Pricing Assumptions**  ->  PC580: ASP €140 , COGS €50   **|**   PC545: ASP €90 , COGS €30
        """)

# ROW 1: INVENTORY & FORECAST 
st.subheader("📦 Inventory Status")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Forecasted Demand", forecast_display, delta=forecast_delta, help="Expected usage for Dec 2025 month with 90% confidence interval")
with c2:
    st.metric("Target Stock", f"{total_target_stock:,}", delta=f"{total_safety_stock:+,} Safety Stock")
with c3:
    st.metric("Current Stock", f"{total_current_stock:,}", delta=f"{deviation:+,} Deviation")
with c4:
    st.metric("Excess Units", f"{total_excess:,}", delta_color="inverse")
with c5:
    st.metric("Shortage Units", f"{total_shortage:,}", delta_color="inverse")
with c6:
    st.metric(
        "Inventory Health", 
        f"{health_pct:.1f}%", 
        f"{healthy_diopters_count} / {total_active_diopters} Diopters",
        help="Inventory Health % = (Healthy Diopters / Total Active Diopters)\n"
             "\n• **Healthy Diopter**: Excess ≤ 1 AND Shortage ≤ 1\n"
             "\n• **Active Diopter**: Current Stock > 0 OR Target > 0"
    )

st.divider()

# ROW 2: FINANCIALS
st.subheader("💰 Financial Impact")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current On-Hand Value", f"€{val_on_hand:,.0f}", help="COGS * Current Stock")
with col2:
    st.metric("Optimized Value", f"€{val_optimized:,.0f}", delta=f"€{val_optimized - val_on_hand:,.0f}", help="COGS * Target Stock")
with col3:
    st.metric("Potential Lost Revenue", f"€{val_lost_revenue:,.0f}", delta_color="inverse", help="ASP * Shortage Units")

st.divider()

# --- 8. VISUALIZATIONS ---
from plotly.subplots import make_subplots # Required for the ribbon layout

st.subheader("Stock Level vs. Target Overview")

# 1. Prepare Data
# Group by Diopter to sum quantities, but keep the 'first' Stock Class found
chart_data = df_final.groupby('diopter').agg({
    'current_stock': 'sum',
    'target_units': 'sum',
    'excess_stock': 'sum',
    'less_stock': 'sum',
    'stock_class': 'first' 
}).reset_index()

# Sort by Diopter (Ascending)
chart_data = chart_data.sort_values('diopter').reset_index(drop=True)

# Calculate "Target Stock"
chart_data['target_stock'] = chart_data['current_stock'] - chart_data['excess_stock']

if not chart_data.empty:
    # --- DEFINE COLORS ---
    # Class Colors for the Ribbon
    class_colors = {
        'Tail/Backups': '#D3D3D3',      # Blue
        'Average': '#9933CC',         # Purple
        'Mainstream': '#3366CC'    # Light Grey
    }
    
    # Map the colors to the dataframe
    chart_data['ribbon_color'] = chart_data['stock_class'].map(class_colors).fillna('#D3D3D3')

    # --- CREATE SUBPLOTS ---
    # Row 1: Main Chart (0.85 height), Row 2: Ribbon (0.15 height)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.85, 0.15]
    )

    # --- ROW 1: INVENTORY BARS ---
    # 1. Target Stock (Green)
    fig.add_trace(go.Bar(
        x=chart_data['diopter'],
        y=chart_data['target_stock'],
        name='Target Stock',
        marker_color='#00CC96',
        hovertemplate='Diopter: %{x}<br>Target: %{y}<extra></extra>'
    ), row=1, col=1)

    # 2. Excess Stock (Red)
    fig.add_trace(go.Bar(
        x=chart_data['diopter'],
        y=chart_data['excess_stock'],
        name='Excess Stock',
        marker_color='#EF553B',
        hovertemplate='Diopter: %{x}<br>Excess: %{y}<extra></extra>'
    ), row=1, col=1)

    # 3. Shortage (Yellow)
    fig.add_trace(go.Bar(
        x=chart_data['diopter'],
        y=chart_data['less_stock'],
        name='Shortage / Gap',
        marker_color='#FECB52',
        hovertemplate='Diopter: %{x}<br>Shortage: %{y}<extra></extra>'
    ), row=1, col=1)

    # --- ROW 2: CLASS RIBBON ---
    # We add a bar chart with constant height (1) to create the "strip" effect
    fig.add_trace(go.Bar(
        x=chart_data['diopter'],
        y=[1] * len(chart_data), # Constant height for the ribbon
        marker_color=chart_data['ribbon_color'],
        name='Stock Class', # Generic name (legend handled below)
        showlegend=False,   # Hide this generic trace from legend
        hovertemplate='Stock Class: %{text}<extra></extra>',
        text=chart_data['stock_class']
    ), row=2, col=1)

    # --- LEGEND FIX ---
    # Add "Dummy" traces so the Legend explains the Ribbon colors
    for cls_name, cls_color in class_colors.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=f"Class: {cls_name}",
            marker_color=cls_color,
            showlegend=True
        ), row=1, col=1) # Row doesn't matter for dummy traces

    # --- LAYOUT CONFIGURATION ---
    chart_title = f"Inventory Balance: {selected_models[0]}" if len(selected_models) > 0 else "Inventory Balance"

    fig.update_layout(
        title=chart_title,
        height=600, # Increased height for the extra panel
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=80, b=20)
    )

    # Specific Axis Tweaks
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text="", row=2, col=1, fixedrange=True) # Hide Y axis for ribbon
    fig.update_xaxes(title_text="Diopter Power", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **How to read this chart:**
    - The **Total Solid Bar (Green + Red)** represents your **Current Physical Stock**.
    - **Green** is healthy stock (matching demand).
    - **Red** is Overstock (money sitting on the shelf).
    - **Yellow** is Lost Sales/Shortage (the gap you need to fill to hit Target).
    - **Bottom Ribbon:** Stock Class (Blue=Mainstream, Purple=Average, Grey=Tail).
    """)

st.divider()

# --- 9. RAW DATA EXPANDER ---
with st.expander("📂 View Detailed SKU Data"):
    st.dataframe(
        df_final[['model', 'diopter', 'stock_class', 
                  'forecasted_units', 'target_units', 'current_stock', 
                  'excess_stock', 'less_stock']].rename(columns={'model' : 'Model', 'diopter' : 'Diopter', 'stock_class': 'Stock Class', 'forecasted_units':'Forecasted Units', 'target_units':'Target Units', 'current_stock':'Current Stock', 'excess_stock':'Excess Units', 'less_stock':'Shortage Units'}).sort_values(by=['Stock Class','Diopter']),
        use_container_width=True
    )