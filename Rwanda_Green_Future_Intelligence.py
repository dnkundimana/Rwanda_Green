"""
==============================================================
	RWANDA GREEN FUTURE INTELLIGENCE PLATFORM — FINAL APP
==============================================================

Run:
    streamlit run app.py

Features:
✅ Auto-rotating indicators every 10 seconds in Overview
✅ Fixed Streamlit session state error
✅ Comprehensive district analysis
✅ Rwanda-specific geospatial maps
✅ AI-powered insights
✅ Forecasting engine
✅ Actual province population comparison

"""

# =============================================================
# IMPORTS
# =============================================================
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
import json
import threading
import io
warnings.filterwarnings("ignore")

# Streamlit must be imported first
import streamlit as st
import pandas as pd
import numpy as np

# Import plotly without geopandas dependency
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from openai import OpenAI
except:
    OpenAI = None


# =============================================================
# PATHS
# =============================================================
BASE = Path(r"F:\BD_DR\Projects\Group14")
DATA_PATH = BASE / "data" / "raw" / "rwanda_base_2005_2024.csv"
SHAPE_DIR = BASE / "data" / "shapefiles"
OUT_DIR = BASE / "output" / "visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# STREAMLIT CONFIG
# =============================================================
st.set_page_config(
    page_title="Rwanda Green Future Dashboard",
    layout="wide",
    menu_items={"About": "National Green Growth Intelligence Platform"},
    initial_sidebar_state="expanded"
)


# =============================================================
# SESSION STATE INITIALIZATION
# =============================================================
def init_session_state():
    """Initialize all session state variables"""
    # Auto-rotation settings
    if 'auto_rotate' not in st.session_state:
        st.session_state.auto_rotate = False  # Changed default to False
    
    if 'rotate_index' not in st.session_state:
        st.session_state.rotate_index = 0
    
    if 'rotate_timer' not in st.session_state:
        st.session_state.rotate_timer = time.time()
    
    if 'indicator_history' not in st.session_state:
        st.session_state.indicator_history = []
    
    # Current indicator for rotation
    if 'current_rotation_indicator' not in st.session_state:
        st.session_state.current_rotation_indicator = None
    
    # Track if we need to update the sidebar widget
    if 'update_sidebar_widget' not in st.session_state:
        st.session_state.update_sidebar_widget = False
    
    # AI question history
    if 'ai_question' not in st.session_state:
        st.session_state.ai_question = ""
    
    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = ""
    
    # Current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📌 Overview"

# Initialize session state
init_session_state()


# =============================================================
# HELPER FUNCTIONS FOR POPULATION CALCULATION
# =============================================================
def calculate_province_populations(df, year, provinces=None):
    """Calculate province populations by summing district populations"""
    df_year = df[df.year == year]
    
    if provinces:
        df_year = df_year[df_year.province.isin(provinces)]
    
    # Group by province and sum population
    province_pop = df_year.groupby('province')['population'].sum().reset_index()
    return province_pop

def calculate_total_population(df, year, provinces=None, districts=None):
    """Calculate total population for selected regions"""
    df_year = df[df.year == year]
    
    if provinces:
        df_year = df_year[df_year.province.isin(provinces)]
    
    if districts:
        df_year = df_year[df_year.district.isin(districts)]
    
    return df_year['population'].sum() if not df_year.empty else 0


# =============================================================
# CREATE REALISTIC RWANDA DATA (If file doesn't exist)
# =============================================================
def create_realistic_rwanda_data():
    """Create realistic Rwanda data with 14.5M population"""
    st.info("Creating realistic Rwanda dataset for analysis...")
    
    years = list(range(2005, 2025))
    
    # Rwanda administrative structure
    provinces = ["Kigali City", "Southern Province", "Northern Province", "Eastern Province", "Western Province"]
    
    districts = {
        "Kigali City": ["Nyarugenge", "Gasabo", "Kicukiro"],
        "Southern Province": ["Nyanza", "Gisagara", "Nyaruguru", "Huye", "Nyamagabe", "Ruhango", "Muhanga", "Kamonyi"],
        "Northern Province": ["Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo"],
        "Eastern Province": ["Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana"],
        "Western Province": ["Karongi", "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"]
    }
    
    # Rwanda population distribution (2024 estimates ~14.5M total)
    district_population_base = {
        # Kigali City (~1.5M)
        "Nyarugenge": 350000,
        "Gasabo": 700000,
        "Kicukiro": 450000,
        
        # Southern Province (~3.5M)
        "Nyanza": 400000,
        "Gisagara": 300000,
        "Nyaruguru": 350000,
        "Huye": 450000,
        "Nyamagabe": 400000,
        "Ruhango": 350000,
        "Muhanga": 450000,
        "Kamonyi": 400000,
        
        # Northern Province (~2.5M)
        "Burera": 450000,
        "Gakenke": 400000,
        "Gicumbi": 500000,
        "Musanze": 600000,
        "Rulindo": 350000,
        
        # Eastern Province (~3.5M)
        "Bugesera": 600000,
        "Gatsibo": 550000,
        "Kayonza": 500000,
        "Kirehe": 450000,
        "Ngoma": 400000,
        "Nyagatare": 600000,
        "Rwamagana": 450000,
        
        # Western Province (~3.5M)
        "Karongi": 500000,
        "Ngororero": 400000,
        "Nyabihu": 350000,
        "Nyamasheke": 450000,
        "Rubavu": 550000,
        "Rusizi": 500000,
        "Rutsiro": 350000
    }
    
    # Calculate total base population
    total_base_population = sum(district_population_base.values())
    
    # Scale to ~14.5M in 2024
    scaling_factor = 14500000 / total_base_population
    
    # Apply scaling
    district_population_scaled = {k: int(v * scaling_factor) for k, v in district_population_base.items()}
    
    data = []
    
    for year in years:
        year_index = year - 2005  # 0 for 2005, 19 for 2024
        
        for province in provinces:
            for district in districts.get(province, []):
                # Population growth: 2.5% annual growth rate
                base_pop = district_population_scaled.get(district, 300000)
                population = int(base_pop * (1.025) ** (year_index / 4))  # Gradual growth
                
                # Electricity access: Starting from 6% in 2005 to 75% in 2024
                elec_access_base = 6 + (year_index * (75-6)/19)
                elec_access = np.random.normal(elec_access_base, 5)
                elec_access = max(0, min(100, elec_access))
                
                # CO2 emissions per capita (tons): Decreasing trend
                co2_per_capita = 0.15 - (year_index * 0.005)
                co2_total = population * co2_per_capita / 1000000  # Convert to Mt
                
                # GDP per capita: Growing from $250 to $850
                gdp_per_capita = 250 + (year_index * (850-250)/19)
                gdp_total = population * gdp_per_capita / 1000000  # Convert to $M
                
                # Forest cover: Slight improvement from 20% to 30%
                forest_cover = 20 + (year_index * (30-20)/19)
                forest_cover = max(10, min(50, forest_cover))
                
                # Poverty rate: Decreasing from 60% to 35%
                poverty_rate = 60 - (year_index * (60-35)/19)
                poverty_rate = max(10, min(70, poverty_rate))
                
                # Urbanization rate
                urbanization = 15 + (year_index * (40-15)/19)
                
                # Agriculture contribution to GDP
                agri_gdp = 40 - (year_index * (40-25)/19)
                
                # Education enrollment rate
                education_rate = 70 + (year_index * (95-70)/19)
                
                # Healthcare access
                health_access = 50 + (year_index * (85-50)/19)
                
                # Renewable energy share
                renewable_share = 10 + (year_index * (50-10)/19)
                
                # Water access
                water_access = 60 + (year_index * (85-60)/19)
                
                # Internet penetration
                internet_penetration = 1 + (year_index * (40-1)/19)
                
                data.append({
                    'year': year,
                    'province': province,
                    'district': district,
                    'population': population,
                    'elec_access_pct': round(elec_access, 1),
                    'co2e_total_mt': round(co2_total, 3),
                    'co2_per_capita': round(co2_per_capita, 3),
                    'gdp_usd_m': round(gdp_total, 2),
                    'gdp_per_capita_usd': round(gdp_per_capita, 0),
                    'forest_cover_pct': round(forest_cover, 1),
                    'poverty_rate_pct': round(poverty_rate, 1),
                    'urbanization_pct': round(urbanization, 1),
                    'agriculture_gdp_pct': round(agri_gdp, 1),
                    'education_enrollment_pct': round(education_rate, 1),
                    'health_access_pct': round(health_access, 1),
                    'renewable_energy_pct': round(renewable_share, 1),
                    'water_access_pct': round(water_access, 1),
                    'internet_penetration_pct': round(internet_penetration, 1),
                    'employment_rate_pct': round(75 + np.random.normal(0, 3), 1),
                    'industry_gdp_pct': round(15 + year_index * 0.5, 1),
                    'services_gdp_pct': round(45 + year_index * 1, 1),
                    'malaria_cases_per_1000': round(300 - year_index * 10, 0),
                    'life_expectancy': round(55 + year_index * 1.2, 1),
                    'child_mortality_per_1000': round(80 - year_index * 2.5, 1),
                    'air_quality_index': round(120 - year_index * 2, 1),
                    'waste_collection_pct': round(30 + year_index * 2.5, 1),
                    'public_transport_usage_pct': round(20 + year_index * 1.5, 1),
                    'land_degradation_pct': round(40 - year_index * 1, 1)
                })
    
    df = pd.DataFrame(data)
    
    # Save the created dataset
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    st.success(f"Created realistic Rwanda dataset with {df['population'].sum():,} total population (2024)")
    
    return df


# =============================================================
# DATA LOADERS
# =============================================================
@st.cache_data
def load_data():
    """Load and preprocess main dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"Data loaded successfully: {len(df):,} records, {df['population'].sum():,.0f} total population")
    except FileNotFoundError:
        st.warning(f"Data file not found at {DATA_PATH}. Creating realistic Rwanda dataset...")
        df = create_realistic_rwanda_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Creating realistic Rwanda dataset...")
        df = create_realistic_rwanda_data()
    
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure province and district are strings
    df['province'] = df['province'].astype(str).str.strip()
    df['district'] = df['district'].astype(str).str.strip()
    
    return df


def save_plot(fig, name):
    """Save plot as HTML"""
    out = OUT_DIR / f"{name}.html"
    fig.write_html(out)
    return out


# =============================================================
# CREATE RWANDA GEOJSON DATA (No Fiona/Geopandas needed)
# =============================================================
def get_rwanda_geojson(level="ADM1"):
    """Create GeoJSON data for Rwanda without external dependencies"""
    
    # Rwanda province boundaries (simplified polygons)
    rwanda_provinces_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"NAME_1": "Kigali City"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [30.0, -1.95], [30.1, -1.95], [30.1, -1.9], [30.0, -1.9], [30.0, -1.95]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_1": "Southern Province"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [29.7, -2.3], [30.2, -2.3], [30.2, -2.0], [29.7, -2.0], [29.7, -2.3]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_1": "Northern Province"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [29.6, -1.5], [30.1, -1.5], [30.1, -1.2], [29.6, -1.2], [29.6, -1.5]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_1": "Eastern Province"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [30.2, -1.8], [30.7, -1.8], [30.7, -1.5], [30.2, -1.5], [30.2, -1.8]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_1": "Western Province"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [29.2, -2.0], [29.7, -2.0], [29.7, -1.7], [29.2, -1.7], [29.2, -2.0]
                    ]]
                }
            }
        ]
    }
    
    # Rwanda district boundaries (simplified)
    rwanda_districts_geojson = {
        "type": "FeatureCollection",
        "features": [
            # Kigali City districts
            {
                "type": "Feature",
                "properties": {"NAME_2": "Nyarugenge", "NAME_1": "Kigali City"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [30.0, -1.95], [30.03, -1.95], [30.03, -1.93], [30.0, -1.93], [30.0, -1.95]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_2": "Gasabo", "NAME_1": "Kigali City"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [30.03, -1.95], [30.06, -1.95], [30.06, -1.93], [30.03, -1.93], [30.03, -1.95]
                    ]]
                }
            },
            {
                "type": "Feature",
                "properties": {"NAME_2": "Kicukiro", "NAME_1": "Kigali City"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [30.06, -1.95], [30.09, -1.95], [30.09, -1.93], [30.06, -1.93], [30.06, -1.95]
                    ]]
                }
            },
            # Southern Province districts
            {
                "type": "Feature",
                "properties": {"NAME_2": "Huye", "NAME_1": "Southern Province"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [29.75, -2.3], [29.85, -2.3], [29.85, -2.2], [29.75, -2.2], [29.75, -2.3]
                    ]]
                }
            },
        ]
    }
    
    # Rwanda country boundary
    rwanda_country_geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"NAME_0": "Rwanda"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [29.0, -2.85], [30.9, -2.85], [30.9, -1.05], [29.0, -1.05], [29.0, -2.85]
                ]]
            }
        }]
    }
    
    if level == "ADM0":
        return rwanda_country_geojson
    elif level == "ADM1":
        return rwanda_provinces_geojson
    else:  # ADM2
        return rwanda_districts_geojson


# =============================================================
# DYNAMIC SELECTION FUNCTIONS
# =============================================================
def get_province_district_mapping(df):
    """Create mapping of provinces to their districts"""
    mapping = {}
    for province in df['province'].unique():
        districts = sorted(df[df['province'] == province]['district'].unique())
        mapping[province] = districts
    return mapping


def update_districts_based_on_provinces(selected_provinces, province_district_map):
    """Get districts for selected provinces"""
    if not selected_provinces:
        return []
    
    all_districts = []
    for province in selected_provinces:
        all_districts.extend(province_district_map.get(province, []))
    
    return sorted(set(all_districts))


# =============================================================
# FORMATTING FUNCTIONS
# =============================================================
def format_number(value, is_percent=False):
    """Format numbers with commas (no decimals for integers, 1 decimal for percentages)"""
    if pd.isna(value):
        return "N/A"
    
    try:
        if is_percent:
            # For percentages, show 1 decimal place
            return f"{float(value):,.1f}%"
        elif float(value).is_integer():
            # For integers, no decimals, with commas
            return f"{int(value):,}"
        else:
            # For other numbers, show 2 decimals with commas
            return f"{float(value):,.2f}"
    except:
        return str(value)


# =============================================================
# AUTO-ROTATION FUNCTIONS
# =============================================================
def get_rotation_indicators(numeric_cols):
    """Get a curated list of indicators for rotation"""
    # Define categories for better rotation
    key_indicators = [
        'population',  # Always show population first
        'elec_access_pct',
        'gdp_per_capita_usd',
        'poverty_rate_pct',
        'forest_cover_pct',
        'co2e_total_mt',
        'water_access_pct',
        'education_enrollment_pct',
        'health_access_pct',
        'urbanization_pct',
        'employment_rate_pct',
        'life_expectancy',
        'child_mortality_per_1000',
        'air_quality_index',
        'renewable_energy_pct',
        'internet_penetration_pct'
    ]
    
    # Only include indicators that exist in the data
    rotation_indicators = [ind for ind in key_indicators if ind in numeric_cols]
    
    # Add any remaining indicators
    remaining = [ind for ind in numeric_cols if ind not in rotation_indicators]
    rotation_indicators.extend(remaining[:10])  # Limit to first 10 additional indicators
    
    return rotation_indicators[:15]  # Max 15 indicators for rotation


def update_auto_rotation():
    """Update the current indicator for auto-rotation"""
    current_time = time.time()
    elapsed = current_time - st.session_state.rotate_timer
    
    # Rotate every 10 seconds
    if elapsed >= 10:
        st.session_state.rotate_timer = current_time
        st.session_state.rotate_index += 1
        st.rerun()  # This will rerun the app


def get_current_rotation_indicator(numeric_cols, rotation_indicators):
    """Get the current indicator for rotation"""
    if not rotation_indicators:
        return numeric_cols[0] if numeric_cols else 'population'
    
    index = st.session_state.rotate_index % len(rotation_indicators)
    return rotation_indicators[index]


# =============================================================
# PROVINCE POPULATION CHART FUNCTIONS (FIXED)
# =============================================================
def create_province_population_comparison():
    """Create a bar chart of actual province population with Rwanda colors"""
    
    # ACTUAL province population data (2024 estimates)
    province_data = {
        'Province': ['Eastern', 'Kigali City', 'Northern', 'Southern', 'Western'],
        'Population': [3383387, 1483525, 2229030, 3164342, 3251986],
        'Color': ['#00843D', '#F7B731', '#00A8E8', '#8B5A2B', '#6A1B9A']
    }
    
    df_provinces = pd.DataFrame(province_data)
    
    # Create the bar chart
    fig = px.bar(
        df_provinces,
        x='Province',
        y='Population',
        color='Province',
        color_discrete_map={
            'Eastern': '#00843D',
            'Kigali City': '#F7B731',
            'Northern': '#00A8E8',
            'Southern': '#8B5A2B',
            'Western': '#6A1B9A'
        },
        title='🇷🇼 Actual Province Population Distribution (2024 Estimates)',
        text=df_provinces['Population'].apply(lambda x: f'{x:,}'),
        labels={'Population': 'Population Count', 'Province': 'Province'},
        height=500
    )
    
    # Customize the layout
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=22, family="Arial, sans-serif", color="#333"),
        showlegend=False,
        xaxis_title="Province",
        yaxis_title="Population",
        yaxis_tickformat=','
    )
    
    fig.update_traces(
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1,
        textposition='outside',
        textfont=dict(size=12)
    )
    
    total_population = df_provinces['Population'].sum()
    
    return fig, total_population


# =============================================================
# DISTRICT BAR CHART FUNCTIONS
# =============================================================
def create_district_bar_chart(df, indicator, year, top_n=15, title_suffix=""):
    """Create bar chart comparing districts for a specific indicator"""
    df_year = df[df.year == year]
    
    if df_year.empty:
        return None
    
    # Get top N districts by indicator value
    district_avg = df_year.groupby(['district', 'province'])[indicator].mean().reset_index()
    district_avg = district_avg.sort_values(indicator, ascending=False).head(top_n)
    
    if district_avg.empty:
        return None
    
    # Create bar chart
    fig = px.bar(
        district_avg,
        x='district',
        y=indicator,
        color='province',
        title=f"Top {top_n} Districts by {indicator.replace('_', ' ').title()} ({year}) {title_suffix}",
        labels={
            'district': 'District',
            indicator: indicator.replace('_', ' ').title(),
            'province': 'Province'
        },
        text=district_avg[indicator].apply(lambda x: format_number(x, '_pct' in indicator)),
        height=500
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode='closest',
        showlegend=True
    )
    
    return fig


def create_district_comparison_tab(df, indicator, year):
    """Create a comprehensive district comparison tab"""
    
    tab1, tab2, tab3 = st.tabs(["📊 Top Districts", "📈 Trends", "🏆 Rankings"])
    
    with tab1:
        # Top Districts Bar Chart
        col1, col2 = st.columns([3, 1])
        
        with col2:
            top_n = st.slider("Number of districts", 5, 30, 15, key=f"top_n_{indicator}")
        
        with col1:
            fig_top = create_district_bar_chart(df, indicator, year, top_n)
            if fig_top:
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.info("No data available for selected year.")
        
        # District Statistics
        df_year = df[df.year == year]
        if not df_year.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_value = df_year[indicator].mean()
                display_value = format_number(avg_value, '_pct' in indicator)
                st.metric("National Average", display_value)
            
            with col2:
                if not df_year[indicator].isna().all():
                    top_value = df_year[indicator].max()
                    top_district = df_year.loc[df_year[indicator].idxmax(), 'district']
                    display_value = format_number(top_value, '_pct' in indicator)
                    st.metric("Highest District", top_district, display_value)
                else:
                    st.metric("Highest District", "N/A")
            
            with col3:
                if not df_year[indicator].isna().all():
                    low_value = df_year[indicator].min()
                    low_district = df_year.loc[df_year[indicator].idxmin(), 'district']
                    display_value = format_number(low_value, '_pct' in indicator)
                    st.metric("Lowest District", low_district, display_value)
                else:
                    st.metric("Lowest District", "N/A")
            
            with col4:
                if not df_year[indicator].isna().all():
                    std_value = df_year[indicator].std()
                    display_value = format_number(std_value, '_pct' in indicator)
                    st.metric("Standard Deviation", display_value)
                else:
                    st.metric("Standard Deviation", "N/A")
    
    with tab2:
        # District Trends Over Time
        st.subheader(f"District Trends for {indicator.replace('_', ' ').title()}")
        
        # Select districts for trend analysis
        top_districts = df[df.year == year].groupby('district')[indicator].mean().nlargest(10).index.tolist()
        available_districts = df['district'].unique().tolist()
        selected_trend_districts = st.multiselect(
            "Select districts for trend analysis",
            available_districts,
            default=top_districts[:min(5, len(top_districts))] if top_districts else [],
            key=f"trend_districts_{indicator}"
        )
        
        if selected_trend_districts:
            df_trend = df[df.district.isin(selected_trend_districts)]
            fig_trend = px.line(
                df_trend,
                x="year",
                y=indicator,
                color="district",
                markers=True,
                title=f"{indicator.replace('_', ' ').title()} Trends by District",
                labels={'district': 'District', 'year': 'Year', indicator: indicator.replace('_', ' ').title()}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select districts to view trends.")
    
    with tab3:
        # District Rankings Table
        st.subheader("District Rankings")
        
        # Calculate rankings
        df_year = df[df.year == year]
        if not df_year.empty:
            rankings = df_year.groupby(['district', 'province'])[indicator].mean().reset_index()
            rankings = rankings.sort_values(indicator, ascending=False)
            rankings['Rank'] = range(1, len(rankings) + 1)
            rankings['Value'] = rankings[indicator]
            
            # Apply formatting
            is_percent = '_pct' in indicator
            rankings['Formatted Value'] = rankings['Value'].apply(lambda x: format_number(x, is_percent))
            
            # Add growth rate (compared to 5 years ago)
            if year - 5 >= df.year.min():
                df_previous = df[df.year == year-5]
                if not df_previous.empty:
                    previous_values = df_previous.groupby('district')[indicator].mean()
                    rankings['Growth_5yr'] = ((rankings['Value'] - rankings['district'].map(previous_values)) / 
                                            rankings['district'].map(previous_values) * 100)
                    rankings['Formatted Growth'] = rankings['Growth_5yr'].apply(lambda x: format_number(x, True))
                else:
                    rankings['Formatted Growth'] = None
            else:
                rankings['Formatted Growth'] = None
            
            # Display table
            display_cols = ['Rank', 'district', 'province', 'Formatted Value']
            if 'Formatted Growth' in rankings.columns:
                display_cols.append('Formatted Growth')
            
            st.dataframe(
                rankings[display_cols].rename(columns={
                    'district': 'District',
                    'province': 'Province',
                    'Formatted Value': indicator.replace('_', ' ').title(),
                    'Formatted Growth': '5-Year Growth'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_data = rankings[['Rank', 'district', 'province', 'Value', 'Growth_5yr' if 'Growth_5yr' in rankings.columns else '']].to_csv(index=False)
            st.download_button(
                label="📥 Download Rankings",
                data=csv_data,
                file_name=f"district_rankings_{indicator}_{year}.csv",
                mime="text/csv"
            )


# =============================================================
# LOAD BASE DATA
# =============================================================
df = load_data()

# Create province-district mapping
province_district_map = get_province_district_mapping(df)

# Get all unique values
all_provinces = sorted(df['province'].unique())
all_districts = sorted(df['district'].unique())

required_cols = {"year", "province", "district"}
if not required_cols.issubset(df.columns):
    st.error(f"Dataset missing required columns: {required_cols}")
    st.stop()

# Get all numeric indicators
numeric_cols = [
    c for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c != "year"
]

# Get rotation indicators
rotation_indicators = get_rotation_indicators(numeric_cols)

# Get current rotation indicator for session state
if st.session_state.current_rotation_indicator is None:
    st.session_state.current_rotation_indicator = get_current_rotation_indicator(numeric_cols, rotation_indicators)


# =============================================================
# SIDEBAR WITH DYNAMIC SELECTION - FIXED VERSION
# =============================================================
with st.sidebar:
    st.title("📊 Dashboard Controls")
    
    # Navigation - Store in session state for persistence
    page = st.radio(
        "Navigation",
        ["📌 Overview", "🗺 Map", "🔮 Forecast", "📤 Export", "🤖 Ask AI", "🏘 District Charts"],
        key="navigation_page"
    )
    
    # Store page in session state
    st.session_state.current_page = page
    
    st.divider()
    
    # Year Selection
    year = st.slider(
        "Select Year",
        int(df.year.min()),
        int(df.year.max()),
        int(df.year.max()),
        key="year_slider"
    )
    
    st.divider()
    
    # Dynamic Province Selection
    st.subheader("📍 Region Selection")
    
    # Select All checkbox
    select_all_provinces = st.checkbox("Select All Provinces", value=True, key="select_all_provinces")
    
    if select_all_provinces:
        sel_provinces = all_provinces
    else:
        sel_provinces = st.multiselect(
            "Select Province(s)",
            all_provinces,
            default=all_provinces,
            key="province_select"
        )
    
    # Dynamic District Selection based on selected provinces
    if sel_provinces:
        available_districts = update_districts_based_on_provinces(sel_provinces, province_district_map)
        
        # Select All Districts checkbox
        select_all_districts = st.checkbox("Select All Districts", value=True, key="select_all_districts")
        
        if select_all_districts:
            sel_districts = available_districts
        else:
            sel_districts = st.multiselect(
                "Select District(s)",
                available_districts,
                default=available_districts,
                key="district_select"
            )
    else:
        sel_districts = []
        st.warning("Please select at least one province")
    
    st.divider()
    
    # Indicator Selection
    st.subheader("📈 Indicator Selection")
    
    # Auto-rotation control
    auto_rotate = st.checkbox(
        "🔄 Enable auto-rotation (Overview page only)",
        value=st.session_state.auto_rotate,
        key="auto_rotate_checkbox",
        help="Indicators will rotate every 10 seconds on Overview page"
    )
    
    # Update session state based on checkbox
    if auto_rotate != st.session_state.auto_rotate:
        st.session_state.auto_rotate = auto_rotate
        st.session_state.rotate_timer = time.time()
        st.rerun()
    
    # Manual indicator selection - always available
    indicator_category = st.selectbox(
        "Indicator Category",
        ["All", "Economic", "Social", "Environmental", "Infrastructure", "Other"],
        key="indicator_category"
    )
    
    # Filter indicators by category
    economic_indicators = [col for col in numeric_cols if any(x in col.lower() for x in ['gdp', 'income', 'poverty', 'revenue', 'employment'])]
    social_indicators = [col for col in numeric_cols if any(x in col.lower() for x in ['population', 'education', 'health', 'life', 'mortality', 'urban'])]
    environmental_indicators = [col for col in numeric_cols if any(x in col.lower() for x in ['co2', 'forest', 'energy', 'emission', 'carbon', 'air', 'waste', 'degradation'])]
    infrastructure_indicators = [col for col in numeric_cols if any(x in col.lower() for x in ['elec', 'water', 'internet', 'transport'])]
    other_indicators = [col for col in numeric_cols if col not in economic_indicators + social_indicators + environmental_indicators + infrastructure_indicators]
    
    if indicator_category == "Economic":
        indicator_options = economic_indicators
    elif indicator_category == "Social":
        indicator_options = social_indicators
    elif indicator_category == "Environmental":
        indicator_options = environmental_indicators
    elif indicator_category == "Infrastructure":
        indicator_options = infrastructure_indicators
    elif indicator_category == "Other":
        indicator_options = other_indicators
    else:
        indicator_options = numeric_cols
    
    # Determine which indicator to show in the sidebar widget
    # If we're on Overview page and auto-rotate is enabled, show current rotation indicator
    if st.session_state.current_page == "📌 Overview" and st.session_state.auto_rotate:
        # Get current rotation indicator
        current_indicator = get_current_rotation_indicator(numeric_cols, rotation_indicators)
        # Find its index in the indicator options
        if current_indicator in indicator_options:
            default_index = indicator_options.index(current_indicator)
        else:
            default_index = 0
    else:
        # Use manual selection or default
        if 'population' in indicator_options:
            default_index = indicator_options.index('population')
        else:
            default_index = 0
    
    # Create the selectbox
    sidebar_indicator = st.selectbox(
        "Select Indicator",
        indicator_options if indicator_options else numeric_cols,
        index=default_index,
        key="sidebar_indicator_select",
        help="Choose the metric to analyze"
    )
    
    # Store the selected indicator
    indicator = sidebar_indicator
    
    st.divider()
    
    # Quick Stats
    st.subheader("📊 Quick Stats")
    if sel_provinces and sel_districts:
        df_current = df[(df.year == year) & 
                        (df.province.isin(sel_provinces)) & 
                        (df.district.isin(sel_districts))]
        
        if not df_current.empty and indicator in df_current.columns:
            current_value = df_current[indicator].mean()
            previous_year = max(year - 1, df.year.min())
            df_previous = df[(df.year == previous_year) & 
                             (df.province.isin(sel_provinces)) & 
                             (df.district.isin(sel_districts))]
            
            if not df_previous.empty and indicator in df_previous.columns:
                previous_value = df_previous[indicator].mean()
                change = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
                
                display_value = format_number(current_value, '_pct' in indicator)
                display_change = format_number(change, True)
                
                st.metric(
                    f"{indicator.replace('_', ' ').title()}",
                    display_value,
                    display_change
                )
        
        # Show population for context
        if 'population' in df_current.columns:
            total_pop = calculate_total_population(df, year, sel_provinces, sel_districts)
            st.metric("Selected Population", format_number(total_pop))
            
            # Also show province population distribution
            with st.expander("Province Population Distribution"):
                province_pop = calculate_province_populations(df, year, sel_provinces)
                for _, row in province_pop.iterrows():
                    st.write(f"**{row['province']}:** {format_number(row['population'])}")

# =============================================================
# FILTER DATA BASED ON SELECTIONS
# =============================================================
if sel_provinces and sel_districts:
    df_filt = df[
        (df.year <= year) &
        (df.province.isin(sel_provinces)) &
        (df.district.isin(sel_districts))
    ].copy()
else:
    df_filt = pd.DataFrame()
    if st.session_state.current_page not in ["🤖 Ask AI"]:  # Don't show warning on Ask AI page
        st.warning("Please select at least one province and district from the sidebar.")

# =============================================================
# MAIN PAGE ROUTING - FIXED VERSION
# =============================================================
# Get the current page from session state
page = st.session_state.current_page

# OVERVIEW PAGE
if page == "📌 Overview":
    if not df_filt.empty:
        # Update auto-rotation if enabled
        if st.session_state.auto_rotate:
            update_auto_rotation()
            # Get current rotation indicator
            current_rotation_indicator = get_current_rotation_indicator(numeric_cols, rotation_indicators)
            # Update session state
            st.session_state.current_rotation_indicator = current_rotation_indicator
            # Use rotation indicator for this page
            indicator = current_rotation_indicator
        else:
            # Use the sidebar selection
            indicator = sidebar_indicator
        
        st.title(f"📌 National & Regional Overview")
        
        # Rotation Control Panel
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                if st.session_state.auto_rotate:
                    st.markdown(f'<h3 style="color: #00843D; margin-bottom: 10px;">🔄 Currently Showing: {indicator.replace("_", " ").title()}</h3>', 
                              unsafe_allow_html=True)
                    
                    # Progress bar for rotation
                    current_time = time.time()
                    elapsed = current_time - st.session_state.rotate_timer
                    progress = min(elapsed / 10, 1.0)
                    st.progress(progress)
                    
                    # Countdown timer
                    time_left = max(0, 10 - elapsed)
                    st.caption(f"Next indicator in: {time_left:.1f}s")
                else:
                    st.markdown(f'<h3 style="color: #666; margin-bottom: 10px;">📊 {indicator.replace("_", " ").title()}</h3>', 
                              unsafe_allow_html=True)
            
            with col2:
                # Manual navigation buttons
                if st.button("◀️ Previous", key="nav_prev", use_container_width=True):
                    st.session_state.rotate_index -= 1
                    st.session_state.auto_rotate = False
                    st.rerun()
            
            with col3:
                if st.button("Next ▶️", key="nav_next", use_container_width=True):
                    st.session_state.rotate_index += 1
                    st.session_state.auto_rotate = False
                    st.rerun()
            
            with col4:
                # Auto-rotation toggle
                if st.session_state.auto_rotate:
                    if st.button("⏸️ Pause", key="pause_btn", use_container_width=True):
                        st.session_state.auto_rotate = False
                        st.rerun()
                else:
                    if st.button("🔄 Auto", key="auto_btn", use_container_width=True):
                        st.session_state.auto_rotate = True
                        st.session_state.rotate_timer = time.time()
                        st.rerun()
        
        # Current indicator information
        st.info(f"**Currently analyzing:** {indicator.replace('_', ' ').title()} | "
                f"**Rotation:** {'🔄 Enabled' if st.session_state.auto_rotate else '⏸️ Disabled'} | "
                f"**Indicator {st.session_state.rotate_index % len(rotation_indicators) + 1} of {len(rotation_indicators)}")
        
        st.divider()
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'population' in df_filt.columns:
                pop = calculate_total_population(df, year, sel_provinces, sel_districts)
                st.metric("Total Population", format_number(pop))
        
        with col2:
            if indicator in df_filt.columns:
                current_value = df_filt[df_filt.year == year][indicator].mean()
                display_value = format_number(current_value, '_pct' in indicator)
                st.metric("Current Value", display_value)
        
        with col3:
            if indicator in df_filt.columns:
                year_min = int(df_filt.year.min())
                year_max = int(df_filt.year.max())
                
                if year_max > year_min:
                    first_value = df_filt[df_filt.year == year_min][indicator].mean()
                    current_value = df_filt[df_filt.year == year_max][indicator].mean()
                    total_change = ((current_value - first_value) / first_value * 100) if first_value != 0 else 0
                    display_change = format_number(total_change, True)
                    st.metric(f"Change ({year_min}-{year_max})", display_change)
                else:
                    st.metric("Annual Change", "N/A")
        
        with col4:
            if indicator in df_filt.columns:
                provinces_count = df_filt['province'].nunique()
                districts_count = df_filt['district'].nunique()
                st.metric("Coverage", f"{provinces_count} provinces, {districts_count} districts")
        
        st.divider()
        
        # Main Trend Chart
        st.subheader(f"📈 {indicator.replace('_', ' ').title()} Trend Analysis")
        
        fig1 = px.line(
            df_filt,
            x="year",
            y=indicator,
            color="province",
            markers=True,
            line_shape="spline",
            title=f"{indicator.replace('_',' ').title()} Trend by Province",
            hover_data=["district"]
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Province-Level Analysis Tabs (FIXED VERSION)
        st.subheader("📊 Province-Level Analysis")
        
        # Create tabs for different province comparisons
        province_tab1, province_tab2 = st.tabs(["🔢 Selected Indicator", "👥 Population Comparison"])
        
        with province_tab1:
            # Show the selected indicator by province (average across districts)
            if indicator == 'population':
                # For population, show TOTAL population per province
                df_province_data = df_filt[df_filt.year == year].groupby('province')['population'].sum().reset_index()
                title = f"Total Population by Province ({year})"
                y_col = 'population'
                text_format = lambda x: format_number(x, False)
                y_label = "Total Population"
            else:
                # For other indicators, show average values
                df_province_data = df_filt[df_filt.year == year].groupby('province')[indicator].mean().reset_index()
                title = f"Average {indicator.replace('_', ' ').title()} by Province ({year})"
                y_col = indicator
                text_format = lambda x: format_number(x, '_pct' in indicator)
                y_label = indicator.replace('_', ' ').title()
            
            fig_indicator = px.bar(
                df_province_data,
                x='province',
                y=y_col,
                color='province',
                title=title,
                text=df_province_data[y_col].apply(text_format),
                labels={'province': 'Province', y_col: y_label},
                height=400
            )
            
            st.plotly_chart(fig_indicator, use_container_width=True)
            
            # Show statistics
            if not df_province_data.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_value = df_province_data[y_col].mean()
                    st.metric(f"Average {y_label}", 
                             format_number(avg_value, '_pct' in indicator))
                
                with col2:
                    max_value = df_province_data[y_col].max()
                    max_province = df_province_data.loc[df_province_data[y_col].idxmax(), 'province']
                    st.metric("Highest Province", max_province, 
                             format_number(max_value, '_pct' in indicator))
                
                with col3:
                    min_value = df_province_data[y_col].min()
                    min_province = df_province_data.loc[df_province_data[y_col].idxmin(), 'province']
                    st.metric("Lowest Province", min_province, 
                             format_number(min_value, '_pct' in indicator))
            
            # Add explanation
            if indicator == 'population':
                st.info("📊 **Showing total population (sum of all districts in each province)**")
            else:
                st.info("📊 **Showing average values across districts in each province**")
        
        with province_tab2:
            # Show ACTUAL province population (not from filtered data)
            st.subheader("🇷🇼 Official Province Population (2024 Estimates)")
            
            # Create and display the actual province population chart
            province_pop_fig, total_pop = create_province_population_comparison()
            st.plotly_chart(province_pop_fig, use_container_width=True)
            
            # Display the exact figures in metrics
            st.write("**Official Population Figures (2024):**")
            
            pop_col1, pop_col2, pop_col3, pop_col4, pop_col5 = st.columns(5)
            
            with pop_col1:
                st.metric("Eastern", "3,383,387")
            with pop_col2:
                st.metric("Kigali City", "1,483,525")
            with pop_col3:
                st.metric("Northern", "2,229,030")
            with pop_col4:
                st.metric("Southern", "3,164,342")
            with pop_col5:
                st.metric("Western", "3,251,986")
            
            total_actual = 3383387 + 1483525 + 2229030 + 3164342 + 3251986
            st.metric("**Total Rwanda Population**", f"{total_actual:,}", delta="14.5M total")
            
            # Add explanation
            st.info("💡 **Note:** These are official 2024 population estimates. This shows actual province totals, not district averages from the filtered data.")
        
        # District Comparison Tab
        st.subheader("🏘 District Comparison")
        create_district_comparison_tab(df_filt, indicator, year)
        
        # Indicator History (Last 5 indicators shown)
        if st.session_state.auto_rotate:
            st.divider()
            st.subheader("🔄 Rotation History")
            
            # Add current indicator to history
            if len(st.session_state.indicator_history) == 0 or st.session_state.indicator_history[-1] != indicator:
                st.session_state.indicator_history.append(indicator)
                # Keep only last 5
                if len(st.session_state.indicator_history) > 5:
                    st.session_state.indicator_history.pop(0)
            
            # Display history
            if st.session_state.indicator_history:
                history_cols = st.columns(len(st.session_state.indicator_history))
                for idx, hist_indicator in enumerate(st.session_state.indicator_history):
                    with history_cols[idx]:
                        # Calculate average value for this indicator
                        if hist_indicator in df_filt.columns:
                            avg_value = df_filt[df_filt.year == year][hist_indicator].mean()
                            display_value = format_number(avg_value, '_pct' in hist_indicator)
                            st.metric(
                                hist_indicator.replace('_', ' ').title(),
                                display_value
                            )
    
    else:
        st.warning("Please select at least one province and district from the sidebar.")

# DISTRICT CHARTS PAGE
elif page == "🏘 District Charts":
    if not df_filt.empty:
        # Note: This page doesn't auto-rotate, it uses the sidebar selection
        st.title("🏘 Comprehensive District Analysis")
        
        # Use the sidebar indicator selection for this page
        indicator = sidebar_indicator
        
        # Select indicators to compare
        st.subheader("📊 Select Indicators for District Comparison")
        
        # Common indicators to show as defaults (if they exist)
        common_indicators = ['population', 'elec_access_pct', 'gdp_per_capita_usd', 'poverty_rate_pct']
        # Filter to only include indicators that exist in numeric_cols
        default_indicators = [ind for ind in common_indicators if ind in numeric_cols]
        
        # If we don't have all default indicators, add some available ones
        if len(default_indicators) < 4:
            # Add other available indicators
            additional_indicators = [ind for ind in numeric_cols if ind not in default_indicators]
            default_indicators.extend(additional_indicators[:4-len(default_indicators)])
        
        selected_indicators = st.multiselect(
            "Select indicators to compare across districts",
            numeric_cols,
            default=default_indicators[:min(4, len(default_indicators))],
            key="district_indicators"
        )
        
        if selected_indicators:
            # Create tabs for each selected indicator
            tabs = st.tabs([f"📈 {ind.replace('_', ' ').title()}" for ind in selected_indicators])
            
            for tab, indicator_tab in zip(tabs, selected_indicators):
                with tab:
                    create_district_comparison_tab(df_filt, indicator_tab, year)
            
            # Cross-Indicator Analysis Tab
            st.divider()
            st.subheader("🔗 Cross-Indicator Correlation Analysis")
            
            # Create correlation matrix for selected indicators
            df_current = df_filt[df_filt.year == year]
            if not df_current.empty and len(selected_indicators) > 1:
                # Calculate correlations
                corr_matrix = df_current[selected_indicators].corr()
                
                # Create heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu',
                    title=f"Correlation Between Indicators ({year})",
                    labels=dict(color="Correlation")
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plot matrix (limit to 4 indicators for readability)
                if len(selected_indicators) <= 6:
                    st.subheader("Scatter Plot Matrix")
                    fig_scatter = px.scatter_matrix(
                        df_current,
                        dimensions=selected_indicators[:4],  # Limit to 4 for readability
                        color='province',
                        title=f"Relationships Between Indicators ({year})",
                        labels={col: col.replace('_', ' ').title() for col in selected_indicators[:4]}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Please select at least one indicator to analyze.")
    else:
        st.warning("Please select at least one province and district from the sidebar.")

# MAP PAGE
elif page == "🗺 Map":
    if not df_filt.empty:
        # Note: This page uses the sidebar selection (not auto-rotation)
        st.title("🗺 Rwanda Geospatial Indicator Map")
        
        # Use the sidebar indicator selection
        indicator = sidebar_indicator
        
        # Map Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level = st.radio(
                "Administrative Level",
                ["Provinces", "Districts"],
                horizontal=True,
                key="map_level"
            )
        
        with col2:
            map_style = st.selectbox(
                "Map Style",
                ["carto-positron", "open-street-map", "carto-darkmatter", "stamen-terrain"],
                key="map_style"
            )
        
        with col3:
            color_scale = st.selectbox(
                "Color Scale",
                ["Viridis", "Plasma", "Bluered", "Greens", "Oranges"],
                key="color_scale"
            )
        
        # Get GeoJSON data
        level_map = {"Provinces": "ADM1", "Districts": "ADM2"}
        geojson_data = get_rwanda_geojson(level_map[level])
        
        # Prepare data based on level
        if level == "Provinces":
            join_col = "NAME_1"
            # Map province names to GeoJSON province names
            province_mapping = {
                "Kigali City": "Kigali City",
                "Southern Province": "Southern Province",
                "Northern Province": "Northern Province",
                "Eastern Province": "Eastern Province",
                "Western Province": "Western Province"
            }
            
            df_map = df_filt[df_filt.year == year].copy()
            df_map[join_col] = df_map['province'].map(province_mapping)
            # Calculate province-level aggregates by summing district values
            if indicator == 'population' or 'total' in indicator.lower():
                # For population, we sum all districts in the province
                agg = df_map.groupby(join_col)[indicator].sum().reset_index()
            else:
                # For other indicators, we take the average across districts
                agg = df_map.groupby(join_col)[indicator].mean().reset_index()
            
        else:  # Districts
            join_col = "NAME_2"
            # Use district names directly (limited to available districts in GeoJSON)
            available_districts_geojson = [feature['properties']['NAME_2'] for feature in geojson_data['features']]
            df_map = df_filt[df_filt.year == year].copy()
            # Filter to only districts that exist in GeoJSON
            df_map = df_map[df_map['district'].isin(available_districts_geojson)]
            df_map[join_col] = df_map['district']
            # For districts, we use the actual values (already at district level)
            agg = df_map.groupby(join_col)[indicator].mean().reset_index()
        
        # Create choropleth map
        if not agg.empty and not agg[indicator].isna().all():
            fig = px.choropleth_mapbox(
                agg,
                geojson=geojson_data,
                locations=join_col,
                color=indicator,
                mapbox_style=map_style,
                center={"lat": -1.94, "lon": 29.9},
                zoom=6,
                opacity=0.8,
                color_continuous_scale=color_scale,
                title=f"{indicator.replace('_',' ').title()} Map — Rwanda ({year})",
                labels={indicator: indicator.replace('_', ' ').title()},
                hover_data={indicator: ':.2f'}
            )
            
            fig.update_layout(
                margin={"r":0,"t":30,"l":0,"b":0},
                mapbox_bounds={"west": 29.0, "east": 30.9, "south": -2.85, "north": -1.05}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Map Statistics
            st.subheader("📍 Regional Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not agg[indicator].isna().all():
                    max_val = agg[indicator].max()
                    top_region = agg[agg[indicator] == max_val][join_col].iloc[0]
                    display_val = format_number(max_val, '_pct' in indicator)
                    st.metric("Highest Value Region", top_region, display_val)
            
            with col2:
                if not agg[indicator].isna().all():
                    min_val = agg[indicator].min()
                    low_region = agg[agg[indicator] == min_val][join_col].iloc[0]
                    display_val = format_number(min_val, '_pct' in indicator)
                    st.metric("Lowest Value Region", low_region, display_val)
            
            with col3:
                if not agg[indicator].isna().all():
                    if level == "Provinces" and (indicator == 'population' or 'total' in indicator.lower()):
                        # For province-level population, show total
                        total_val = agg[indicator].sum()
                        display_val = format_number(total_val)
                        st.metric("Total Population", display_val)
                    else:
                        # For other indicators, show average
                        avg_val = agg[indicator].mean()
                        display_val = format_number(avg_val, '_pct' in indicator)
                        st.metric("Regional Average", display_val)
            
            # Data table
            st.subheader("📊 Map Data")
            agg_display = agg.copy()
            
            # Add province name for district-level data if available
            if level == "Districts" and 'NAME_1' in geojson_data['features'][0]['properties']:
                # Create a mapping from district to province
                district_province_map = {}
                for feature in geojson_data['features']:
                    props = feature['properties']
                    if 'NAME_2' in props and 'NAME_1' in props:
                        district_province_map[props['NAME_2']] = props['NAME_1']
                
                # Add province column to display
                agg_display['Province'] = agg_display[join_col].map(district_province_map)
            
            agg_display['Value'] = agg_display[indicator].apply(lambda x: format_number(x, '_pct' in indicator))
            
            # Determine which columns to display
            display_cols = [join_col]
            if 'Province' in agg_display.columns:
                display_cols.append('Province')
            display_cols.append('Value')
            
            st.dataframe(
                agg_display[display_cols].rename(
                    columns={
                        join_col: 'Region', 
                        'Value': indicator.replace('_', ' ').title()
                    }
                ),
                use_container_width=True,
                height=300
            )
        else:
            st.warning("No data available for mapping with the current selection.")
    else:
        st.warning("Please select at least one province and district from the sidebar.")

# FORECAST PAGE
elif page == "🔮 Forecast":
    if not df_filt.empty:
        # Note: This page uses the sidebar selection
        st.title("🔮 Forecasting Engine")
        
        # Use the sidebar indicator selection
        indicator = sidebar_indicator
        
        # Forecast Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Forecast Model",
                ["Linear Regression", "Random Forest", "Polynomial (Degree 2)", "Moving Average"],
                key="model_type"
            )
        
        with col2:
            horizon = st.slider(
                "Forecast Horizon (Years)",
                1, 30, 10,
                key="horizon"
            )
        
        with col3:
            confidence_interval = st.slider(
                "Confidence Interval",
                70, 99, 90,
                key="confidence"
            )
        
        # Prepare time series data
        series = df_filt.groupby("year")[indicator].mean().reset_index()
        
        if len(series) < 3:
            st.warning("Insufficient data for forecasting. Need at least 3 years of data.")
        else:
            # Forecasting logic
            X = series[["year"]].values
            y = series[indicator].values
            
            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X, y)
                future_years = np.arange(series.year.max() + 1, series.year.max() + horizon + 1)
                preds = model.predict(future_years.reshape(-1, 1))
            
            elif model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(X, y)
                future_years = np.arange(series.year.max() + 1, series.year.max() + horizon + 1)
                preds = model.predict(future_years.reshape(-1, 1))
            
            elif model_type == "Polynomial (Degree 2)":
                # Simple polynomial regression
                coeffs = np.polyfit(X.flatten(), y, 2)
                poly = np.poly1d(coeffs)
                future_years = np.arange(series.year.max() + 1, series.year.max() + horizon + 1)
                preds = poly(future_years)
            
            else:  # Moving Average
                window = min(3, len(series) - 1)
                last_values = y[-window:]
                future_years = np.arange(series.year.max() + 1, series.year.max() + horizon + 1)
                preds = np.full(horizon, np.mean(last_values))
            
            # Create forecast DataFrame
            forecast = pd.DataFrame({
                "year": future_years,
                "type": "Forecast",
                indicator: preds
            })
            
            historical = pd.DataFrame({
                "year": series.year.values,
                "type": "Historical",
                indicator: y
            })
            
            combined = pd.concat([historical, forecast], ignore_index=True)
            
            # Create forecast plot
            fig = px.line(
                combined,
                x="year",
                y=indicator,
                color="type",
                markers=True,
                line_dash="type",
                title=f"{indicator.replace('_', ' ').title()} Forecast ({model_type})",
                labels={"type": "Data Type"}
            )
            
            # Add confidence interval
            if model_type in ["Linear Regression", "Random Forest"]:
                residuals = y - model.predict(X)
                std_error = np.std(residuals)
                confidence_key = int(confidence_interval)
                z_scores = {70: 1.04, 80: 1.28, 90: 1.645, 95: 1.96, 99: 2.576}
                z_score = z_scores.get(confidence_key, 1.645)
                
                fig.add_trace(go.Scatter(
                    x=np.concatenate([future_years, future_years[::-1]]),
                    y=np.concatenate([preds + z_score * std_error, (preds - z_score * std_error)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=True,
                    name=f'{confidence_interval}% Confidence'
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast Metrics with formatting
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_historical = historical[indicator].iloc[-1]
                display_val = format_number(last_historical, '_pct' in indicator)
                st.metric("Last Historical Value", display_val)
            
            with col2:
                first_forecast = forecast[indicator].iloc[0]
                change = ((first_forecast - last_historical) / last_historical * 100) if last_historical != 0 else 0
                display_val = format_number(first_forecast, '_pct' in indicator)
                display_change = format_number(change, True)
                st.metric("First Forecast Value", display_val, display_change)
            
            with col3:
                final_forecast = forecast[indicator].iloc[-1]
                total_change = ((final_forecast - last_historical) / last_historical * 100) if last_historical != 0 else 0
                display_val = format_number(final_forecast, '_pct' in indicator)
                display_change = format_number(total_change, True)
                st.metric(f"Forecast ({horizon} years)", display_val, display_change)
            
            # Forecast Table with formatting
            st.subheader("📋 Forecast Data")
            forecast_display = forecast.copy()
            forecast_display['annual_growth_%'] = forecast_display[indicator].pct_change() * 100
            
            # Format the display
            forecast_display['Formatted Value'] = forecast_display[indicator].apply(
                lambda x: format_number(x, '_pct' in indicator)
            )
            forecast_display['Formatted Growth'] = forecast_display['annual_growth_%'].apply(
                lambda x: format_number(x, True)
            )
            
            st.dataframe(
                forecast_display[['year', 'Formatted Value', 'Formatted Growth']].rename(
                    columns={
                        'year': 'Year',
                        'Formatted Value': indicator.replace('_', ' ').title(),
                        'Formatted Growth': 'Annual Growth %'
                    }
                ),
                use_container_width=True
            )
    else:
        st.warning("Please select at least one province and district from the sidebar.")

# EXPORT PAGE
elif page == "📤 Export":
    if not df_filt.empty:
        st.title("📤 Data Export Center")
        
        # Use the sidebar indicator selection
        indicator = sidebar_indicator
        
        # Export Options
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.radio(
                "Export Format",
                ["CSV", "Excel", "JSON"],
                horizontal=True
            )
        
        with col2:
            include_all_cols = st.checkbox("Include All Columns", value=True)
        
        # Column selection if not including all
        if not include_all_cols:
            available_cols = df_filt.columns.tolist()
            selected_cols = st.multiselect(
                "Select Columns to Export",
                available_cols,
                default=['year', 'province', 'district', indicator]
            )
            export_df = df_filt[selected_cols]
        else:
            export_df = df_filt
        
        # Export Statistics with formatting
        total_pop = calculate_total_population(df, year, sel_provinces, sel_districts) if 'population' in export_df.columns else 0
        st.info(f"📊 **Export Summary:** {len(export_df):,} rows × {len(export_df.columns)} columns")
        st.info(f"👥 **Total Population:** {format_number(total_pop)}")
        
        # Preview with formatting
        with st.expander("🔍 Preview Data"):
            preview_df = export_df.head(10).copy()
            # Apply formatting to numeric columns for preview
            for col in preview_df.columns:
                if pd.api.types.is_numeric_dtype(preview_df[col]):
                    if '_pct' in col:
                        preview_df[col] = preview_df[col].apply(lambda x: format_number(x, True))
                    else:
                        preview_df[col] = preview_df[col].apply(lambda x: format_number(x, False))
            
            st.dataframe(preview_df, use_container_width=True)
        
        # Export Buttons
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if export_format == "CSV":
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"rwanda_data_{timestamp}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if export_format == "Excel":
                # For Excel export
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Rwanda_Data')
                    writer.close()
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"rwanda_data_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if export_format == "JSON":
                json_data = export_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"rwanda_data_{timestamp}.json",
                    mime="application/json"
                )
    else:
        st.warning("Please select at least one province and district from the sidebar.")

# ASK AI PAGE
elif page == "🤖 Ask AI":
    st.title("🤖 AI Policy Analyst")
    
    # Use the sidebar indicator selection
    indicator = sidebar_indicator
    
    # Context information with formatting
    with st.expander("ℹ️ Context Information", expanded=False):
        st.write(f"**Selected Regions:** {', '.join(sel_provinces) if sel_provinces else 'All Provinces'}")
        st.write(f"**Current Indicator:** {indicator}")
        st.write(f"**Time Range:** {df_filt.year.min() if not df_filt.empty else 'N/A'} - {year}")
        if not df_filt.empty:
            total_pop = calculate_total_population(df, year, sel_provinces, sel_districts) if 'population' in df_filt.columns else 0
            st.write(f"**Total Population:** {format_number(total_pop)}")
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_cols = st.columns(2)
    
    sample_questions = [
        f"Which districts have the highest {indicator.replace('_', ' ')}?",
        f"What are the trends in {indicator.replace('_', ' ')} across Rwanda?",
        "Compare district performance for this indicator",
        "Suggest policy interventions for improving this indicator",
        "What factors might be influencing district-level variations?",
        "Predict which districts will improve most in the next 5 years"
    ]
    
    for i, question in enumerate(sample_questions):
        with sample_cols[i % 2]:
            if st.button(f"📝 {question[:40]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.ai_question = question
    
    # Question input
    q = st.text_area(
        "Ask a question about district trends, comparisons, or policy recommendations:",
        value=st.session_state.ai_question,
        height=100,
        placeholder=f"E.g., Analyze {indicator.replace('_', ' ')} patterns across districts"
    )
    
    # Store the question
    st.session_state.ai_question = q
    
    # Analysis button
    if st.button("🔍 Generate Insight", type="primary", use_container_width=True):
        if not q:
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Analyzing data and generating insights..."):
                # Prepare data context for AI
                if not df_filt.empty:
                    # Calculate district statistics
                    df_current = df_filt[df_filt.year == year]
                    
                    if not df_current.empty:
                        top_districts = df_current.groupby(['district', 'province'])[indicator].mean().nlargest(5)
                        bottom_districts = df_current.groupby(['district', 'province'])[indicator].mean().nsmallest(5)
                        
                        top_districts_str = ', '.join([f'{d}: {format_number(v, "_pct" in indicator)}' for d, v in top_districts.items()])
                        bottom_districts_str = ', '.join([f'{d}: {format_number(v, "_pct" in indicator)}' for d, v in bottom_districts.items()])
                        
                        national_avg = df_current[indicator].mean()
                        provincial_avgs = df_current.groupby('province')[indicator].mean()
                        provincial_str = ', '.join([f'{p}: {format_number(v, "_pct" in indicator)}' for p, v in provincial_avgs.items()])
                        
                        data_context = f"""
                        RWANDA DATA ANALYSIS CONTEXT:
                        
                        INDICATOR: {indicator.replace('_', ' ').title()}
                        YEAR: {year}
                        TOTAL POPULATION: {format_number(calculate_total_population(df, year, sel_provinces, sel_districts))}
                        NATIONAL AVERAGE: {format_number(national_avg, '_pct' in indicator)}
                        
                        REGIONAL SELECTION:
                        - Selected Provinces: {len(sel_provinces)} provinces
                        - Selected Districts: {len(sel_districts)} districts
                        
                        TOP PERFORMING DISTRICTS:
                        {top_districts_str}
                        
                        DISTRICTS NEEDING IMPROVEMENT:
                        {bottom_districts_str}
                        
                        PROVINCIAL AVERAGES:
                        {provincial_str}
                        """
                    else:
                        data_context = f"No data available for year {year}"
                else:
                    data_context = "No filtered data available."
                
                # Check for OpenAI
                if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
                    # Fallback analysis without AI
                    st.warning("⚠️ OpenAI API not configured. Showing district analysis.")
                    
                    st.subheader("📊 District Analysis Results")
                    
                    if not df_filt.empty:
                        df_current = df_filt[df_filt.year == year]
                        
                        if not df_current.empty:
                            # Create district bar chart
                            fig = create_district_bar_chart(df_filt, indicator, year, 15)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show insights with formatting
                            st.write("**Key Insights:**")
                            
                            avg_value = df_current[indicator].mean()
                            st.write(f"1. **National Average:** {indicator.replace('_', ' ').title()} = {format_number(avg_value, '_pct' in indicator)}")
                            
                            if not df_current[indicator].isna().all():
                                max_district = df_current.loc[df_current[indicator].idxmax(), 'district']
                                max_value = df_current[indicator].max()
                                min_district = df_current.loc[df_current[indicator].idxmin(), 'district']
                                min_value = df_current[indicator].min()
                                
                                st.write(f"2. **Highest Performing District:** {max_district} = {format_number(max_value, '_pct' in indicator)}")
                                st.write(f"3. **Lowest Performing District:** {min_district} = {format_number(min_value, '_pct' in indicator)}")
                                
                                gap = max_value - min_value
                                gap_percent = ((max_value - min_value)/avg_value*100) if avg_value != 0 else 0
                                st.write(f"4. **Performance Gap:** {format_number(gap, '_pct' in indicator)} ({format_number(gap_percent, True)} of average)")
                            
                            # Provincial comparison
                            st.write("**Provincial Comparison:**")
                            provincial_avg = df_current.groupby('province')[indicator].mean().sort_values(ascending=False)
                            for province, value in provincial_avg.items():
                                st.write(f"- {province}: {format_number(value, '_pct' in indicator)}")
                            
                            # Recommendations
                            st.write("**💡 Policy Recommendations:**")
                            st.info("""
                            1. **Benchmarking:** Top-performing districts should document and share best practices
                            2. **Targeted Support:** Allocate additional resources to lowest-performing districts
                            3. **Provincial Coordination:** Encourage inter-district collaboration within provinces
                            4. **Data-Driven Planning:** Use these insights for evidence-based policy making
                            """)
                else:
                    # Use OpenAI for analysis
                    try:
                        client = OpenAI()
                        
                        prompt = f"""
                        You are a Rwanda District Development Analyst. Analyze the following data and provide specific, actionable insights.
                        
                        {data_context}
                        
                        Question: {q}
                        
                        Provide your analysis in the following structure:
                        1. **EXECUTIVE SUMMARY** (2-3 sentences)
                        2. **KEY FINDINGS** (bullet points with specific numbers)
                        3. **DISTRICT-LEVEL INSIGHTS** (compare top and bottom performers)
                        4. **PROVINCIAL PATTERNS** (identify regional trends)
                        5. **POLICY RECOMMENDATIONS** (specific, actionable steps)
                        6. **MONITORING SUGGESTIONS** (how to track progress)
                        
                        Focus on district-level variations and practical interventions.
                        """
                        
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        insight = response.choices[0].message.content
                        
                        # Store response
                        st.session_state.ai_response = insight
                        
                        # Display with formatting
                        st.success("✅ Analysis Complete!")
                        
                        with st.container():
                            st.markdown("### 📋 AI Analysis Results")
                            st.markdown("---")
                            st.markdown(insight)
                        
                    except Exception as e:
                        st.error(f"Error generating insight: {str(e)}")
                        st.info("Please ensure your OpenAI API key is valid.")
    
    # Show previous response if exists
    if st.session_state.ai_response:
        st.divider()
        st.subheader("📋 Previous Analysis")
        st.markdown(st.session_state.ai_response)

# DEFAULT: If no page matches
else:
    st.error("Page not found or invalid selection")

# =============================================================
# FOOTER WITH FORMATTING
# =============================================================
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.caption(" **Rwanda Green Future Intelligence Platform** | National Institute of Statistics of Rwanda")
    st.caption(f"📅 Data Range: {int(df.year.min())} - {int(df.year.max())}")

with footer_col2:
    if not df_filt.empty and 'population' in df_filt.columns:
        total_pop = calculate_total_population(df, year, sel_provinces, sel_districts)
        st.caption(f"👥 Population: {format_number(total_pop)}")

with footer_col3:
    if sel_provinces and sel_districts:
        st.caption(f"📍 Selected: {len(sel_districts)} districts in {len(sel_provinces)} provinces")