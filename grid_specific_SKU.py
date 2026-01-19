import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from geopy.distance import geodesic

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Jumbo Tour Planner v10")

# Constants for the Grid System
BOUNDS = {
    "TOP_LAT": 13.35,
    "LEFT_LON": 77.25,
    "BOTTOM_LAT": 12.65,
    "RIGHT_LON": 78.00
}
GRID_ROWS = 7
GRID_COLS = 7

# --- 2. DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Homes.csv")
        
        # 1. Coordinate Cleaning
        df['Building/Lat'] = pd.to_numeric(df['Building/Lat'], errors='coerce')
        df['Building/Long'] = pd.to_numeric(df['Building/Long'], errors='coerce')
        df = df.dropna(subset=['Building/Lat', 'Building/Long'])
        
        # 2. Status Cleaning & Mapping
        status_map = {
            '✅ Live': 'Live', 
            'Live': 'Live',
            'Inspection Pending': 'Inspection Pending',
            'Catalogue Pending': 'Catalogue Pending'
        }
        if 'Internal/Status' in df.columns:
            df['Clean_Status'] = df['Internal/Status'].map(status_map).fillna('Other')
        else:
            df['Clean_Status'] = 'Live' 
            
        # 3. Price Cleaning
        def clean_price(val):
            try:
                # Remove 'L', commas, and spaces, then convert to float
                return float(str(val).replace('L', '').replace(',', '').strip())
            except:
                return 0.0
        
        col_price = 'Home/Ask_Price (lacs)' if 'Home/Ask_Price (lacs)' in df.columns else 'Clean_Price'
        if col_price in df.columns:
            df['Clean_Price'] = df[col_price].apply(clean_price)
        else:
            df['Clean_Price'] = 0.0

        # 4. Config Cleaning (Extract number from "2 BHK")
        df['BHK_Num'] = df['Home/Configuration'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        # 5. Locality Handling (Fallback to Building Name if Locality missing)
        if 'Locality' not in df.columns:
            if 'Building/Locality' in df.columns:
                df['Locality'] = df['Building/Locality']
            else:
                df['Locality'] = df['Building/Name'] # Fallback

        return df
        
    except FileNotFoundError:
        return None

# --- 3. GRID CLASS (Logic + HTML Popup) ---
class OpsGrid:
    def __init__(self, grid_id, min_lat, min_lon, max_lat, max_lon, level=1):
        self.id = grid_id
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.level = level
        self.total_supply = 0
        self.stats = {} # Holds the breakdown data
        self.df_subset = pd.DataFrame() 

    def calculate_stats(self, df_subset, prem_threshold=100):
        self.df_subset = df_subset
        self.total_supply = len(df_subset)
        
        if self.total_supply > 0:
            # Logic Helpers
            unique_bldgs = df_subset['Building/Name'].nunique()
            is_2bhk = df_subset['BHK_Num'] == 2
            is_3bhk = df_subset['BHK_Num'] == 3
            is_tail = ~df_subset['BHK_Num'].isin([2, 3]) # Anything not 2 or 3
            is_prem = df_subset['Clean_Price'] > prem_threshold
            is_budg = df_subset['Clean_Price'] <= prem_threshold

            # Populate Stats Dictionary
            self.stats = {
                'Buildings': unique_bldgs,
                '2BHK_Budg': len(df_subset[is_2bhk & is_budg]),
                '2BHK_Prem': len(df_subset[is_2bhk & is_prem]),
                '3BHK_Budg': len(df_subset[is_3bhk & is_budg]),
                '3BHK_Prem': len(df_subset[is_3bhk & is_prem]),
                'Tail': len(df_subset[is_tail])
            }
        
        return self.total_supply

    def split(self):
        mid_lat = (self.min_lat + self.max_lat) / 2
        mid_lon = (self.min_lon + self.max_lon) / 2
        
        nw = OpsGrid(f"{self.id}-NW", mid_lat, self.min_lon, self.max_lat, mid_lon, self.level + 1)
        ne = OpsGrid(f"{self.id}-NE", mid_lat, mid_lon, self.max_lat, self.max_lon, self.level + 1)
        sw = OpsGrid(f"{self.id}-SW", self.min_lat, self.min_lon, mid_lat, mid_lon, self.level + 1)
        se = OpsGrid(f"{self.id}-SE", self.min_lat, mid_lon, mid_lat, self.max_lon, self.level + 1)
        
        return [nw, ne, sw, se]

# --- 4. CACHED GRID CALCULATOR ---
# This runs only when Filters or Thresholds change. Not on Zoom.
@st.cache_data
def process_grids(df, threshold, prem_val):
    # A. Generate Base 7x7 Grids
    grids = []
    lat_step = (BOUNDS["TOP_LAT"] - BOUNDS["BOTTOM_LAT"]) / GRID_ROWS
    lon_step = (BOUNDS["RIGHT_LON"] - BOUNDS["LEFT_LON"]) / GRID_COLS
    x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    for r in range(GRID_ROWS): 
        for c in range(GRID_COLS):
            g_max_lat = BOUNDS["TOP_LAT"] - (r * lat_step)
            g_min_lat = g_max_lat - lat_step
            g_min_lon = BOUNDS["LEFT_LON"] + (c * lon_step)
            g_max_lon = g_min_lon + lon_step
            grid_id = f"JB-{x_labels[c]}{str(r+1).zfill(2)}"
            grids.append(OpsGrid(grid_id, g_min_lat, g_min_lon, g_max_lat, g_max_lon, level=1))

    # B. Recursive Processing
    final_output = []
    queue = grids
    
    while queue:
        grid = queue.pop(0)
        
        # Filter Data for this specific grid box
        mask = (
            (df['Building/Lat'] >= grid.min_lat) & (df['Building/Lat'] < grid.max_lat) &
            (df['Building/Long'] >= grid.min_lon) & (df['Building/Long'] < grid.max_lon)
        )
        subset = df[mask]
        
        # Calculate Stats (Pass the Premium Threshold here!)
        count = grid.calculate_stats(subset, prem_threshold=prem_val)
        
        if count > threshold:
            # If too many homes, split it
            children = grid.split()
            queue.extend(children)
        else:
            # Add to final list
            final_output.append(grid)
            
    return final_output

# --- 5. CACHED MAP GENERATOR ---
# This generates the Folium object. Only reruns if grids change.
@st.cache_resource
def generate_map(_grids, center, zoom):
    m = folium.Map(location=center, zoom_start=zoom, prefer_canvas=True)
    
    for g in _grids:
        if g.total_supply == 0: continue

        # Visual Styling
        if g.level == 1: col, op = "#333", 0.05
        elif g.level == 2: col, op = "#ff9800", 0.15
        else: col, op = "#d32f2f", 0.25
        
        # --- POPUP HTML GENERATION ---
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 11px; width: 180px;">
            <b style="font-size: 12px;">{g.id}</b><br>
            <span style="color: gray;">Total Homes: {g.total_supply}</span>
            <hr style="margin: 5px 0;">
            <b>🏢 Buildings: {g.stats.get('Buildings', 0)}</b>
            <table style="width:100%; margin-top:5px; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #ddd; background: #f0f0f0;">
                    <th style="text-align:left; padding:2px;">Type</th>
                    <th style="text-align:center; padding:2px;">Budg</th>
                    <th style="text-align:center; padding:2px;">Prem</th>
                </tr>
                <tr>
                    <td style="padding:2px;"><b>2 BHK</b></td>
                    <td style="text-align:center; background:#e6ffe6;">{g.stats.get('2BHK_Budg', 0)}</td>
                    <td style="text-align:center; background:#fff0f0;">{g.stats.get('2BHK_Prem', 0)}</td>
                </tr>
                <tr>
                    <td style="padding:2px;"><b>3 BHK</b></td>
                    <td style="text-align:center; background:#e6ffe6;">{g.stats.get('3BHK_Budg', 0)}</td>
                    <td style="text-align:center; background:#fff0f0;">{g.stats.get('3BHK_Prem', 0)}</td>
                </tr>
                 <tr>
                    <td style="padding:2px;"><b>Tail</b></td>
                    <td colspan="2" style="text-align:center; color:#666;">{g.stats.get('Tail', 0)}</td>
                </tr>
            </table>
        </div>
        """

        folium.Rectangle(
            bounds=[[g.min_lat, g.min_lon], [g.max_lat, g.max_lon]],
            color=col, weight=1, fill=True, fill_opacity=op,
            tooltip=f"{g.id} (Homes: {g.total_supply})",
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)
        
        # Grid Label
        folium.Marker(
            location=[(g.min_lat + g.max_lat)/2, (g.min_lon + g.max_lon)/2],
            icon=folium.DivIcon(html=f'<div style="font-size:8px; color:{col}; font-weight:bold;">{g.id}</div>')
        ).add_to(m)
    return m

# --- 6. MAIN APP LOGIC ---

df_homes = load_data()

if df_homes is None:
    st.error("❌ 'Homes.csv' not found. Please upload the file.")
    st.stop()

st.title("🗺️ Jumbo Homes: Interactive Matrix")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Thresholds
    split_threshold = st.number_input("Grid Split Threshold (Homes)", 10, 500, 50)
    prem_threshold = st.number_input("Premium Threshold (Lacs)", value=100, step=10, help="Prices ABOVE this are Premium. EQUAL or BELOW are Budget.")
    
    st.divider()
    
    # Status Filter
    st.subheader("📋 Status")
    status_options = ['Live', 'Inspection Pending', 'Catalogue Pending', 'Other']
    default_status = ['Live', 'Inspection Pending', 'Catalogue Pending']
    selected_status = st.multiselect("Select Status", status_options, default=default_status)
    
    # Price & Config Filters
    st.subheader("🔍 Filters")
    price_options = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500, 1000]
    c1, c2 = st.columns(2)
    with c1: min_price = st.selectbox("Min Price (L)", price_options, index=0)
    with c2: max_price = st.selectbox("Max Price (L)", price_options, index=len(price_options)-3)

    available_bhk = sorted([int(x) for x in df_homes['BHK_Num'].unique() if x > 0])
    selected_bhk = st.multiselect("Configuration (BHK)", options=available_bhk, default=available_bhk)

    # Metric Switch
    st.divider()
    metric_type = st.radio("📊 Matrix Metric", ["Show House Count", "Show Building Count"], index=0)

# --- APPLY FILTERS ---
filtered_df = df_homes[
    (df_homes['Clean_Price'] >= min_price) & 
    (df_homes['Clean_Price'] <= max_price) &
    (df_homes['BHK_Num'].isin(selected_bhk)) &
    (df_homes['Clean_Status'].isin(selected_status))
].copy()

# Apply Dynamic Budget/Premium Segment
filtered_df['Segment'] = filtered_df['Clean_Price'].apply(lambda x: 'Prem' if x > prem_threshold else 'Budg')

st.sidebar.info(f"Active Inventory: {len(filtered_df)}")

# --- PROCESS GRIDS ---
# This uses the CACHED function. 
# It passes 'prem_threshold' so if you change 100L -> 120L, it recalculates.
ops_grids = process_grids(filtered_df, split_threshold, prem_threshold)

# --- MAP RENDER ---
is_draw_mode = st.toggle("✨ Draw Mode (Polygon/Circle)", value=False)

start_loc = [12.9716, 77.5946]
zoom = 11

# Generate map (Cached)
m_static = generate_map(ops_grids, start_loc, zoom)

# Add Draw control dynamically
if is_draw_mode:
    draw = Draw(
        draw_options={'rectangle': True, 'circle': True, 'polyline': False, 'polygon': False, 'marker': False, 'circlemarker': False},
        edit_options={'edit': False}
    )
    draw.add_to(m_static)

output = st_folium(m_static, width="100%", height=500)

# --- DATA TABLE GENERATION ---
final_df_for_table = pd.DataFrame()
current_selection_name = "All Visible Grids"

# 1. Determine Data Source (Draw vs Grid)
if is_draw_mode and output and output['last_active_drawing']:
    drawing = output['last_active_drawing']
    geom_type = drawing['geometry']['type']
    
    if geom_type == 'Point': 
        center = drawing['geometry']['coordinates']
        radius_m = drawing['properties']['radius']
        # Simple distance check
        filtered_df['dist'] = filtered_df.apply(lambda row: geodesic((row['Building/Lat'], row['Building/Long']), (center[1], center[0])).meters, axis=1)
        final_df_for_table = filtered_df[filtered_df['dist'] <= radius_m].copy()
        current_selection_name = "Custom Circle Zone"
        
    elif geom_type == 'Polygon':
        coords = drawing['geometry']['coordinates'][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        mask = (
            (filtered_df['Building/Lat'] >= min(lats)) & (filtered_df['Building/Lat'] <= max(lats)) &
            (filtered_df['Building/Long'] >= min(lons)) & (filtered_df['Building/Long'] <= max(lons))
        )
        final_df_for_table = filtered_df[mask].copy()
        current_selection_name = "Custom Box Zone"
else:
    # Combine data from all active OpsGrids
    frames = []
    for g in ops_grids:
        if not g.df_subset.empty:
            sub = g.df_subset.copy()
            sub['Grid_ID'] = g.id
            frames.append(sub)
    if frames:
        final_df_for_table = pd.concat(frames)

# 2. Build the Matrix Table
st.divider()
st.markdown(f"### 📊 Inventory Matrix: {current_selection_name}")

if not final_df_for_table.empty:
    
    # Helper columns for Pivot
    final_df_for_table['Is_2BHK'] = final_df_for_table['BHK_Num'] == 2
    final_df_for_table['Is_3BHK'] = final_df_for_table['BHK_Num'] == 3
    final_df_for_table['Is_Tail'] = ~final_df_for_table['BHK_Num'].isin([2, 3])
    
    # Grouping Level
    group_cols = ['Grid_ID', 'Locality'] if 'Grid_ID' in final_df_for_table.columns else ['Locality']
    
    # Determine what we are counting (Buildings or Houses)
    if metric_type == "Show Building Count":
        agg_func = lambda x: x.nunique()
        val_col = 'Building/Name'
    else:
        agg_func = 'count'
        val_col = 'House_ID'
    
    def get_pivot_col(mask_col, seg_val=None):
        if seg_val:
            mask = (final_df_for_table[mask_col]) & (final_df_for_table['Segment'] == seg_val)
        else:
            mask = (final_df_for_table[mask_col])
        return final_df_for_table[mask].groupby(group_cols)[val_col].apply(agg_func)

    # Calculate Columns
    s_2bhk_budg = get_pivot_col('Is_2BHK', 'Budg')
    s_2bhk_prem = get_pivot_col('Is_2BHK', 'Prem')
    s_3bhk_budg = get_pivot_col('Is_3BHK', 'Budg')
    s_3bhk_prem = get_pivot_col('Is_3BHK', 'Prem')
    s_tail      = get_pivot_col('Is_Tail')
    
    # Unique Buildings is ALWAYS a building count, regardless of switch
    s_unique_bldgs = final_df_for_table.groupby(group_cols)['Building/Name'].nunique()

    # Combine
    display_df = pd.DataFrame({
        '2BHK (Budg)': s_2bhk_budg,
        '2BHK (Prem)': s_2bhk_prem,
        '3BHK (Budg)': s_3bhk_budg,
        '3BHK (Prem)': s_3bhk_prem,
        'Tail': s_tail,
        'Unique Bldgs': s_unique_bldgs
    }).fillna(0).astype(int)
    
    # Sort
    if 'Grid_ID' in final_df_for_table.columns:
        display_df = display_df.sort_index(level=0)

    # Display with merged index (Grid -> Locality)
    st.dataframe(
        display_df.style
        .background_gradient(cmap="Reds", subset=['Unique Bldgs'])
        .format("{:,}"),
        use_container_width=True,
        height=600
    )

else:
    st.warning("No data found for the current selection/filters.")
