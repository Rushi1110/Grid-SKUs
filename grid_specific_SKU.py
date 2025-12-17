import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Jumbo Territory Planner v3")

# --- 1. CONFIGURATION & CONSTANTS ---
BOUNDS = {
    "TOP_LAT": 13.35,
    "LEFT_LON": 77.25,
    "BOTTOM_LAT": 12.65,
    "RIGHT_LON": 78.00
}

GRID_ROWS = 7
GRID_COLS = 7

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Homes.csv")
        
        # 1. Coordinate Cleaning
        df['Building/Lat'] = pd.to_numeric(df['Building/Lat'], errors='coerce')
        df['Building/Long'] = pd.to_numeric(df['Building/Long'], errors='coerce')
        df = df.dropna(subset=['Building/Lat', 'Building/Long'])
        
        # 2. Status Cleaning - Normalize
        # We need specific statuses for the table later
        status_map = {
            '✅ Live': 'Live',
            'Live': 'Live',
            'Inspection Pending': 'Inspection Pending',
            'Catalogue Pending': 'Catalogue Pending'
        }
        if 'Internal/Status' in df.columns:
            df['Clean_Status'] = df['Internal/Status'].map(status_map).fillna('Other')
            # Filter for active only
            df = df[df['Clean_Status'].isin(['Live', 'Inspection Pending', 'Catalogue Pending'])]
        else:
            df['Clean_Status'] = 'Live' # Fallback
            
        # 3. Price Cleaning
        def clean_price(val):
            try:
                return float(str(val).replace('L', '').replace(',', '').strip())
            except:
                return 0.0
        
        col_price = 'Home/Ask_Price (lacs)' if 'Home/Ask_Price (lacs)' in df.columns else 'Clean_Price'
        if col_price in df.columns:
            df['Clean_Price'] = df[col_price].apply(clean_price)
        else:
            df['Clean_Price'] = 0.0

        # 4. Config Cleaning (2BHK, 3BHK)
        # Extract number from string
        df['BHK_Num'] = df['Home/Configuration'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ 'Homes.csv' not found. Please place it in the application folder.")
        return None

df_homes = load_data()

if df_homes is None:
    st.stop()

# --- 3. GRID CLASS ---

class OpsGrid:
    def __init__(self, grid_id, min_lat, min_lon, max_lat, max_lon, level=1):
        self.id = grid_id
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.level = level
        self.buildings = set()
        self.total_supply = 0
        self.children = []
        self.df_subset = pd.DataFrame() # Store reference to data

    def calculate_stats(self, df_subset):
        self.df_subset = df_subset
        if df_subset.empty:
            return 0
            
        if 'Building/Name' in df_subset.columns:
            self.buildings = set(df_subset['Building/Name'].unique())
        
        self.total_supply = len(df_subset)
        return len(self.buildings)

    def split(self):
        mid_lat = (self.min_lat + self.max_lat) / 2
        mid_lon = (self.min_lon + self.max_lon) / 2
        
        nw = OpsGrid(f"{self.id}-NW", mid_lat, self.min_lon, self.max_lat, mid_lon, self.level + 1)
        ne = OpsGrid(f"{self.id}-NE", mid_lat, mid_lon, self.max_lat, self.max_lon, self.level + 1)
        sw = OpsGrid(f"{self.id}-SW", self.min_lat, self.min_lon, mid_lat, mid_lon, self.level + 1)
        se = OpsGrid(f"{self.id}-SE", self.min_lat, mid_lon, mid_lat, self.max_lon, self.level + 1)
        
        self.children = [nw, ne, sw, se]
        return self.children

# --- 4. ALGORITHM ---

def generate_7x7_matrix():
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
    return grids

def process_grids(base_grids, df, threshold):
    final_output = []
    flat_list = [] # List of ALL grids (Parents + Children) for Dropdowns
    
    for grid in base_grids:
        mask = (
            (df['Building/Lat'] >= grid.min_lat) & (df['Building/Lat'] < grid.max_lat) &
            (df['Building/Long'] >= grid.min_lon) & (df['Building/Long'] < grid.max_lon)
        )
        subset = df[mask]
        b_count = grid.calculate_stats(subset)
        flat_list.append(grid)
        
        if b_count > threshold:
            children = grid.split()
            processed_children = []
            
            for child in children:
                c_mask = (
                    (subset['Building/Lat'] >= child.min_lat) & (subset['Building/Lat'] < child.max_lat) &
                    (subset['Building/Long'] >= child.min_lon) & (subset['Building/Long'] < child.max_lon)
                )
                c_subset = subset[c_mask]
                c_b_count = child.calculate_stats(c_subset)
                flat_list.append(child)
                
                if c_b_count > threshold:
                    grand_children = child.split()
                    for gc in grand_children:
                        gc_mask = (
                            (c_subset['Building/Lat'] >= gc.min_lat) & (c_subset['Building/Lat'] < gc.max_lat) &
                            (c_subset['Building/Long'] >= gc.min_lon) & (c_subset['Building/Long'] < gc.max_lon)
                        )
                        gc.calculate_stats(c_subset[gc_mask])
                        processed_children.append(gc)
                        flat_list.append(gc)
                else:
                    processed_children.append(child)
            
            final_output.extend(processed_children)
        else:
            final_output.append(grid)
            
    return final_output, flat_list

# --- 5. UI: SIDEBAR ---

st.title("🗺️ Jumbo Homes: Operational Planner")

# A. Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    split_threshold = st.number_input("Split Threshold (Buildings)", 10, 200, 50)
    
    st.divider()
    
    st.subheader("💰 Price Filter (Lakhs)")
    price_options = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500, 1000]
    
    c1, c2 = st.columns(2)
    with c1:
        min_price = st.selectbox("Min Price", price_options, index=0)
    with c2:
        max_price = st.selectbox("Max Price", price_options, index=len(price_options)-3)
        
    if min_price >= max_price:
        st.error("⚠️ Min Price must be less than Max Price!")
        st.stop()
        
    st.info(f"Showing homes between {min_price}L - {max_price}L")

# Apply Filter
filtered_df = df_homes[
    (df_homes['Clean_Price'] >= min_price) & 
    (df_homes['Clean_Price'] <= max_price)
]

# Run Algo
base_grids = generate_7x7_matrix()
ops_grids, all_grids_flat = process_grids(base_grids, filtered_df, split_threshold)

# --- 6. UI: MAIN AREA ---

# A. Search Bar
c_search, c_map = st.columns([1, 3])

with c_search:
    st.subheader("🔍 Find Location")
    search_type = st.radio("Search By:", ["House ID / Project", "Landmark (OpenMap)"])
    query = st.text_input("Enter Query", placeholder="e.g. JB-101 or Indiranagar")
    
    search_loc = None
    
    if query:
        if search_type == "House ID / Project":
            mask = (
                filtered_df['House_ID'].astype(str).str.contains(query, case=False) |
                filtered_df['Building/Name'].astype(str).str.contains(query, case=False)
            )
            res = filtered_df[mask]
            if not res.empty:
                first = res.iloc[0]
                search_loc = [first['Building/Lat'], first['Building/Long']]
                st.success(f"Found: {first['Building/Name']}")
            else:
                st.warning("No match in database.")
                
        else:
            # GEOPY SEARCH
            try:
                geolocator = Nominatim(user_agent="jumbo_ops_app")
                # Append Bangalore for better context
                location = geolocator.geocode(f"{query}, Bangalore, India")
                if location:
                    search_loc = [location.latitude, location.longitude]
                    st.success(f"📍 {location.address}")
                else:
                    st.warning("Location not found on OpenStreetMap.")
            except Exception as e:
                st.error("Search Service Unavailable.")

# B. MAP
with c_map:
    center = search_loc if search_loc else [12.9716, 77.5946]
    zoom = 13 if search_loc else 11
    
    m = folium.Map(location=center, zoom_start=zoom, prefer_canvas=True)
    
    for g in ops_grids:
        # Coloring
        if g.level == 1:
            col, op = "#333", 0.05
        elif g.level == 2:
            col, op = "#ff9800", 0.15
        else:
            col, op = "#d32f2f", 0.25
            
        folium.Rectangle(
            bounds=[[g.min_lat, g.min_lon], [g.max_lat, g.max_lon]],
            color=col, weight=1, fill=True, fill_opacity=op,
            tooltip=f"{g.id} ({len(g.buildings)} Projects)",
            popup=folium.Popup(f"<b>ID: {g.id}</b><br>Supply: {g.total_supply}", max_width=100)
        ).add_to(m)
        
        # Label
        folium.Marker(
            location=[(g.min_lat + g.max_lat)/2, (g.min_lon + g.max_lon)/2],
            icon=folium.DivIcon(html=f'<div style="font-size:8px; color:{col}; font-weight:bold;">{g.id}</div>')
        ).add_to(m)
        
    if search_loc:
        folium.Marker(search_loc, icon=folium.Icon(color="green", icon="star")).add_to(m)
        
    st_folium(m, width="100%", height=600)

# --- 7. DRILL DOWN TABLE ---

st.divider()
st.subheader("📊 Grid Drill-Down")

# Create Hierarchy Dict for Dropdowns
# Structure: { 'JB-A01': ['JB-A01-NW', 'JB-A01-SE'...] }
parent_grids = sorted([g.id for g in all_grids_flat if g.level == 1])
subgrid_map = {g.id: [] for g in all_grids_flat}

# Populate children map (simple string matching)
for g in all_grids_flat:
    if g.level > 1:
        parent_id = g.id.rsplit('-', 1)[0]
        # Handle L3 (Grandchildren) -> Find L1 Parent
        # Actually simplest is just: If I select "JB-A01", show anything starting with "JB-A01"
        pass 

col_dd1, col_dd2, col_dd3 = st.columns([1, 1, 3])

with col_dd1:
    selected_parent = st.selectbox("Select Parent Grid", parent_grids)

with col_dd2:
    # Find all subgrids that start with the parent ID
    possible_subs = [g.id for g in all_grids_flat if g.id.startswith(selected_parent) and g.id != selected_parent]
    
    if possible_subs:
        selected_sub = st.selectbox("Select Subgrid (Optional)", ["All"] + sorted(possible_subs))
    else:
        selected_sub = "All"
        st.caption("No subgrids defined.")

# TABLE LOGIC
target_id = selected_sub if selected_sub != "All" else selected_parent

# Find the object corresponding to target_id
target_obj = next((g for g in all_grids_flat if g.id == target_id), None)

with col_dd3:
    if target_obj and not target_obj.df_subset.empty:
        st.markdown(f"**Inventory Status: {target_id}**")
        
        # Pivot Data
        # Filter again just to be safe, though target_obj has it.
        # However, target_obj.df_subset might include children if it's a parent.
        # If we selected "All", we want the Parent's total data.
        
        data = target_obj.df_subset
        
        # Group by Status and BHK
        pivot = data.groupby(['Clean_Status', 'BHK_Num']).size().unstack(fill_value=0)
        
        # Ensure columns exist (2.0 and 3.0)
        if 2.0 not in pivot.columns: pivot[2.0] = 0
        if 3.0 not in pivot.columns: pivot[3.0] = 0
        
        # Ensure rows exist
        required_rows = ['Live', 'Inspection Pending', 'Catalogue Pending']
        pivot = pivot.reindex(required_rows, fill_value=0)
        
        # Select only 2 and 3 BHK columns
        display_table = pivot[[2.0, 3.0]].astype(int)
        display_table.columns = ['2 BHK Count', '3 BHK Count']
        
        st.table(display_table)
        
    else:
        st.info("No active inventory in this grid.")
