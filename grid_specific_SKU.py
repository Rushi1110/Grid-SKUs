import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import math

st.set_page_config(layout="wide", page_title="Jumbo Territory Planner")

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
        
        # 2. Status Cleaning (Active Only)
        valid_statuses = ['✅ Live', 'Inspection Pending', 'Catalogue Pending', 'Live', 'Inspection Pending', 'Catalogue Pending']
        # Check if 'Internal/Status' exists, else assume all are active
        if 'Internal/Status' in df.columns:
            df = df[df['Internal/Status'].isin(valid_statuses)]
            
        # 3. Price Cleaning (For Sliders)
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

        # 4. Config Cleaning (For 2BHK/3BHK counts)
        df['BHK_Num'] = df['Home/Configuration'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ 'Homes.csv' not found. Please place it in the application folder.")
        return None

df_homes = load_data()

if df_homes is None:
    st.stop()

# --- 3. GRID CLASS (RECURSIVE) ---

class OpsGrid:
    def __init__(self, grid_id, min_lat, min_lon, max_lat, max_lon, level=1):
        self.id = grid_id
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.level = level
        
        # Stats
        self.buildings = set() # Unique Projects
        self.total_supply = 0
        self.bhk2_count = 0
        self.bhk3_count = 0
        self.price_stats = {"min": 0, "max": 0, "avg": 0}
        self.children = []

    def calculate_stats(self, df_subset):
        if df_subset.empty:
            return 0
            
        # Unique Buildings
        if 'Building/Name' in df_subset.columns:
            self.buildings = set(df_subset['Building/Name'].unique())
        
        # Counts
        self.total_supply = len(df_subset)
        self.bhk2_count = len(df_subset[df_subset['BHK_Num'] == 2])
        self.bhk3_count = len(df_subset[df_subset['BHK_Num'] == 3])
        
        # Price Stats
        prices = df_subset['Clean_Price']
        if not prices.empty:
            self.price_stats = {
                "min": prices.min(),
                "max": prices.max(),
                "avg": prices.mean()
            }
            
        return len(self.buildings)

    def split(self):
        mid_lat = (self.min_lat + self.max_lat) / 2
        mid_lon = (self.min_lon + self.max_lon) / 2
        
        # Directions: NW, NE, SW, SE
        # IDs append direction: e.g., JB-A01-NW
        
        # 1. North West (Top Left)
        nw = OpsGrid(f"{self.id}-NW", mid_lat, self.min_lon, self.max_lat, mid_lon, self.level + 1)
        # 2. North East (Top Right)
        ne = OpsGrid(f"{self.id}-NE", mid_lat, mid_lon, self.max_lat, self.max_lon, self.level + 1)
        # 3. South West (Bottom Left)
        sw = OpsGrid(f"{self.id}-SW", self.min_lat, self.min_lon, mid_lat, mid_lon, self.level + 1)
        # 4. South East (Bottom Right)
        se = OpsGrid(f"{self.id}-SE", self.min_lat, mid_lon, mid_lat, self.max_lon, self.level + 1)
        
        self.children = [nw, ne, sw, se]
        return self.children

# --- 4. ALGORITHM ---

def generate_7x7_matrix():
    grids = []
    
    lat_step = (BOUNDS["TOP_LAT"] - BOUNDS["BOTTOM_LAT"]) / GRID_ROWS
    lon_step = (BOUNDS["RIGHT_LON"] - BOUNDS["LEFT_LON"]) / GRID_COLS
    
    # X Axis: A to G
    x_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    for r in range(GRID_ROWS): # 0 to 6
        for c in range(GRID_COLS): # 0 to 6
            # Calculate Bounds
            # Lat: Starts Top, goes down. Row 0 is Top (13.35)
            g_max_lat = BOUNDS["TOP_LAT"] - (r * lat_step)
            g_min_lat = g_max_lat - lat_step
            
            # Lon: Starts Left, goes right. Col 0 is Left (77.25)
            g_min_lon = BOUNDS["LEFT_LON"] + (c * lon_step)
            g_max_lon = g_min_lon + lon_step
            
            # ID: Row 0 -> '01', Row 6 -> '07'
            # ID: Col 0 -> 'A', Col 6 -> 'G'
            grid_id = f"JB-{x_labels[c]}{str(r+1).zfill(2)}"
            
            grids.append(OpsGrid(grid_id, g_min_lat, g_min_lon, g_max_lat, g_max_lon, level=1))
            
    return grids

def process_grids(base_grids, df, threshold):
    final_output = []
    
    for grid in base_grids:
        # 1. Filter Data for this Grid
        mask = (
            (df['Building/Lat'] >= grid.min_lat) & (df['Building/Lat'] < grid.max_lat) &
            (df['Building/Long'] >= grid.min_lon) & (df['Building/Long'] < grid.max_lon)
        )
        subset = df[mask]
        
        # 2. Calculate Stats
        b_count = grid.calculate_stats(subset)
        
        # 3. Split Check
        if b_count > threshold:
            children = grid.split()
            for child in children:
                # Recursive Step (Level 2)
                # Filter again for child
                c_mask = (
                    (subset['Building/Lat'] >= child.min_lat) & (subset['Building/Lat'] < child.max_lat) &
                    (subset['Building/Long'] >= child.min_lon) & (subset['Building/Long'] < child.max_lon)
                )
                c_subset = subset[c_mask]
                c_b_count = child.calculate_stats(c_subset)
                
                # Check Level 3 Split
                if c_b_count > threshold:
                    grand_children = child.split()
                    for gc in grand_children:
                        gc_mask = (
                            (c_subset['Building/Lat'] >= gc.min_lat) & (c_subset['Building/Lat'] < gc.max_lat) &
                            (c_subset['Building/Long'] >= gc.min_lon) & (c_subset['Building/Long'] < gc.max_lon)
                        )
                        gc.calculate_stats(c_subset[gc_mask])
                        final_output.append(gc) # Add L3
                else:
                    final_output.append(child) # Add L2
        else:
            final_output.append(grid) # Add L1
            
    return final_output

# --- 5. UI LAYOUT ---

st.title("🗺️ Jumbo Homes: Operational Grid Planner")

# Sidebar / Top Bar
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.caption("📍 Grid Boundaries (Fixed)")
    st.text(f"NW: {BOUNDS['TOP_LAT']}, {BOUNDS['LEFT_LON']}")
    st.text(f"SE: {BOUNDS['BOTTOM_LAT']}, {BOUNDS['RIGHT_LON']}")

with c2:
    split_threshold = st.number_input("✂️ Split Threshold (Buildings)", min_value=10, max_value=200, value=50)

with c3:
    # Price Filter Slider
    min_p, max_p = int(df_homes['Clean_Price'].min()), int(df_homes['Clean_Price'].max())
    price_range = st.slider("💰 Price Filter (Lakhs)", min_p, max_p, (min_p, max_p))

# Apply Filter BEFORE processing grids
filtered_df = df_homes[
    (df_homes['Clean_Price'] >= price_range[0]) & 
    (df_homes['Clean_Price'] <= price_range[1])
]

# Run Logic
base_grids = generate_7x7_matrix()
ops_grids = process_grids(base_grids, filtered_df, split_threshold)

# --- 6. SEARCH & MAP ---

st.divider()
search_col, map_col = st.columns([1, 3])

with search_col:
    st.subheader("🔍 Search")
    query = st.text_input("Find House ID, Project, or Locality", placeholder="e.g. JB-102 or Indiranagar")
    
    search_result_loc = None
    
    if query:
        # Search Logic
        mask = (
            df_homes['House_ID'].astype(str).str.contains(query, case=False) |
            df_homes['Building/Name'].astype(str).str.contains(query, case=False) |
            df_homes['Building/Locality'].astype(str).str.contains(query, case=False)
        )
        results = df_homes[mask]
        
        if not results.empty:
            st.success(f"Found {len(results)} matches")
            # Pick first match to center map
            first = results.iloc[0]
            search_result_loc = [first['Building/Lat'], first['Building/Long']]
            st.info(f"📍 {first['Building/Name']} ({first['House_ID']})")
        else:
            st.error("No matches found.")

    st.markdown("### 📊 Territory Stats")
    st.write(f"**Total Grids:** {len(ops_grids)}")
    st.write(f"**Total Active Supply:** {len(filtered_df)}")
    
    # Legend
    st.markdown("""
    <div style='background-color:#eee; padding:10px; border-radius:5px;'>
    <b>Legend:</b><br>
    ⬛ <b>Black:</b> L1 Parent (Sparse)<br>
    🟧 <b>Orange:</b> L2 Zone (Active)<br>
    🟥 <b>Red:</b> L3 Beat (Dense)
    </div>
    """, unsafe_allow_html=True)

with map_col:
    # Center map
    start_loc = search_result_loc if search_result_loc else [12.9716, 77.5946]
    zoom = 13 if search_result_loc else 11
    
    m = folium.Map(location=start_loc, zoom_start=zoom, prefer_canvas=True)
    
    # Draw Bounds Box
    folium.Rectangle(
        bounds=[[BOUNDS["TOP_LAT"], BOUNDS["LEFT_LON"]], [BOUNDS["BOTTOM_LAT"], BOUNDS["RIGHT_LON"]]],
        color="blue", fill=False, weight=1, dash_array="5, 5"
    ).add_to(m)
    
    for g in ops_grids:
        # Styling
        if g.level == 1:
            col, w, op = "#333", 1, 0.05
        elif g.level == 2:
            col, w, op = "#ff9800", 2, 0.15
        else:
            col, w, op = "#d32f2f", 2, 0.25
            
        # Tooltip Content
        tooltip_html = f"""
        <div style='font-family:sans-serif; width:180px;'>
            <b>ID:</b> {g.id}<br>
            <b>Level:</b> {g.level}<br>
            <hr style='margin:5px 0;'>
            <b>🏢 Buildings:</b> {len(g.buildings)}<br>
            <b>🏠 Total Active:</b> {g.total_supply}<br>
            • 2 BHK: {g.bhk2_count}<br>
            • 3 BHK: {g.bhk3_count}<br>
            <hr style='margin:5px 0;'>
            <b>Price Range:</b><br>
            {int(g.price_stats['min'])}L - {int(g.price_stats['max'])}L
        </div>
        """
        
        # Draw Grid
        folium.Rectangle(
            bounds=[[g.min_lat, g.min_lon], [g.max_lat, g.max_lon]],
            color=col,
            weight=w,
            fill=True,
            fill_opacity=op,
            popup=folium.Popup(tooltip_html, max_width=250),
            tooltip=f"{g.id} ({len(g.buildings)} Projects)"
        ).add_to(m)
        
        # Center Label
        folium.Marker(
            location=[(g.min_lat + g.max_lat)/2, (g.min_lon + g.max_lon)/2],
            icon=folium.DivIcon(html=f'<div style="font-size:8px; font-weight:bold; color:{col};">{g.id}</div>')
        ).add_to(m)

    # If search result exists, add a marker
    if search_result_loc:
        folium.Marker(
            location=search_result_loc,
            icon=folium.Icon(color="green", icon="star"),
            tooltip="Search Result"
        ).add_to(m)

    st_folium(m, width="100%", height=700)