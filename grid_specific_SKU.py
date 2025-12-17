import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, box

st.set_page_config(layout="wide", page_title="Jumbo Tour Planner v7")

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
        
        # 2. Status Cleaning
        status_map = {
            '✅ Live': 'Live',
            'Live': 'Live',
            'Inspection Pending': 'Inspection Pending',
            'Catalogue Pending': 'Catalogue Pending'
        }
        if 'Internal/Status' in df.columns:
            df['Clean_Status'] = df['Internal/Status'].map(status_map).fillna('Other')
            df = df[df['Clean_Status'].isin(['Live', 'Inspection Pending', 'Catalogue Pending'])]
        else:
            df['Clean_Status'] = 'Live' 
            
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

        # 4. Config Cleaning
        df['BHK_Num'] = df['Home/Configuration'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0)
        
        return df
        
    except FileNotFoundError:
        st.error("❌ 'Homes.csv' not found.")
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
        self.df_subset = pd.DataFrame() 

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
    flat_list = [] 
    
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
st.title("🗺️ Jumbo Homes: Tour Planner")

with st.sidebar:
    st.header("⚙️ Configuration")
    split_threshold = st.number_input("Split Threshold (Buildings)", 10, 200, 50)
    st.divider()
    st.subheader("💰 Price Filter (Lakhs)")
    price_options = [0, 20, 40, 60, 80, 100, 120, 150, 200, 300, 500, 1000]
    c1, c2 = st.columns(2)
    with c1: min_price = st.selectbox("Min Price", price_options, index=0)
    with c2: max_price = st.selectbox("Max Price", price_options, index=len(price_options)-3)
        
    if min_price >= max_price:
        st.error("⚠️ Min > Max")
        st.stop()
        
    st.info(f"Showing homes between {min_price}L - {max_price}L")

filtered_df = df_homes[
    (df_homes['Clean_Price'] >= min_price) & 
    (df_homes['Clean_Price'] <= max_price)
]

base_grids = generate_7x7_matrix()
ops_grids, all_grids_flat = process_grids(base_grids, filtered_df, split_threshold)

# --- 6. UI: MAIN CONTROL ---
st.divider()

# MODE TOGGLE
c_mode, c_search = st.columns([1, 2])
with c_mode:
    is_draw_mode = st.toggle("✨ Dynamic Search (Draw Mode)", value=False)

with c_search:
    st.caption("Locate Landmark")
    c_s1, c_s2 = st.columns([1, 2])
    with c_s1: search_type = st.radio("Type", ["House ID", "Landmark"], label_visibility="collapsed")
    with c_s2: query = st.text_input("Search", placeholder="e.g. JB-101 or Indiranagar", label_visibility="collapsed")
    
    search_loc = None
    if query:
        if search_type == "House ID":
            mask = (
                filtered_df['House_ID'].astype(str).str.contains(query, case=False) |
                filtered_df['Building/Name'].astype(str).str.contains(query, case=False)
            )
            res = filtered_df[mask]
            if not res.empty:
                first = res.iloc[0]
                search_loc = [first['Building/Lat'], first['Building/Long']]
                st.toast(f"Found: {first['Building/Name']}")
            else:
                st.toast("No match found.")
        else:
            try:
                geolocator = Nominatim(user_agent="jumbo_ops_app")
                location = geolocator.geocode(f"{query}, Bangalore, India")
                if location:
                    search_loc = [location.latitude, location.longitude]
                    st.toast(f"📍 {location.address}")
            except:
                pass

# --- 7. GRID SELECTOR (ONLY IF DRAW MODE OFF) ---
target_id = None
target_obj = None

if not is_draw_mode:
    c_sel1, c_sel2 = st.columns(2)
    with c_sel1:
        parent_grids = sorted([g.id for g in all_grids_flat if g.level == 1])
        selected_parent = st.selectbox("Select Parent Grid", parent_grids)
    with c_sel2:
        possible_subs = [g.id for g in all_grids_flat if g.id.startswith(selected_parent) and g.id != selected_parent]
        selected_sub = st.selectbox("Select Subgrid", ["All"] + sorted(possible_subs)) if possible_subs else "All"
    
    target_id = selected_sub if selected_sub != "All" else selected_parent
    target_obj = next((g for g in all_grids_flat if g.id == target_id), None)
else:
    st.info("🎨 Drawing Mode Active: The map is clear. Use the toolbar on the left of the map to draw a Circle or Rectangle.")

# --- 8. MAP SECTION ---
c_map, c_data = st.columns([3, 2])

# Logic for Custom Search Zone
active_drawing = None

with c_map:
    # 1. Base Map Center
    start_loc = [12.9716, 77.5946]
    zoom = 11

    if search_loc:
        start_loc = search_loc
        zoom = 13
    elif target_obj and not is_draw_mode:
        start_loc = [(target_obj.min_lat + target_obj.max_lat)/2, (target_obj.min_lon + target_obj.max_lon)/2]
        zoom = 13 if target_obj.level > 1 else 12

    m = folium.Map(location=start_loc, zoom_start=zoom, prefer_canvas=True)
    
    # 2. Add Draw Control ONLY if Toggle is ON
    if is_draw_mode:
        draw = Draw(
            draw_options={
                'polyline': False,
                'polygon': False, 
                'rectangle': True,
                'circle': True,
                'marker': False,
                'circlemarker': False
            },
            edit_options={'edit': False}
        )
        draw.add_to(m)

    # 3. Draw Grids (Background Layer) - ONLY IF DRAW MODE IS OFF
    if not is_draw_mode:
        for g in ops_grids:
            # Highlight Logic
            is_selected = (g.id == target_id)
            
            if g.level == 1: col, op = "#333", 0.05
            elif g.level == 2: col, op = "#ff9800", 0.15
            else: col, op = "#d32f2f", 0.25
            
            if is_selected:
                col = "blue"
                op = 0.05
                weight = 3
            else:
                weight = 1

            folium.Rectangle(
                bounds=[[g.min_lat, g.min_lon], [g.max_lat, g.max_lon]],
                color=col, weight=weight, fill=True, fill_opacity=op,
                tooltip=f"{g.id}",
                popup=folium.Popup(f"<b>{g.id}</b><br>Count: {g.total_supply}", max_width=100)
            ).add_to(m)
            
            folium.Marker(
                location=[(g.min_lat + g.max_lat)/2, (g.min_lon + g.max_lon)/2],
                icon=folium.DivIcon(html=f'<div style="font-size:8px; color:{col}; font-weight:bold;">{g.id}</div>')
            ).add_to(m)
    
    if search_loc:
        folium.Marker(search_loc, icon=folium.Icon(color="green", icon="star")).add_to(m)

    # 4. Render Map & Capture Drawing
    output = st_folium(m, width="100%", height=600)

# --- 9. DATA LOGIC ---

final_data = pd.DataFrame()
title_text = "Select a Grid"

# A. DRAW MODE LOGIC
if is_draw_mode and output and output['last_active_drawing']:
    drawing = output['last_active_drawing']
    geom_type = drawing['geometry']['type']
    
    if geom_type == 'Point': 
        center = drawing['geometry']['coordinates'] # [lon, lat]
        radius_m = drawing['properties']['radius']
        
        def is_in_circle(row):
            dist = geodesic((row['Building/Lat'], row['Building/Long']), (center[1], center[0])).meters
            return dist <= radius_m
            
        final_data = filtered_df[filtered_df.apply(is_in_circle, axis=1)]
        title_text = "Custom Circle Zone"
        
    elif geom_type == 'Polygon': 
        coords = drawing['geometry']['coordinates'][0]
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        mask = (
            (filtered_df['Building/Lat'] >= min(lats)) & (filtered_df['Building/Lat'] <= max(lats)) &
            (filtered_df['Building/Long'] >= min(lons)) & (filtered_df['Building/Long'] <= max(lons))
        )
        final_data = filtered_df[mask]
        title_text = "Custom Box Zone"

# B. GRID MODE LOGIC
elif not is_draw_mode and target_obj:
    final_data = target_obj.df_subset.copy()
    title_text = f"Grid: {target_id}"

# --- 10. DISPLAY TABLE ---
with c_data:
    st.markdown(f"### 📊 Inventory: {title_text}")
    
    if not final_data.empty:
        # 1. SUMMARY MATRIX
        pivot = final_data.groupby(['BHK_Num', 'Clean_Status']).size().unstack(fill_value=0)
        required_cols = ['Live', 'Inspection Pending', 'Catalogue Pending']
        for c in required_cols:
            if c not in pivot.columns: pivot[c] = 0
        
        pivot = pivot[required_cols]
        pivot.index = pivot.index.map(lambda x: f"{int(x)} BHK")
        
        st.table(pivot)

        # 2. TOUR LIST
        st.markdown("#### 📋 Tour List")
        
        tour_df = final_data[['House_ID', 'Building/Name', 'Clean_Status', 'BHK_Num', 'Clean_Price']].copy()
        tour_df['BHK_Num'] = tour_df['BHK_Num'].astype(int).astype(str) + " BHK"
        tour_df = tour_df.sort_values(by=['Clean_Status', 'Clean_Price'])
        tour_df.columns = ['ID', 'Project', 'Status', 'Config', 'Price (L)']
        
        st.dataframe(
            tour_df.style.format({"Price (L)": "{:.0f}"}), 
            use_container_width=True,
            height=400
        )
    else:
        if is_draw_mode:
            st.info("Draw a shape on the map to see homes.")
        else:
            st.info("No active inventory here.")
