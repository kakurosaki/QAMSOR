import pandas as pd

def load_population_data(data_file, country):
    """Load and transform population data from wide to long format"""
    try:
        df = pd.read_csv(data_file)
        country_data = df[df["Country Name"] == country]
        
        if country_data.empty:
            print(f"No data found for {country}")
            return pd.DataFrame()
        
        # Get all year columns (1960-2023)
        year_cols = [str(y) for y in range(1960, 2024)]
        
        # Reshape from wide to long format
        melted = country_data.melt(
            id_vars=["Country Name", "Country Code"],
            value_vars=year_cols,
            var_name="Year",
            value_name="Population"
        )
        
        # Convert types and set index
        melted["Year"] = melted["Year"].astype(int)
        result = melted.set_index("Year")[["Population"]].sort_index()
        
        return result.dropna()
        
    except Exception as e:
        print(f"Error loading data for {country}: {str(e)}")
        return pd.DataFrame()

def get_all_countries(data_file):
    """Get list of unique countries in dataset"""
    try:
        df = pd.read_csv(data_file)
        return df["Country Name"].unique().tolist()
    except Exception as e:
        print(f"Error reading countries: {str(e)}")
        return []