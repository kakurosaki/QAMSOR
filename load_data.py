import pandas as pd

def load_population_data(filename, country):
    """
    Loads the population dataset, cleans it, and extracts data for a specific country.

    Args:
    - filename (str): Path to the CSV file.
    - country (str): Name of the country to filter.

    Returns:
    - DataFrame: Transformed dataset with years as index and population values.
    """
    df = pd.read_csv(filename)

    # Remove unnecessary columns
    df = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code", "Unnamed: 68"], errors="ignore")

    # Select data for the chosen country
    df_country = df[df["Country Name"] == country].set_index("Country Name").T

    # Convert index (years) to **numeric** and sort for correct plotting
    df_country.index = pd.to_numeric(df_country.index, errors='coerce')  # Ensure years are numbers
    df_country.dropna(inplace=True)  # Remove any non-numeric rows

    # Rename column to "Population"
    df_country.columns = ["Population"]

    # Ensure the data is sorted in ascending order (for proper plotting)
    df_country.sort_index(inplace=True)

    return df_country

def get_all_countries(filename):
    """
    Extracts all unique country names from the dataset.

    Args:
    - filename (str): Path to the CSV file.

    Returns:
    - List of country names.
    """
    df = pd.read_csv(filename)
    return df["Country Name"].unique().tolist()
