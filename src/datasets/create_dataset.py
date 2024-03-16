import pandas as pd

#Add the path for each data index
Germany_dax = r"C:\Users\Nagham\Investor\Data\indices\dax_d.csv"
USA_spx = r"C:\Users\Nagham\Investor\Data\indices\spx_d.csv"
Hungary_bux = r"C:\Users\Nagham\Investor\Data\indices\^bux.txt"
Czech_Republic_px = r"C:\Users\Nagham\Investor\Data\indices\^px.txt"
Bulgaria_sofix = r"C:\Users\Nagham\Investor\Data\indices\^sofix.txt"
Latvia_omxr = r"C:\Users\Nagham\Investor\Data\indices\^omxr.txt"
Estonia_omxt = r"C:\Users\Nagham\Investor\Data\indices\^omxt.txt"
Lithuania_omxv = r"C:\Users\Nagham\Investor\Data\indices\^omxv.txt"
Poland_wig20 = r"C:\Users\Nagham\Investor\Data\indices\wig20_d.csv"




def read_csv_file(file_path: str, delimiter: str = '\t') -> pd.DataFrame:
    """
    Read a TXT file and convert it to tabular data.

    Parameters:
        file_path (str): The path to the TXT file.
        delimiter (str): The delimiter used in the TXT file. Default is '\t' (tab).

    Returns:
        pandas.DataFrame: The tabular data.
    """
    try:
        # Read the TXT file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None
    
    
dax = read_csv_file(Germany_dax, ",")
spx = read_csv_file(USA_spx, ",")
bux = read_csv_file(Hungary_bux, ",")
px = read_csv_file(Czech_Republic_px, ",")
sofix = read_csv_file(Bulgaria_sofix, ",")
omxr = read_csv_file(Latvia_omxr, ",")
omxt = read_csv_file(Estonia_omxt, ",")
omxv = read_csv_file(Lithuania_omxv, ",")
wig20 = read_csv_file(Poland_wig20, ",")


def import_data_set(country_name: str) -> pd.DataFrame:
    """
     Takes a Country Name and return the raw data of the country index as downloaded from:https://stooq.com/
       In {this specific day} 

    Parameters:
        Country Name (str): The path to the TXT file.
    Returns:
        pandas.DataFrame: Raw data for the country index untid XX date.
    """
    name = country_name.lower()
    if name == "germany":
        return dax
    elif name == "usa":
        return spx
    elif name == "hungary":
        return bux
    elif name == "czech republic":
        return px
    elif name == "latvia":
        return omxr
    elif name == "estonia":
        return omxt
    elif name == "lithuania":
        return omxv
    elif name == "poland":
        return wig20
    else:
        print("The Country Index does not exist in our database")
    #===============================================================#
    
print(spx)
print(dax)