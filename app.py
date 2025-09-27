import requests
import json
import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from bs4 import BeautifulSoup
import time
from typing import Dict, Any, List

# --- Configuration ---
TICKER = "INFY" # Change this to the desired NSE code (e.g., "RELIANCE")
ISIN = "INE009A01021" # Change this to the corresponding ISIN
OUTPUT_FILENAME = f"{TICKER}_Stock_Report.docx"

# NSE API Endpoints (These are the underlying APIs used by the NSE website)
NSE_BASE_URL = "https://www.nseindia.com/"
NSE_QUOTE_API = f"https://www.nseindia.com/api/quote-equity?symbol={TICKER}"
NSE_FINANCIAL_API = f"https://www.nseindia.com/api/corporates-financial-results?symbol={TICKER}&index=equities&period=Quarterly"
NSE_SHAREHOLDING_API = f"https://www.nseindia.com/api/corporates-shareholding?symbol={TICKER}"

# Cogencis URL for Ownership Data (Scraping required)
COGENCIS_OWNERSHIP_URL = f"https://iinvest.cogencis.com/{ISIN}/symbol/ns/{TICKER}/Infosys%20Limited?tab=ownership-data&type=capital-history"

# Headers to mimic a browser, essential for accessing NSE APIs
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/555.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/555.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}

def create_nse_session() -> requests.Session:
    """Creates a session and fetches initial cookies needed for NSE API access."""
    session = requests.Session()
    try:
        # A preliminary call to the base URL is often required to get necessary cookies
        session.get(NSE_BASE_URL, headers=HEADERS, timeout=10)
        return session
    except requests.RequestException as e:
        print(f"Error establishing session with NSE: {e}")
        return None

def fetch_nse_quote_data(session: requests.Session) -> Dict[str, Any]:
    """
    Fetches the main stock quote data from the NSE API, including Volume, Value, and P/E.
    """
    data = {}
    print(f"Fetching live quote data for {TICKER}...")
    try:
        response = session.get(NSE_QUOTE_API, headers=HEADERS, timeout=10)
        response.raise_for_status()
        quote_data = response.json()
        
        # Extract required metrics
        total_traded_value = quote_data.get('totalTradedValue', 0)
        
        data['Latest Price Data'] = {
            'Company Name': quote_data.get('info', {}).get('companyName', 'N/A'),
            'Latest Trade Date': quote_data.get('metadata', {}).get('lastUpdateTime', 'N/A'),
            # Adjusted P/E is usually found in the preOpenMarket or marketStatus sections
            'Adjusted P/E (TTM)': quote_data.get('preOpenMarket', {}).get('finalPrice', {}).get('pE', 'N/A'),
            'Total Traded Volume (Shares)': f"{quote_data.get('totalTradedVolume', 'N/A'):,}",
            # Convert value to Crores (1 Crore = 10 million, or 100 lakh)
            'Total Traded Value (₹ Crores)': f"{round(total_traded_value / 10000000, 2):,}" 
        }
    except Exception as e:
        print(f"Failed to fetch NSE quote data: {e}")
    return data

def fetch_nse_financial_data(session: requests.Session) -> pd.DataFrame:
    """
    Fetches quarterly financial results from NSE API, including Income, Expenses, and Tax.
    All values converted to Crores.
    """
    df = pd.DataFrame()
    print(f"Fetching quarterly financial data for {TICKER}...")
    time.sleep(1) # Small delay to respect API limits
    
    try:
        response = session.get(NSE_FINANCIAL_API, headers=HEADERS, timeout=10)
        response.raise_for_status()
        financial_data = response.json()
        
        if 'data' in financial_data:
            results = []
            for item in financial_data['data']:
                income = item.get('totalIncome', 0)
                expenses = item.get('totalExpenses', 0)
                tax = item.get('taxExpense', 0)
                
                # Assuming data is in Lakhs, converting to Crores
                results.append({
                    'Quarter Ended': item.get('period'),
                    'Total Income (Cr)': round(income / 100, 2), 
                    'Total Expenses (Cr)': round(expenses / 100, 2), 
                    'Total Tax Expense (Cr)': round(tax / 100, 2)
                })
            
            # Use the latest 4 quarters for a concise view and transpose for better display
            df = pd.DataFrame(results).head(4)
            if not df.empty:
                df = df.set_index('Quarter Ended').T
            
    except Exception as e:
        print(f"Failed to fetch NSE financial data: {e}")
    return df

def fetch_nse_shareholding_data(session: requests.Session) -> pd.DataFrame:
    """
    Fetches the latest quarterly FII/DII shareholding pattern.
    """
    df = pd.DataFrame()
    print(f"Fetching quarterly shareholding data for {TICKER}...")
    time.sleep(1)
    
    try:
        response = session.get(NSE_SHAREHOLDING_API, headers=HEADERS, timeout=10)
        response.raise_for_status()
        shareholding_data = response.json()
        
        if 'data' in shareholding_data:
            results = []
            for item in shareholding_data['data']:
                category = item.get('category', 'N/A')
                percent = item.get('value', '0.0')
                
                # Filter for key institutional categories
                if 'FII' in category.upper() or 'DII' in category.upper() or 'MUTUAL FUND' in category.upper():
                    results.append({'Category': category, 'Percentage (%)': float(percent)})
            
            # Get the latest filing date
            latest_date = shareholding_data.get('latest_date', 'N/A')
            
            if results:
                df = pd.DataFrame(results)
                # Aggregate and format
                df_agg = df.groupby('Category')['Percentage (%)'].sum().reset_index()
                df_agg['Percentage (%)'] = df_agg['Percentage (%)'].apply(lambda x: f"{x:.2f}%")
                df_agg.columns = ['Category', f'Latest Percentage ({latest_date})']
                df = df_agg.set_index('Category').T
        
    except Exception as e:
        print(f"Failed to fetch NSE shareholding data: {e}")
    return df

def fetch_cogencis_ownership_data() -> Dict[str, pd.DataFrame]:
    """
    Scrapes all tables from the Cogencis ownership data page using BeautifulSoup.
    """
    scraped_data = {}
    print(f"Fetching ownership data from Cogencis...")
    time.sleep(2)
    
    try:
        # Note: Cogencis is not an NSE API, so standard requests.get is used.
        response = requests.get(COGENCIS_OWNERSHIP_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for all tables that likely contain the data
        tables = soup.find_all('table')
        
        if not tables:
            scraped_data['Error'] = "No tables found on the Cogencis ownership page."
            return scraped_data
            
        for i, table in enumerate(tables):
            try:
                # Use Pandas to read the table HTML into a DataFrame
                df_list = pd.read_html(str(table))
                if df_list:
                    df = df_list[0]
                    # Attempt to find a suitable header/title preceding the table
                    title_tag = table.find_previous(['h3', 'h4', 'h5', 'p'], text=True)
                    table_title = title_tag.text.strip() if title_tag and len(title_tag.text.strip()) > 5 else f"Ownership/Capital Table {i+1}"
                    
                    if not df.empty:
                        # Clean up multi-index headers
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' '.join(col).strip() for col in df.columns.values]
                        scraped_data[table_title] = df.fillna('') # Replace NaN with empty string
            except Exception:
                continue # Skip tables that fail to parse

    except Exception as e:
        scraped_data['Error'] = f"Failed to fetch Cogencis data: {e}"
        
    return scraped_data

def add_dataframe_to_word(document, df: pd.DataFrame, table_style: str = 'Table Grid'):
    """Helper function to convert a Pandas DataFrame to a Word table."""
    # Add an empty paragraph before the table for spacing
    document.add_paragraph()
    
    # Check if the DataFrame has a column index (for transposed tables)
    has_index = df.index.name is not None or not pd.RangeIndex(start=0, stop=len(df.index)).equals(df.index)

    # Initialize table with dimensions
    rows, cols = df.shape
    num_cols = cols + (1 if has_index else 0)
    table = document.add_table(rows + 1, num_cols)
    table.style = table_style
    
    # Set up header row
    hdr_cells = table.rows[0].cells
    
    current_col = 0
    if has_index:
        hdr_cells[0].text = str(df.index.name or 'Index')
        current_col = 1
        
    for j, col_name in enumerate(df.columns):
        hdr_cells[current_col + j].text = str(col_name)
        hdr_cells[current_col + j].paragraphs[0].runs[0].font.bold = True
        hdr_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Populate data rows
    for i, (index_name, row) in enumerate(df.iterrows()):
        row_cells = table.rows[i + 1].cells
        current_col = 0
        
        if has_index:
            row_cells[0].text = str(index_name)
            current_col = 1
            
        for j, value in enumerate(row):
            # Format numbers consistently (especially for crores/percentages)
            text_value = str(value)
            try:
                # Basic attempt to format large numbers
                if text_value.replace(',', '').replace('.', '', 1).isdigit() and len(text_value.replace('.', '')) > 4:
                    text_value = f"{float(text_value.replace(',', '')):,}"
            except:
                pass # Keep as is if formatting fails

            row_cells[current_col + j].text = text_value
            # Align numbers to the right
            if any(char.isdigit() for char in text_value) and not any(char.isalpha() for char in text_value):
                 row_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                 row_cells[current_col + j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT


def generate_word_document(
    quote_data: Dict[str, Any], 
    financial_data: pd.DataFrame, 
    shareholding_data: pd.DataFrame,
    ownership_data: Dict[str, pd.DataFrame], 
    filename: str
):
    """Generates a professional Word document with all collected data."""
    document = Document()
    
    # Set document style
    style = document.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # Main Title
    document.add_heading(f"Comprehensive Stock Analysis Report: {TICKER}", 0)
    document.add_paragraph(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    document.add_paragraph("---")

    # --- Section 1: NSE Live Quote and Valuation Data ---
    document.add_heading("1. Live Market and Valuation Metrics (NSE)", level=1)
    if 'Latest Price Data' in quote_data:
        data = quote_data['Latest Price Data']
        
        document.add_paragraph(f"Company: {data['Company Name']}")
        document.add_paragraph(f"Last Updated: {data['Latest Trade Date']}")
        document.add_paragraph(f"Adjusted TTM P/E: {data['Adjusted P/E (TTM)']}")
        document.add_paragraph(f"Total Traded Volume: {data['Total Traded Volume']} Shares")
        document.add_paragraph(f"Total Traded Value: ₹{data['Total Traded Value (₹ Crores)']} Crores")
    else:
        document.add_paragraph("Live market data could not be retrieved from NSE.")
    
    document.add_paragraph("---")

    # --- Section 2: Quarterly Financial Results Comparison (NSE) ---
    document.add_heading("2. Quarterly Financial Results Comparison (₹ Crores)", level=1)
    if not financial_data.empty:
        add_dataframe_to_word(document, financial_data, table_style='Light Grid Accent 2')
    else:
        document.add_paragraph("Quarterly financial comparison data could not be retrieved from NSE.")

    document.add_paragraph("---")

    # --- Section 3: FII/DII Shareholding Pattern (NSE) ---
    document.add_heading("3. Institutional Shareholding Pattern (NSE)", level=1)
    if not shareholding_data.empty:
        add_dataframe_to_word(document, shareholding_data, table_style='Grid Table 4 Accent 1')
        document.add_paragraph("Data represents the latest quarterly percentage breakdown reported by the company.")
    else:
        document.add_paragraph("Quarterly shareholding data (FII/DII breakdown) could not be retrieved from NSE.")
    
    document.add_paragraph("---")

    # --- Section 4: Cogencis Ownership Data ---
    document.add_heading(f"4. Ownership and Capital History (Source: Cogencis)", level=1)

    if 'Error' in ownership_data:
        document.add_paragraph(f"Error scraping Cogencis data: {ownership_data['Error']}")
    elif ownership_data:
        for title, df in ownership_data.items():
            document.add_heading(f"4.{list(ownership_data.keys()).index(title) + 1}: {title}", level=2)
            add_dataframe_to_word(document, df, table_style='List Table 4 Accent 3')
    else:
        document.add_paragraph("No ownership or capital history tables were successfully scraped from Cogencis.")
        
    # Save the document
    document.save(filename)
    print(f"\n--- SUCCESS ---")
    print(f"Report generated successfully: {filename}")


# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Initialize session
    nse_session = create_nse_session()
    
    if nse_session:
        # 2. Fetch data
        quote_data_result = fetch_nse_quote_data(nse_session)
        financial_df = fetch_nse_financial_data(nse_session)
        shareholding_df = fetch_nse_shareholding_data(nse_session)
        ownership_data_result = fetch_cogencis_ownership_data()

        # 3. Generate the Word Document
        generate_word_document(
            quote_data_result,
            financial_df,
            shareholding_df,
            ownership_data_result,
            OUTPUT_FILENAME
        )
    else:
        print("Could not initialize the required session to fetch data. Please check your network connection.")
