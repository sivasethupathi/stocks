import requests
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from bs4 import BeautifulSoup
import re
import time
from typing import Dict, Any, List

# --- Configuration ---
TICKER = "INFY"
ISIN = "INE009A01021"
OUTPUT_FILENAME = f"{TICKER}_Stock_Report.docx"
NSE_BASE_URL = "https://www.nseindia.com/"
# NSE API endpoint for current equity data
NSE_QUOTE_API = f"https://www.nseindia.com/api/quote-equity?symbol={TICKER}"
# NSE API endpoint for quarterly financial results
NSE_FINANCIAL_API = f"https://www.nseindia.com/api/corporates-financial-results?symbol={TICKER}&index=equities&period=Quarterly"
# Cogencis URL (The second URL provided)
COGENCIS_OWNERSHIP_URL = f"https://iinvest.cogencis.com/{ISIN}/symbol/ns/{TICKER}/Infosys%20Limited?tab=ownership-data&type=capital-history"

# Headers to mimic a browser, essential for accessing NSE APIs
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
}

def create_nse_session() -> requests.Session:
    """Creates a session and fetches initial cookies needed for NSE API access."""
    session = requests.Session()
    try:
        session.get(NSE_BASE_URL, headers=HEADERS, timeout=10)
        return session
    except requests.RequestException as e:
        print(f"Error establishing session with NSE: {e}")
        return None

def fetch_nse_quote_data(session: requests.Session) -> Dict[str, Any]:
    """
    Fetches the main stock quote data from the NSE API.
    Required: Total Traded Volume, Total Traded Value, Adjusted P/E.
    """
    data = {}
    print(f"Fetching live quote data for {TICKER}...")
    try:
        response = session.get(NSE_QUOTE_API, headers=HEADERS, timeout=10)
        response.raise_for_status()
        quote_data = response.json()
        
        # Extract required metrics from the complex JSON structure
        data['Latest Price Data'] = {
            'Company Name': quote_data.get('info', {}).get('companyName', 'N/A'),
            'Latest Trade Date': quote_data.get('metadata', {}).get('lastUpdateTime', 'N/A'),
            'Adjusted P/E (TTM)': quote_data.get('preOpenMarket', {}).get('finalPrice', {}).get('pE', 'N/A'),
            'Total Traded Volume': quote_data.get('totalTradedVolume', 'N/A'),
            'Total Traded Value (Cr)': round(quote_data.get('totalTradedValue', 0) / 10000000, 2)
        }
    except requests.RequestException as e:
        print(f"Failed to fetch NSE quote data: {e}")
    except json.JSONDecodeError:
        print("Failed to decode NSE quote JSON. Request may have been blocked.")
    
    return data

def fetch_nse_financial_data(session: requests.Session) -> pd.DataFrame:
    """
    Fetches quarterly financial results from NSE API.
    Required: Total Income, Total Expenses, Total Tax Expenses.
    """
    df = pd.DataFrame()
    print(f"Fetching quarterly financial data for {TICKER}...")
    
    # Introduce small delay
    time.sleep(1)
    
    try:
        response = session.get(NSE_FINANCIAL_API, headers=HEADERS, timeout=10)
        response.raise_for_status()
        financial_data = response.json()
        
        if 'data' in financial_data:
            # Create a list of dictionaries for the DataFrame
            results = []
            
            for item in financial_data['data']:
                # The data structure may be complex, we look for key items like Income, Expense, Tax
                # The values are often in Lakhs or Crores; we assume they are large numbers
                income = item.get('totalIncome', 0)
                expenses = item.get('totalExpenses', 0)
                tax = item.get('taxExpense', 0)
                
                # Check for zero values, they might indicate the data is not in the expected field
                if income > 0 and expenses > 0:
                    results.append({
                        'Quarter Ended': item.get('period'),
                        'Total Income (Cr)': round(income / 100, 2), # Assuming data is in Lakhs (1 Cr = 100 Lakhs)
                        'Total Expenses (Cr)': round(expenses / 100, 2), 
                        'Total Tax Expense (Cr)': round(tax / 100, 2)
                    })
            
            # Use the latest 4 quarters for a concise view
            df = pd.DataFrame(results).head(4)
            # Reorder columns and sort by date
            if not df.empty:
                df = df.set_index('Quarter Ended').T
            
    except requests.RequestException as e:
        print(f"Failed to fetch NSE financial data: {e}")
    except json.JSONDecodeError:
        print("Failed to decode NSE financial JSON. Request may have been blocked.")
        
    return df

def fetch_cogencis_ownership_data() -> Dict[str, pd.DataFrame]:
    """
    Scrapes all tables from the Cogencis ownership data page.
    """
    scraped_data = {}
    print(f"Fetching ownership data from Cogencis...")
    
    # Introduce small delay
    time.sleep(2)
    
    try:
        response = requests.get(COGENCIS_OWNERSHIP_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for all tables on the page
        tables = soup.find_all('table')
        
        if not tables:
            scraped_data['Error'] = "No tables found on the Cogencis ownership page. Data is likely loaded dynamically."
            return scraped_data
            
        # Iterate over all found tables
        for i, table in enumerate(tables):
            # Use Pandas to read the table HTML into a DataFrame
            # This is the fastest way to get tabular data from BeautifulSoup object
            try:
                # pandas.read_html returns a list of DataFrames; we assume one table per element
                df_list = pd.read_html(str(table))
                if df_list:
                    df = df_list[0]
                    # Attempt to find a suitable header/title near the table
                    title_tag = table.find_previous(['h1', 'h2', 'h3', 'h4', 'p'])
                    table_title = title_tag.text.strip() if title_tag and len(title_tag.text.strip()) > 5 else f"Ownership Table {i+1}"
                    
                    # Store only non-empty DataFrames
                    if not df.empty:
                        # Clean up multi-index headers that sometimes result from scraping
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [' '.join(col).strip() for col in df.columns.values]
                        scraped_data[table_title] = df
            except Exception as e:
                # print(f"Could not parse table {i+1}: {e}")
                continue # Skip tables that fail to parse

        if not scraped_data:
            scraped_data['Error'] = "Scraping failed to extract any meaningful tables from Cogencis."

    except requests.RequestException as e:
        scraped_data['Error'] = f"Failed to fetch Cogencis data: {e}"
        
    return scraped_data

def generate_word_document(
    quote_data: Dict[str, Any], 
    financial_data: pd.DataFrame, 
    ownership_data: Dict[str, pd.DataFrame], 
    filename: str
):
    """Generates a professional Word document with all collected data."""
    document = Document()
    
    # Set document title style
    style = document.styles['Normal']
    font = style.font
    font.name = 'Arial'
    font.size = Pt(11)

    # Main Title
    document.add_heading(f"Comprehensive Stock Analysis Report: {TICKER}", 0)
    document.add_paragraph(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    document.add_paragraph(f"NSE Symbol: {TICKER} | ISIN: {ISIN}").add_run().bold = True
    document.add_paragraph("---")

    # --- Section 1: NSE Live Quote and Valuation Data ---
    document.add_heading("1. Live Market and Valuation Metrics (NSE)", level=1)
    if 'Latest Price Data' in quote_data:
        data = quote_data['Latest Price Data']
        
        p = document.add_paragraph()
        p.add_run(f"Company: {data['Company Name']}\n").bold = True
        p.add_run(f"Last Updated: {data['Latest Trade Date']}\n")
        p.add_run(f"Adjusted TTM P/E: {data['Adjusted P/E (TTM)']}\n")
        p.add_run(f"Total Traded Volume: {data['Total Traded Volume']:,}\n")
        p.add_run(f"Total Traded Value (Crores): ₹{data['Total Traded Value (Cr)']:,}")
    else:
        document.add_paragraph("Live market data could not be retrieved from NSE.")

    # --- Section 2: Quarterly Financial Results (NSE) ---
    document.add_heading("2. Quarterly Financial Results Comparison (₹ Crores)", level=1)
    if not financial_data.empty:
        # Convert DataFrame to Word table
        t = document.add_table(financial_data.shape[0] + 1, financial_data.shape[1] + 1)
        t.style = 'Light Shading Accent 1'
        
        # Add Header row (Categories)
        t.cell(0, 0).text = 'Metric'
        for j, col_name in enumerate(financial_data.columns):
            t.cell(0, j + 1).text = col_name
            t.cell(0, j + 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        
        # Add Data rows
        for i, (index_name, row) in enumerate(financial_data.iterrows()):
            t.cell(i + 1, 0).text = index_name
            for j, value in enumerate(row):
                t.cell(i + 1, j + 1).text = f"{value:,}"
                t.cell(i + 1, j + 1).paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    else:
        document.add_paragraph("Quarterly financial comparison data could not be retrieved from NSE.")
    
    # --- Section 3: Cogencis Ownership Data ---
    document.add_heading(f"3. Ownership and Capital History (Source: Cogencis)", level=1)
    document.add_paragraph(f"Data scraped from: {COGENCIS_OWNERSHIP_URL}")

    if 'Error' in ownership_data:
        document.add_paragraph(f"Error scraping Cogencis data: {ownership_data['Error']}")
    elif ownership_data:
        for title, df in ownership_data.items():
            document.add_heading(f"3.{list(ownership_data.keys()).index(title) + 1} {title}", level=2)
            
            # Generate table from DataFrame
            doc_table = document.add_table(df.shape[0] + 1, df.shape[1])
            doc_table.style = 'Table Grid'
            
            # Add Header Row
            for j, col in enumerate(df.columns):
                doc_table.cell(0, j).text = str(col)
                doc_table.cell(0, j).paragraphs[0].runs[0].font.bold = True
            
            # Add Data Rows
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    # Convert non-string data to string for docx cell
                    doc_table.cell(i + 1, j).text = str(df.iloc[i, j])
    else:
        document.add_paragraph("No ownership or capital history tables were successfully scraped from Cogencis.")
        
    document.save(filename)
    print(f"\n--- SUCCESS ---")
    print(f"Report generated successfully: {filename}")
    print(f"Please open the file for the neatly arranged data.")

# --- Main Execution ---
if __name__ == "__main__":
    nse_session = create_nse_session()
    
    if nse_session:
        # 1. Fetch data from NSE Quote API
        quote_data_result = fetch_nse_quote_data(nse_session)
        
        # 2. Fetch quarterly financial data from NSE
        financial_df = fetch_nse_financial_data(nse_session)
        
        # 3. Fetch ownership data from Cogencis
        ownership_data_result = fetch_cogencis_ownership_data()

        # 4. Generate the Word Document
        generate_word_document(
            quote_data_result,
            financial_df,
            ownership_data_result,
            OUTPUT_FILENAME
        )
    else:
        print("Could not initialize the required session to fetch data. Exiting.")
