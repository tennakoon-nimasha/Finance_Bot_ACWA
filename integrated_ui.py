import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
import psycopg2
from urllib.parse import urlparse
import time

from langsmith.wrappers import wrap_openai
from langsmith import traceable
 
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Financial Insights Assistant",
    page_icon="💰",
    layout="wide"
)

# Apply custom styling
st.markdown("""
<style>
/* Main app styling */
.stApp {
    background-color: #f5f7fa;
}

/* Custom component styling */
.reasoning-box {
    background-color: #e1f5fe;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #03a9f4;
}

.sql-box {
    background-color: #f5f5f5;
    border-radius: 5px;
    padding: 15px;
    font-family: monospace;
    white-space: pre-wrap;
    overflow-x: auto;
    border-left: 4px solid #607d8b;
}

.thinking-box {
    background-color: #fff8e1;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    border-left: 4px solid #ffc107;
}

.model-info-box {
    background-color: #f1f8e9;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    border-left: 4px solid #8bc34a;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPASBASE_CONNECTION_STRING = os.getenv("SUPASBASE_CONNECTION_STRING")
DB_URL = SUPASBASE_CONNECTION_STRING

# Initialize session state for chat history and settings
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'process_state' not in st.session_state:
    st.session_state.process_state = 0
if 'model' not in st.session_state:
    st.session_state.model = "gpt-4.1"
if 'reasoning_effort' not in st.session_state:
    st.session_state.reasoning_effort = "medium"
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = {}

client = OpenAI()

# @traceable
def call_llm(prompt):
    """
    Enhanced call_llm function that supports both standard and reasoning models
    """
    # Get settings from session state
    model = st.session_state.model
    reasoning_effort = st.session_state.reasoning_effort
    
    # Determine if this is a reasoning model
    reasoning_models = ["o3", "o4-mini", "o3-mini"]
    is_reasoning_model = model in reasoning_models
    
    if is_reasoning_model:
        try:
            # Use the responses.create API for reasoning models
            response = client.responses.create(
                model=model,
                reasoning={"effort": reasoning_effort},
                input=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )
            
            # Store token usage statistics
            if hasattr(response, 'usage'):
                st.session_state.token_usage = {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                # Add reasoning tokens if available
                if hasattr(response.usage, 'output_tokens_details') and hasattr(response.usage.output_tokens_details, 'reasoning_tokens'):
                    st.session_state.token_usage['reasoning_tokens'] = response.usage.output_tokens_details.reasoning_tokens
            
            # Format response to be compatible with the rest of the code
            formatted_response = type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': response.output_text,
                            'thinking': '' # Reasoning models don't have thinking attribute
                        })
                    })
                ]
            })
            
            return formatted_response
        except Exception as e:
            st.error(f"Error calling reasoning model: {str(e)}")
            # Fallback to GPT-4.1 if there's an error
            st.warning("Falling back to GPT-4.1 due to error with reasoning model.")
            model = "gpt-4.1"
            is_reasoning_model = False
    
    if not is_reasoning_model:
        # Use the standard chat completions API for non-reasoning models
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Store token usage statistics
        if hasattr(response, 'usage'):
            st.session_state.token_usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        return response

# Agent prompts
REASONING_AGENT_PROMPT = """
You are a specialized financial data analyst. Your task is to:
 
1. Analyze the user's natural language query about financial data
2. Review the provided database schema and context index
3. Create a detailed, step-by-step reasoning plan for how to answer the query with SQL
 
## Guidelines:
- We are working with a single PostgreSQL table named 'acwa_finance'
- Break down the problem into clear logical steps
- Identify which columns from the schema will be needed
- Explain any calculations, aggregations, or filtering that will be required
 
## Using Financial Data Speficiations:
 
1. Cumulative Closing Balances, All values in the "Closing Balance" column are cumulative from January of the same year.
   - To compute values for a specific month, subtract the previous month's closing balance from the current month's closing balance.
   - To compute a year-to-date value, simply use the closing balance from December (or the last available period of the year).
 
2. Use the Month coloumn to filter the data for the specific month/year of interest.
 
3. Always Use Closing Balance For all financial calculations, the "Closing Balance" column should be used.
 
4. Adjusted Trial Balance in December:
   - For the month of December, if an adjusted trial balance exists, use it instead of the regular one.
   - Adjusted trial balances can be identified from the "File Name" column.
     - Adjusted: contains the substring `AdjPeriod` (e.g., "Trial Balance_Corp_SAR_Dec23_AdjPeriod")
     - Regular: does not contain `AdjPeriod` (e.g., "Trial Balance_Corp_SAR_Dec23")
 
## Financial Concepts Reference:
 
### Basic Column Mappings:
- "Entity" refers to the "Entity" column
- Period Name format is "Jan-25" (Month-Year)
 
### Account column value and Account Grouping value for the Financial Category:
- Revenue: Account in range 40000
- Direct cost: Account in range 50000
- G&A expenses: (Account in range 60000) and Account Grouping is "Other Overheads - G&A costs"
- Finance costs: Account in range 60000 and Account Grouping is "Other Overheads - Finance costs"
- Tax: Account in range 60000 and Account Grouping is "Other Overheads - Tax and zakat"
- Staff costs: Account in range 60000 and Account Grouping is "Other Overheads - Staff"
 
### Financial Statement Categories:
- Income: accounts in range 40000
- Expense: accounts in ranges 50000 & 60000
- Asset: accounts in range 10000
- Liability: accounts in range 20000
- Equity: accounts in range 30000
 
### Verified Common Calculations:
- Profit = Revenue + (Direct cost + G&A expenses + Finance cost + Tax + Staff cost)
- Change in expenses = Current period expenses - Previous period expenses
 
### Special Account Definitions:
1. Long-term intercompany receivables:
   - Account Grouping = "Non current assets"
   - Account number = 11611
   - Intercompany code ≠ "00000"
 
2. Long-term intercompany payables:
   - Account Grouping = "Non current liabilities"
   - Account numbers starting with 21000
   - Intercompany code ≠ "00000"

#Here are some most frequent queries and the logic behind them to answer:
 
Q: Can you tell me the total balance for the long-term intercompany receivables of <<entity>> as of <<date>> ?
Logic : The entity is identified by the "Entity" column in the database.
- The date is specified in the "Month" column in the format "Jan-25".
- Filter the codes by the "Intercompany" column to exclude "00000".
- Filter the "Account" column to include only "11611".
- The balance is the sum of "Closing Balance" column for the filtered rows.
 
Q: What is the change in <<expenses>> for <<month>> of <<entity>>?
Logic :The date is specified in the "Month" column in the format "Jan-25".
-Filter the "Account" column to include values in the range of 50000 and 70000.
-Filter the "Entity" column to include only the specified entity.
-Filter the "Month" column to include only the specified month.
-Calculate the change in expenses by subtracting the previous month's total from the current month's total.
 
Q: Can you confirm whether we have any dividend payables or distributions pending for related parties under <<entity>> as of <<date>>?
Logic : The date is specified in the "Month" column in the format "Jan-25".
- Filter the "Account" column to include values equal to 22111.
- Filter the "Entity" column to include only the specified entity.
- The answer is the sum of the "Closing Balance" column for the filtered rows.
 
Q: Can you update me on the<<income/expense/asset/liability>> for the <<period>> of <<entity>>?
Logic : The date is specified in the "Month" column in the format "Jan-25".
- Filter the "Account" column to include values in the range of 40000-49999 for income, 50000-69999 for expenses, 10000-19999 for assets, and 20000-29999 for liabilities.
- Filter the "Entity" column to include only the specified entity.
- Then answer is the sum of the "Closing Balance" column for the filtered rows.
 
Q: What is the profit, revenue, staff cost and G&A expenses for <<year>>, <<entity>>?
Logic : The date is specified in the "Month" column in the format "Jan-25".
- For the staff cost, filter the "Account" column to include values in the range of 60000 and "Account Grouping" is "Other Overheads - Staff". The answer is the sum of the "Closing Balance" column for the filtered rows.
- For the G&A expenses, filter the "Account" column to include values in the range of 60000 and "Account Grouping" is "Other Overheads - G&A costs". The answer is the sum of the "Closing Balance" column for the filtered rows.
- For the revenue, filter the "Account" column to include values in the range of 40000. The answer is the sum of the "Closing Balance" column for the filtered rows.
- For total cost, filter the "Account" column to include values in the range of 50000 and 60000. The answer is the sum of the "Closing Balance" column for the filtered rows.
- For the profit, use the expression: Profit = Revenue + (Total cost).
- Note that the incomes are negative values in the database and the expenses are positive values.
 
Q: Give comparison of profit, revenue, staff cost and G&A expenses for  to  <<entity>> for <<year1>>, <<year2>>
Logic : Follow the same filtering and calculation rules as the question "What is the profit, revenue, staff cost and G&A expenses for <<year>>, <<entity>>?".
- As the values are cumulative, you will need to filter the values for the month of 'December' for both years.
- Since this is a comparison, you will need to filter the data for both years and calculate the values separately.
- Then, the final answers will be the difference between the values in the two years.
 
## Output Format:
Provide a numbered list of reasoning steps explaining the approach to take.
Each step should be clear and build on the previous ones to form a complete plan.
 
USER QUERY: {user_query}
 
DATABASE CONTEXT INDEX:
{db_context_index}
"""

SQL_GENERATION_PROMPT = """
You are a specialized SQL query generation assistant for financial data. Your task is to:

1. Use the provided reasoning plan to create a SQL query that will answer the user's question
2. Ensure the query follows the reasoning plan exactly
3. Generate complete and executable PostgreSQL SQL code

## Guidelines:
- We are working with a single PostgreSQL table named 'acwa_finance'
- Use advanced SQL operations (GROUP BY, CASE WHEN, etc.) as outlined in the reasoning plan
- ALL data processing must be done within the SQL query itself - do not rely on any post-processing
- Format and structure your query results to directly answer the question
- Double-check all column references against the schema
- Use CTEs (WITH clauses) for complex multi-stage processing
- Use PostgreSQL syntax conventions
- Important domain-specific knowledge:
  - "Entity" refers to "Entity" column
  - In "Account Grouping": income includes "Other income" and "Revenue - Services"
  - In "Account Grouping": expenses include "Dividend paid", "Other Overheads - Consultancy", "Other Overheads - G&A costs", "Other Overheads - Staff"
  - Profile calculation: Income - Expenses
  - Period Name comes as "Jan-25"
  - long-term intercompany recivables defined as, 
        1. In "Account Grouping" choose "Non current assets",
        2. "Account Name" includes "Long term Inter company receivable"
 - When calculating change in expenses for entity,
        1. In "Entity" for one "Month",
        2. Sum up the "Closing Balance" where "Account" number starting from 6

## Output Format:
You must respond with ONLY the SQL query without any additional text, explanations, or formatting.
- Do NOT use markdown code blocks
- Do NOT include any explanations before or after the query
- Return ONLY the bare SQL query that can be executed directly
- Do NOT add semicolons at the end of the query

USER QUERY: {user_query}

DATABASE SCHEMA:
{db_schema}

REASONING PLAN:
{reasoning_plan}

{error_context}
"""

# Result interpretation prompt
RESULT_INTERPRETATION_PROMPT = """
You are a financial analyst expert assistant. The user asked this question:
 
"{user_query}"
 
This SQL query was generated to answer the question:
```sql
{sql_query}
```
 
The query returned {row_count} rows with these columns: {columns}
 
Here's a sample of the data:
```
{data_sample}
```
 
Please provide:
1. A concise, conversational response that directly answers the user's question in natural language
2. 2-3 key insights or observations from the data that would be valuable for business understanding
3. Any relevant context or caveats about the financial interpretation
 
Guidelines:
- Use plain language that a business user without SQL knowledge would understand
- Translate financial terms and concepts into everyday language
- Be precise with numbers and calculations
- Format currency values properly
- Highlight trends, comparisons, or noteworthy patterns
- Keep your response concise and focused
- Do not describe the SQL or database operations
- Focus on the business meaning of the results
 
Your response should read as if a financial advisor is providing an analysis, not a technical explanation of data.
"""

def get_db_schema():
    """
    Returns a formatted string describing the acwa_finance table schema.
    """
    # Since we're working with a fixed schema, we can simply return it directly
    return """Table: acwa_finance
    "File Name" (text)
    "Month" (text)
    "Ledger Name" (text)
    "Combination" (text)
    "Entity" (bigint)
    "Entity Name" (text)
    "Business Line" (bigint)
    "Business Line Name" (text)
    "Cost Center" (text)
    "Cost Center Name" (text)
    "Project" (text)
    "Project Name" (text)
    "Intercompany" (text)
    "Intercompany Name" (text)
    "Account" (bigint)
    "Account Grouping" (text)
    "Account Name" (text)
    "Analytical Code" (text)
    "Analytical Code Name" (text)
    "Region" (text)
    "Region Description" (text)
    "Period Name" (text)
    "Currency" (text)
    "Opening Balance" (text)
    "Total Debit" (text)
    "Total Credit" (text)
    "Closing Balance" (text)
    "Opening Balance_1" (text)
    "Total Debit_1" (text)
    "Total Credit_1" (text)
    "Closing Balance_1" (text)"""

def create_database_context_index(refresh_cache=False):
    """
    Creates a comprehensive context index of the database with real data.
    Extracts column names, data types, and unique values for all potentially important columns.
    Uses caching to avoid repeated expensive queries.
    
    Args:
        refresh_cache: Force refresh the cache even if it exists
        
    Returns:
        A markdown-formatted string with database context information
    """
    # Check cache first if not forcing refresh
    cache_file = "db_context_cache.json"
    if not refresh_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Verify cache has the expected structure
                if all(key in cached_data for key in ["timestamp", "context_markdown"]):
                    # Check if cache is recent (less than 24 hours old)
                    cache_age = time.time() - cached_data["timestamp"]
                    if cache_age < 86400:  # 24 hours in seconds
                        return cached_data["context_markdown"]
        except (json.JSONDecodeError, KeyError):
            # Cache is invalid, continue to rebuild it
            pass
    
    # Connect to the database
    try:
        conn = psycopg2.connect(DB_URL, sslmode='require', options="-c AddressFamily=ipv4")
    except Exception as e:
        # Fallback to minimal context if DB connection fails
        return create_minimal_context_index(str(e))
    
    try:
        # Dictionary to store all the context data
        context_data = {
            "table_name": "acwa_finance",
            "columns": {},
            "categorical_values": {},
            "numeric_stats": {},
            "financial_interpretations": {}
        }
        
        with conn.cursor() as cur:
            # 1. Get all column information
            cur.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'acwa_finance'
                ORDER BY ordinal_position
            """)
            columns_info = cur.fetchall()
            
            # Store column info
            for col_name, data_type, is_nullable in columns_info:
                context_data["columns"][col_name] = {
                    "data_type": data_type,
                    "nullable": is_nullable
                }
            
            # 2. Get row count for reference
            cur.execute("SELECT COUNT(*) FROM acwa_finance")
            total_count = cur.fetchone()[0]
            
            # 3. Identify important categorical and financial columns
            # List of columns that we consider important for financial analysis
            important_columns = [
                "Entity", "Entity Name", "Business Line", "Business Line Name", 
                "Account Grouping", "Account Name", "Period Name", "Currency", 
                "Intercompany", "Intercompany Name", "Region", "Region Description",
                "Month", "Ledger Name", "File Name", "Cost Center", "Cost Center Name",
                "Project", "Project Name", "Analytical Code", "Analytical Code Name"
            ]
            
            # Get unique values for important columns
            for col_name in important_columns:
                if col_name in context_data["columns"]:
                    try:
                        # Get all distinct values for this column
                        cur.execute(f"""
                            SELECT DISTINCT "{col_name}" 
                            FROM acwa_finance 
                            WHERE "{col_name}" IS NOT NULL 
                            ORDER BY "{col_name}"
                        """)
                        values = [row[0] for row in cur.fetchall()]
                        
                        if values:  # Only add if we have values
                            context_data["categorical_values"][col_name] = values
                    except Exception as e:
                        print(f"Error fetching values for {col_name}: {str(e)}")
                        # Continue with other columns if one fails
                        continue
            
            # 4. Get statistics for numeric/financial columns
            # Important financial columns that might contain numeric data
            financial_columns = [
                "Closing Balance", "Opening Balance", "Total Debit", "Total Credit",
                "Closing Balance_1", "Opening Balance_1", "Total Debit_1", "Total Credit_1"
            ]
            
            for col_name in financial_columns:
                if col_name in context_data["columns"]:
                    try:
                        cur.execute(f"""
                            SELECT 
                                MIN(CAST("{col_name}" AS NUMERIC)), 
                                MAX(CAST("{col_name}" AS NUMERIC)),
                                AVG(CAST("{col_name}" AS NUMERIC)),
                                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY CAST("{col_name}" AS NUMERIC))
                            FROM acwa_finance
                            WHERE "{col_name}" ~ '^-?[0-9]*\\.?[0-9]+$'
                        """)
                        min_val, max_val, avg_val, median = cur.fetchone()
                        
                        if min_val is not None:  # Only store if we got results
                            context_data["numeric_stats"][col_name] = {
                                "min": float(min_val) if min_val is not None else None,
                                "max": float(max_val) if max_val is not None else None,
                                "avg": float(avg_val) if avg_val is not None else None,
                                "median": float(median) if median is not None else None
                            }
                    except Exception as e:
                        print(f"Error fetching stats for {col_name}: {str(e)}")
                        # Skip if we have issues with this column
                        continue
        
        # 5. Add domain-specific financial interpretations
        context_data["financial_interpretations"] = {
            "income_categories": ["Other income", "Revenue - Services"],
            "expense_categories": ["Dividend paid", "Other Overheads - Consultancy", 
                                  "Other Overheads - G&A costs", "Other Overheads - Staff"],
            "profit_calculation": "Sum of income categories minus sum of expense categories",
            "long_term_intercompany_receivables": {
                "account_grouping": "Non current assets",
                "account_name_pattern": "Long term Inter company receivable"
            }
        }
        
        # 6. Format as markdown for AI prompt
        context_markdown = format_context_as_markdown(context_data)
        
        # 7. Cache the results
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "context_data": context_data,
                "context_markdown": context_markdown
            }, f)
        
        return context_markdown
        
    except Exception as e:
        # Log the error
        print(f"Error building database context: {str(e)}")
        # Return minimal context if anything fails
        return create_minimal_context_index(str(e))
    finally:
        # Always close the connection
        conn.close()

def format_context_as_markdown(context_data):
    """
    Formats the context data dictionary as a markdown string for AI prompt.
    Includes ALL values for categorical columns without truncation.
    """
    md_sections = ["# Database Context Index"]
    
    # Table info
    md_sections.append(f"\n## Table: {context_data['table_name']}")
    
    # Column info
    md_sections.append("\n## Schema Information")
    for col_name, info in context_data["columns"].items():
        md_sections.append(f"- \"{col_name}\" ({info['data_type']})")
    
    # Categorical values - Include ALL values without truncation
    md_sections.append("\n## Categorical Column Values")
    for col_name, values in context_data["categorical_values"].items():
        # Format the values for display
        if all(isinstance(v, str) for v in values):
            # Text values get quotes
            formatted_values = [f'"{v}"' for v in values]
        else:
            # Non-text values don't need quotes
            formatted_values = [str(v) for v in values]
            
        values_str = ", ".join(formatted_values)
        md_sections.append(f"\n### {col_name}")
        md_sections.append(f"- Values: {values_str}")
        md_sections.append(f"- Total unique values: {len(values)}")
    
    # Numeric stats
    if context_data["numeric_stats"]:
        md_sections.append("\n## Numeric Column Statistics")
        for col_name, stats in context_data["numeric_stats"].items():
            md_sections.append(f"\n### {col_name}")
            for stat_name, value in stats.items():
                if value is not None:
                    md_sections.append(f"- {stat_name.capitalize()}: {value}")
    
    # Financial interpretations
    md_sections.append("\n## Financial Domain Information")
    
    income_categories = context_data["financial_interpretations"]["income_categories"]
    md_sections.append("\n### Income Categories")
    md_sections.append("- " + "\n- ".join(income_categories))
    
    expense_categories = context_data["financial_interpretations"]["expense_categories"]
    md_sections.append("\n### Expense Categories")
    md_sections.append("- " + "\n- ".join(expense_categories))
    
    md_sections.append("\n### Key Calculations")
    md_sections.append(f"- Profit: {context_data['financial_interpretations']['profit_calculation']}")
    
    md_sections.append("\n### Long-term Intercompany Receivables")
    ltir = context_data["financial_interpretations"]["long_term_intercompany_receivables"]
    md_sections.append(f"- Account Grouping: \"{ltir['account_grouping']}\"")
    md_sections.append(f"- Account Name includes: \"{ltir['account_name_pattern']}\"")
    
    # Return the combined markdown
    return "\n".join(md_sections)

def create_minimal_context_index(error_msg):
    """
    Creates a minimal context index when database connection fails.
    """
    return f"""# Database Context Index (Minimal Fallback)

    ## Warning
    Could not connect to database to extract actual schema: {error_msg}

    ## Table: acwa_finance
    - Contains financial data including entities, accounts, and balances

    ## Important Financial Concepts
    - Income includes "Other income" and "Revenue - Services"
    - Expenses include "Dividend paid", "Other Overheads - Consultancy", "Other Overheads - G&A costs", "Other Overheads - Staff"
    - Profit calculation: Income - Expenses
    - Period Name format: "Jan-25"
    - Long-term intercompany receivables are in "Non current assets" with account names containing "Long term Inter company receivable"
    """

def display_progress_steps(process_state):
    """
    Display a visual representation of the processing steps using Streamlit native components
    """
    steps = [
        {"icon": "⚙️", "title": "Analyzing Query", "description": "Understanding your financial question"},
        {"icon": "📈", "title": "Creating Reasoning", "description": "Planning the approach to solve this question"},
        {"icon": "📝", "title": "Collecting Data", "description": "Collecting data based on reasoning"},
        {"icon": "🔍", "title": "Executing the logic", "description": "Processing retrived data from financial database"},
        {"icon": "✨", "title": "Creating Insights", "description": "Analyzing results for meaningful insights"}
    ]
    
    # Progress section title
    st.subheader("Processing your question...")
    
    # Progress bar
    progress_value = (process_state + 1) / len(steps)
    st.progress(progress_value)
    
    # Create columns for each step
    cols = st.columns(len(steps))
    
    # Fill each column with the appropriate step information
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            # Determine the status and styling for this step
            if i < process_state:
                status = "Complete"
                icon_color = "#2196F3"  # Blue
                text_color = "#2196F3"
                icon = "✅"
            elif i == process_state:
                status = "In Progress"
                icon_color = "#4CAF50"  # Green
                text_color = "#4CAF50"
                icon = step["icon"]
            else:
                status = "Pending"
                icon_color = "#9E9E9E"  # Gray
                text_color = "#9E9E9E"
                icon = step["icon"]
            
            # Display the step information with appropriate styling
            st.markdown(f"<h4 style='text-align: center; color: {text_color};'>{icon} {step['title']}</h4>", unsafe_allow_html=True)
            st.caption(f"<div style='text-align: center;'>{step['description']}</div>", unsafe_allow_html=True)
            st.caption(f"<div style='text-align: center; color: {text_color};'>{status}</div>", unsafe_allow_html=True)

def generate_reasoning(user_query, db_schema, db_context_index):
    """
    Uses the selected LLM to generate a reasoning plan based on the user query and schema.
    """
    prompt = REASONING_AGENT_PROMPT.format(
        user_query=user_query,
        db_schema=db_schema,
        db_context_index=db_context_index
    )
    
    # Call LLM to generate a reasoning plan
    response = call_llm(prompt=prompt)
    
    # Extract the reasoning plan
    reasoning_plan = response.choices[0].message.content
    
    # Extract thinking if available
    thinking_content = ""
    if hasattr(response.choices[0].message, 'thinking'):
        thinking_content = response.choices[0].message.thinking
    
    return reasoning_plan, thinking_content

def generate_sql_from_reasoning(user_query, db_schema, reasoning_plan, error_context=""):
    """
    Uses the selected LLM to generate SQL based on the reasoning plan.
    """
    prompt = SQL_GENERATION_PROMPT.format(
        user_query=user_query,
        db_schema=db_schema,
        reasoning_plan=reasoning_plan,
        error_context=error_context
    )
    
    # Call LLM to generate SQL
    response = call_llm(prompt=prompt)
    
    # Extract the SQL
    sql_query = response.choices[0].message.content
    
    # Extract thinking if available
    thinking_content = ""
    if hasattr(response.choices[0].message, 'thinking'):
        thinking_content = response.choices[0].message.thinking
    
    # Clean the response to extract only the SQL code
    sql_query = clean_sql_response(sql_query)
    
    return sql_query, thinking_content

def clean_sql_response(response_text):
    sql_keywords = ["WITH", "SELECT", "INSERT", "UPDATE", "DELETE"]
    response_text = response_text.strip().strip('`').strip()
    for keyword in sql_keywords:
        match = re.search(rf"(?i)(^|\n)\s*{keyword}\b", response_text)
        if match:
            return response_text[match.start():].strip().rstrip(";")
    return response_text

def execute_sql_query(sql_query):
    """
    Executes a SQL query and returns the results
    """
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(DB_URL, sslmode='require', options="-c AddressFamily=ipv4")
 
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                results = cur.fetchall()
               
                # Get column names
                column_names = [desc[0] for desc in cur.description]
               
                # Create a list of dictionaries for the dataframe
                rows = []
                for row in results:
                    rows.append(dict(zip(column_names, row)))
               
                return rows, None
 
    except Exception as e:
        return None, str(e)

def generate_insights(user_query, sql_query, data):
    """
    Generate natural language insights from SQL results using the selected LLM
    """
    if data is None or len(data) == 0:
        return "I didn't find any data that matches your query. Could you try asking in a different way?"
   
    # Convert data to DataFrame for easier handling
    df = pd.DataFrame(data)
   
    # Convert DataFrame to a formatted string representation
    data_sample = df.head(5).to_string()
    row_count = len(df)
    columns = ", ".join(list(df.columns))
   
    prompt = RESULT_INTERPRETATION_PROMPT.format(
        user_query=user_query,
        sql_query=sql_query,
        row_count=row_count,
        columns=columns,
        data_sample=data_sample
    )
   
    # Call LLM to generate insights
    response = call_llm(prompt=prompt)
   
    insights = response.choices[0].message.content
   
    return insights

def create_chat_message(role, content, dataframe=None, reasoning=None):
    """
    Create a chat message with either user or assistant styling using native Streamlit components
    """
    # Create a container for the message
    message_container = st.container()
    
    with message_container:
        # Create a two-column layout for avatar and message
        cols = st.columns([1, 12])
        
        with cols[0]:
            # Display a simple avatar
            if role == "user":
                st.markdown("### 👤")
            else:
                st.markdown("### 🤖")
        
        with cols[1]:
            # Display the message with appropriate styling
            if role == "user":
                st.info(content)
            else:
                st.success(content)
                
                # If there's reasoning, display it in an expander
                if reasoning:
                    with st.expander("View Reasoning Plan", expanded=False):
                        st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)
            
            # If there's a dataframe, display it in an expander
            if dataframe is not None and not dataframe.empty:
                with st.expander("View Data Table", expanded=False):
                    st.dataframe(dataframe, use_container_width=True)

def display_model_info():
    """
    Display current model and token usage information
    """
    reasoning_models = ["o3", "o4-mini", "o3-mini"]
    
    # Create a container for the model info
    with st.expander("Model Information", expanded=False):
        st.markdown(f'<div class="model-info-box">Currently using: <b>{st.session_state.model}</b></div>', unsafe_allow_html=True)
        
        if st.session_state.model in reasoning_models:
            st.markdown(f'<div class="model-info-box">Reasoning Intensity: <b>{st.session_state.reasoning_effort}</b></div>', unsafe_allow_html=True)
            st.info("""
            **About Reasoning Models:**
            Reasoning models like o4-mini and o3-mini "think before they answer," producing a long internal chain of thought.
            - **Low intensity:** Favors speed and economical token usage
            - **Medium intensity:** Balanced approach (recommended)
            - **High intensity:** Favors more complete reasoning for complex problems
            
            Note: Reasoning tokens are billed as output tokens even though they're not visible in the response.
            """)
        
        # Add a note about token usage
        if 'token_usage' in st.session_state and st.session_state.token_usage:
            st.write("**Token Usage (Last Query):**")
            st.write(f"- Input Tokens: {st.session_state.token_usage.get('input_tokens', 'N/A')}")
            st.write(f"- Output Tokens: {st.session_state.token_usage.get('output_tokens', 'N/A')}")
            
            if st.session_state.model in reasoning_models and 'reasoning_tokens' in st.session_state.token_usage:
                st.write(f"- Reasoning Tokens: {st.session_state.token_usage.get('reasoning_tokens', 'N/A')}")
            
            st.write(f"- Total Tokens: {st.session_state.token_usage.get('total_tokens', 'N/A')}")

def show_settings_page():
    """
    Display settings page for LLM model selection and configuration
    """
    st.title("Settings")
    
    # Model selection
    st.subheader("Model Selection")
    
    model_options = {
        "Standard Models": ["gpt-4.1", "gpt-4o"],
        "Reasoning Models": ["o3", "o4-mini", "o3-mini"]
    }
    
    model_category = st.radio(
        "Model Category", 
        options=list(model_options.keys()),
        index=0 if st.session_state.model in model_options["Standard Models"] else 1
    )
    
    selected_model = st.selectbox(
        "Select Model",
        options=model_options[model_category],
        index=model_options[model_category].index(st.session_state.model) if st.session_state.model in model_options[model_category] else 0
    )
    
    # Only show reasoning settings for reasoning models
    if model_category == "Reasoning Models":
        st.subheader("Reasoning Settings")
        
        reasoning_effort = st.select_slider(
            "Reasoning Intensity",
            options=["low", "medium", "high"],
            value=st.session_state.reasoning_effort,
            help="Controls how much thinking the model does before answering. Higher settings provide more thorough analysis but use more tokens."
        )
        
        # Explanation of reasoning intensity
        st.info("""
        **Reasoning Intensity Settings:**
        - **Low**: Faster responses, uses fewer tokens, but may miss details in complex problems
        - **Medium**: Balanced approach suitable for most questions
        - **High**: More thorough analysis for complex questions, uses more tokens
        """)
    else:
        reasoning_effort = "medium"  # Default value for non-reasoning models
    
    # Apply button
    if st.button("Apply Settings", type="primary"):
        st.session_state.model = selected_model
        st.session_state.reasoning_effort = reasoning_effort
        st.success(f"Settings updated! Now using {selected_model} model.")
        st.rerun()
    
    # Model information
    if model_category == "Reasoning Models":
        st.subheader("About Reasoning Models")
        st.markdown("""
        **Reasoning models** like o4-mini and o3-mini are designed to "think before they answer," producing a long internal chain of thought before responding to the user. 
        
        **Key benefits:**
        - Better at complex problem-solving
        - More systematic approaches to multi-step tasks
        - Enhanced analytical capabilities for financial data analysis
        
        **How it works:**
        Reasoning models use "reasoning tokens" to work through the problem step by step (though these steps aren't visible). These tokens count toward your usage but provide deeper analysis.
        """)
    
    # Reset button
    st.subheader("Reset Conversation")
    if st.button("Reset Conversation History", type="secondary"):
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.process_state = 0
        st.success("Conversation history has been reset!")
        st.rerun()

def process_user_query(user_query):
    """
    Process a user query with visual feedback on each step
    """
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_query})
   
    # Start processing
    st.session_state.processing = True
    
    # Create a placeholder for the progress display
    progress_placeholder = st.empty()
    
    try:
        # Step 1: Analyze Query
        st.session_state.process_state = 0
        with progress_placeholder.container():
            display_progress_steps(0)
        time.sleep(0.5)  # Brief delay to allow UI update
        
        # Step 2: Generate Reasoning Plan
        st.session_state.process_state = 1
        with progress_placeholder.container():
            display_progress_steps(1)
        
        db_schema = get_db_schema()
        db_context_index = create_database_context_index()
        reasoning_plan, reasoning_thinking = generate_reasoning(user_query, db_schema, db_context_index)
        
        # Display reasoning plan
        with st.expander("View Reasoning Plan", expanded=True):
            st.markdown(f'<div class="reasoning-box">{reasoning_plan}</div>', unsafe_allow_html=True)
        
        # Step 3: Generate SQL from Reasoning
        st.session_state.process_state = 2
        with progress_placeholder.container():
            display_progress_steps(2)
        
        sql_query, sql_thinking = generate_sql_from_reasoning(user_query, db_schema, reasoning_plan)
        
        # Display SQL query
        with st.expander("View Generated SQL", expanded=False):
            st.markdown(f'<div class="sql-box">{sql_query}</div>', unsafe_allow_html=True)
       
        # Step 4: Execute Query
        st.session_state.process_state = 3
        with progress_placeholder.container():
            display_progress_steps(3)
        data, error = execute_sql_query(sql_query)
       
        # Step 5: Interpret Results
        st.session_state.process_state = 4
        with progress_placeholder.container():
            display_progress_steps(4)
       
        if data is not None:
            df = pd.DataFrame(data)
           
            # Generate natural language insights
            insights = generate_insights(user_query, sql_query, data)
           
            # Add AI message with insights and dataframe to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": insights,
                "dataframe": df,
                "sql_query": sql_query,
                "reasoning": reasoning_plan
            })
        else:
            # Handle error case
            error_message = f"""
            I couldn't find an answer to your question. The database returned an error:
            
            ```
            {error}
            ```
            
            Could you try rephrasing your question?
            """
           
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sql_query": sql_query,
                "reasoning": reasoning_plan
            })
    finally:
        # Reset processing state and clear progress display
        st.session_state.processing = False
        st.session_state.process_state = 0
        progress_placeholder.empty()

def financial_data_pipeline(user_query, max_attempts=3):
    """
    Main function that orchestrates the entire pipeline with retry logic and reasoning step
    """
    progress_container = st.empty()
    
    try:
        # Step 1: Get database schema and context
        db_schema = get_db_schema()
        db_context_index = create_database_context_index()
        
        # Step 2: Generate reasoning plan
        reasoning_plan, reasoning_thinking = generate_reasoning(user_query, db_schema, db_context_index)
        
        # Process with up to 3 attempts
        data = None
        error = None
        sql_query = None
       
        for attempt in range(max_attempts):
            if attempt > 0:
                with progress_container.container():
                    st.warning(f"🔄 Retrying with improved SQL (Attempt {attempt+1}/{max_attempts})")
           
            # Generate SQL with more specific error context after first attempt
            if attempt == 0:
                error_context = ""
            else:
                error_context = f"""
    PREVIOUS ERROR: {error}
     
    Your previous SQL query had errors. Please fix the following issues:
    1. Check for balanced parentheses
    2. Verify column names match exactly what's in the schema (case-sensitive)
    3. Make sure your SQL is valid for PostgreSQL
    4. Check for any unnecessary semicolons or special characters
    5. Ensure the query is focused on the acwa_finance table
    6. Make sure all calculations and data processing are done directly in SQL
    7. Remember to handle NULL or empty string values with NULLIF() and appropriate CAST functions
     
    Return ONLY the corrected SQL query with no explanation.
    """
           
            # Generate SQL from reasoning plan
            sql_query, sql_thinking = generate_sql_from_reasoning(user_query, db_schema, reasoning_plan, error_context)
           
            # Execute SQL
            data, error = execute_sql_query(sql_query)
           
            # If successful, break out of the loop
            if data is not None:
                break
           
            # If we've exhausted all attempts, show an error
            if attempt == max_attempts - 1:
                with progress_container.container():
                    st.error("❌ All attempts failed. Could not generate a working SQL query.")
        
        return {
            "success": data is not None,
            "error": error,
            "sql_query": sql_query,
            "data": data,
            "reasoning_plan": reasoning_plan
        }
    finally:
        progress_container.empty()

def create_improved_chat_ui():
    """
    Create an improved chat UI
    """
    # Container for the chat
    chat_container = st.container()
   
    with chat_container:
        # Display all previous messages
        for message in st.session_state.messages:
            create_chat_message(
                role=message["role"],
                content=message["content"],
                dataframe=message.get("dataframe"),
                reasoning=message.get("reasoning")
            )
 
    # If processing, show the enhanced progress indicator
    if st.session_state.processing:
        display_progress_steps(st.session_state.process_state)

def main():
    """
    Main function that sets up the application UI with settings and chat pages
    """
    # Set up sidebar navigation
    st.sidebar.title("Financial Insights Assistant")
    page = st.sidebar.radio("Navigation", ["Chat", "Settings"])
    
    # Display model info in sidebar
    with st.sidebar:
        display_model_info()
    
    # Display the selected page
    if page == "Chat":
        # App header
        st.title("Financial Insights Assistant 💰")
        
        # Brief description
        st.info("""
        Ask questions about your financial data in plain English. 
        I'll analyze the data and provide insights with detailed reasoning.
        """)
        
        # Add example questions
        with st.expander("Example questions you can ask"):
            st.markdown("""
            - What is the change in expenses for Entity 10101 in Jan-25?
            - What's the total for long-term intercompany receivables of 10202 in Jan-25?
            - Extract all long-term intercompany receivables grouped by entity
            - Show me the top 5 entities by expenses in Jan-25
            - Calculate the profit for each business line in Jan-25
            """)
        
        # Display chat interface
        create_improved_chat_ui()
       
        # Input for user question
        user_query = st.chat_input("Ask about your financial data...")
       
        if user_query:
            # Process the query
            process_user_query(user_query)
            st.rerun()
    
    elif page == "Settings":
        show_settings_page()

# Run the app
if __name__ == "__main__":
    main()
