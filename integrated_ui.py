import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI
import re
import psycopg2
from urllib.parse import urlparse
import time
 
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

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPASBASE_CONNECTION_STRING = os.getenv("SUPASBASE_CONNECTION_STRING")
DB_URL = SUPASBASE_CONNECTION_STRING
 
# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'process_state' not in st.session_state:
    st.session_state.process_state = 0
 
# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
 
client = get_openai_client()
 
# Agent prompts
REASONING_AGENT_PROMPT = """
You are a specialized financial data analyst. Your task is to:

1. Analyze the user's natural language query about financial data
2. Review the provided database schema and context index
3. Create a detailed, step-by-step reasoning plan for how to answer the query with SQL

## Guidelines:
- We are working with a single PostgreSQL table named 'structured_rag'
- Break down the problem into clear logical steps
- Identify which columns from the schema will be needed
- Explain any calculations, aggregations, or filtering that will be required
- Clarify how to handle specific financial concepts like:
  - "Entity" refers to "Entity" column
  - Income includes "Other income" and "Revenue - Services"
  - Expenses include "Dividend paid", "Other Overheads - Consultancy", "Other Overheads - G&A costs", "Other Overheads - Staff"
  - Profile calculation: Income - Expenses
  - Period Name comes as "Jan-25"
  - long-term intercompany recivables defined as, 
        1. In "Account Grouping" choose "Non current assets",
        2. "Account Name" includes "Long term Inter company receivable"
  - When calculating change in expenses for entity,
        1. In "Entity" for one "Month",
        2. Sum up the "Closing Balance" where "Account" number starting from 6

## Output Format:
Provide a numbered list of reasoning steps explaining the approach to take.
Each step should be clear and build on the previous ones to form a complete plan.

USER QUERY: {user_query}

DATABASE SCHEMA:
{db_schema}

DATABASE CONTEXT INDEX:
{db_context_index}
"""

SQL_GENERATION_PROMPT = """
You are a specialized SQL query generation assistant for financial data. Your task is to:

1. Use the provided reasoning plan to create a SQL query that will answer the user's question
2. Ensure the query follows the reasoning plan exactly
3. Generate complete and executable PostgreSQL SQL code

## Guidelines:
- We are working with a single PostgreSQL table named 'structured_rag'
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
    Returns a formatted string describing the structured_rag table schema.
    """
    # Since we're working with a fixed schema, we can simply return it directly
    return """Table: structured_rag
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
 
def create_database_context_index():
    """
    Creates a comprehensive context index of the database with specific information for our financial data.
    """
    context_sections = ["# Database Context Index"]
   
    # Add table information
    context_sections.append("\n## Table: structured_rag")
   
    # Add specific information about field interpretations
    context_sections.append("\n## Important Field Interpretations")
    context_sections.append("- \"Entity\" refers to entity name")
    context_sections.append("- In \"Account Grouping\":")
    context_sections.append("  - Income categories: \"Other income\", \"Revenue - Services\"")
    context_sections.append("  - Expense categories: \"Dividend paid\", \"Other Overheads - Consultancy\", \"Other Overheads - G&A costs\", \"Other Overheads - Staff\"")
    context_sections.append("- Profile calculation: Income - Expenses")
   
    # Add column analysis based on our schema
    context_sections.append("\n### Column Analysis:")
   
    # Add some key columns with sample values
    key_columns = [
        ("Entity Name", "TEXT", ["Company A", "Company B", "Company C"]),
        ("Business Line Name", "TEXT", ["Division 1", "Division 2", "Division 3"]),
        ("Account Grouping", "TEXT", ["Revenue - Services", "Other income", "Other Overheads - Staff", "Other Overheads - G&A costs"]),
        ("Region", "TEXT", ["North America", "Europe", "Asia Pacific"]),
        ("Currency", "TEXT", ["USD", "EUR", "GBP"]),
        ("Closing Balance", "TEXT", ["10000", "25000", "-5000"])
    ]
   
    for col_name, data_type, samples in key_columns:
        context_sections.append(f"\n#### {col_name} ({data_type}):")
        context_sections.append(f"- Sample values: {', '.join(samples)}")
   
    # Add common calculation patterns for financial analysis
    context_sections.append("\n## Recommended Calculation Patterns")
    context_sections.append("- For financial aggregations: Use SUM, AVG, MIN, MAX on monetary columns")
    context_sections.append("- For grouping: Use GROUP BY with categories like 'Entity Name', 'Business Line Name', or 'Account Grouping'")
    context_sections.append("- For profit calculations: Sum income categories minus sum of expense categories")
    context_sections.append("- For time analysis: Filter by 'Month' or 'Period Name'")
    context_sections.append("- For regional analysis: Group by 'Region' or 'Region Description'")
   
    return "\n".join(context_sections)

def display_progress_steps(process_state):
    """
    Display a visual representation of the processing steps using Streamlit native components
    """
    steps = [
        {"icon": "⚙️", "title": "Analyzing Query", "description": "Understanding your financial question"},
        {"icon": "📈", "title": "Creating Reasoning", "description": "Planning the approach to solve this question"},
        {"icon": "📝", "title": "Crafting SQL", "description": "Building a database query based on reasoning"},
        {"icon": "🔍", "title": "Executing Query", "description": "Retrieving data from financial database"},
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
    Uses Claude to generate a reasoning plan based on the user query and schema.
    """
    prompt = REASONING_AGENT_PROMPT.format(
        user_query=user_query,
        db_schema=db_schema,
        db_context_index=db_context_index
    )
    
    # Call Claude to generate a reasoning plan
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet:thinking",
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the reasoning plan
    reasoning_plan = response.choices[0].message.content
    
    # Extract thinking if available
    thinking_content = ""
    if hasattr(response.choices[0].message, 'thinking'):
        thinking_content = response.choices[0].message.thinking
    
    return reasoning_plan, thinking_content

def generate_sql_from_reasoning(user_query, db_schema, reasoning_plan, error_context=""):
    """
    Uses Claude to generate SQL based on the reasoning plan.
    """
    prompt = SQL_GENERATION_PROMPT.format(
        user_query=user_query,
        db_schema=db_schema,
        reasoning_plan=reasoning_plan,
        error_context=error_context
    )
    
    # Call Claude to generate SQL
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet:thinking",
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
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
        conn = psycopg2.connect(DB_URL, sslmode='require')
 
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
    Generate natural language insights from SQL results
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
   
    # Call Claude to generate insights
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        max_tokens=1000,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
   
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
        with st.expander("View Reasoning Plan", expanded=False):
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
    5. Ensure the query is focused on the structured_rag table
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
    Main function that sets up the application UI
    """
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
   
    # Reset button (only show if there are messages)
    if st.session_state.messages:
        reset_col1, reset_col2, reset_col3 = st.columns([1, 1, 2])
        with reset_col1:
            if st.button("Start New Conversation", type="secondary"):
                st.session_state.messages = []
                st.session_state.processing = False
                st.session_state.process_state = 0
                st.rerun()
 
# Run the app
if __name__ == "__main__":
    main()