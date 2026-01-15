#Step 1: Load Important Libraries
#%pip install -r requirements.txt
# python --version

import os
import sys
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import streamlit as st
from pathlib import Path
import webbrowser
import asyncio
import logging

# Disable CrewAI telemetry to avoid signal handler errors in Streamlit
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
os.environ["CREWAI_NO_SIGNAL_HANDLERS"] = "1"

# AI Libraries
from mistralai import Mistral
import firecrawl as FireCrawl
import huggingface_hub as hf
from huggingface_hub import HfApi
from crewai import Agent, Task, Crew, LLM
from crewai_tools import (CodeInterpreterTool, WebsiteSearchTool,)

# Step 2: Load Environment Variables
# Extracting the API key from the .env file
env_path = find_dotenv("HOME.env", raise_error_if_not_found=True)
print(find_dotenv("HOME.env", raise_error_if_not_found=True))
load_dotenv(env_path, override=True)

# Step 2.1 - Defining CSS for Streamlit App

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
def load_css(css_file):
    css_path = current_dir / "styles" / "matrix.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def matrix_background():
    st.markdown(
        """
        <div class="matrix">
            """ +
            "".join(
                f"<span style='left:{i*4}%; animation-duration:{5 + i%5}s; animation-delay:{-i}s;'>"
                + "<br>".join(str((i*j) % 10) for j in range(40))
                + "</span>"
                for i in range(25)
            ) +
        """
        </div>
        """,
        unsafe_allow_html=True
    )

load_css("styles/matrix.css")
matrix_background()

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #000000; /* Your desired color */
    }
    </style>
    """,unsafe_allow_html=True)

st.set_page_config( page_title="Le Coup de Main", page_icon="🤝",)
st.sidebar.success("Select a page.")

# Step 3: Initialize Mistral AI client

st.title(" Le Coup de Main 🤝")
st.write("---")
st.write("Bongiurno! Je suis votre assistant virtuel, Comment ca va?")

selection = st.selectbox('Select Topic to understand:', ['Mistral AI 😸', 'HuggingFace 🤗', 'Crewai 👥', 
'FireCrawl 🔥', 'Crawl4Ai 🕷️', 'Tensorflow', 'Databricks', 'Automation Anywhere','Tech Learning 🧠', 
'Google Cloud','AWS Bedrock','Robocorp','Langchain 🦜', 'LlamaIndex'], key = "Initial Selection")

if selection == "Mistral AI 😸":
    st.subheader("Mistral AI 😸")
    MISTRAL_API_KEY = st.text_input("Enter your Mistral AI API Key: ", type="password", key="mistral_api_key_1")
    mistral = Mistral(api_key=MISTRAL_API_KEY)
    models_response = mistral.models.list()
    st.write("For Vibe Code: pip install mistral-vibe")
    selected_model = st.selectbox('Select Mistral Model', [model.id for model in models_response.data], key="model_select_1")
    st.write(f"Model Selected: {selected_model}")
    user_query = st.text_input("How may I help you today?", key="user_query_1")

    if user_query:
        mistral = Mistral(api_key=MISTRAL_API_KEY)
        chat_response = mistral.chat.complete(model= selected_model if selected_model else "mistral-tiny",messages=[{"role": "user", "content": user_query}],
                        max_tokens=200, temperature=0.7, stream = False, n=1, stop=None,)
        st.write("Mistral AI Response:")
        st.write(chat_response.choices[0].message.content)
    
elif selection == "HuggingFace 🤗":
    st.subheader("Hugging Face Model Selector 🤗")
    HUGGINGFACE_API_KEY = st.text_input("Enter your Hugging Face API Key: ", type="password", key="hf_api_key")
    api = HfApi()
    models_iterator = list(api.list_models(limit = 10))
    models_list = list(models_iterator)
    unique_tags = list(set(model.pipeline_tag for model in models_list if model.pipeline_tag))
    selected_tag = st.selectbox('Select Pipeline Tag', unique_tags, key="pipeline_tag_select")
    top_models = list(api.list_models(task=selected_tag, sort="downloads", direction=-1, limit=10))
    
    model_options = [f"{model.modelId} - Likes: {model.likes}" for model in top_models]
    st.selectbox('Select Hugging Face Model', model_options, key="hf_model_select")

    st.bar_chart(pd.DataFrame({
            "Model": [model.modelId for model in top_models],
            "Downloads": [model.downloads for model in top_models]
        }).set_index("Model")
    )

elif selection == "Crewai 👥":
    st.subheader("Crewai 👥")
    webbrowser.open("https://docs.crewai.com/en/tools/overview")
    webbrowser.open("https://docs.crewai.com/en/concepts/agents")
    webbrowser.open("https://docs.crewai.com/en/mcp/overview")
    webbrowser.open("https://huggingface.co/blog/ifahim/multi-agent-generic-doc-gen")
    

elif selection == "FireCrawl 🔥":
    st.subheader("FireCrawl 🔥")
    webbrowser.open("https://docs.firecrawl.dev/introduction")
    webbrowser.open("https://firecrawl.dev/")
    FIRECRAWL_API_KEY = st.text_input("Enter your FireCrawl API Key: ", type="password", key="firecrawl_api_key")
    scrape_site = st.text_input("Enter URL to scrape:", key="firecrawl_url")

    if scrape_site:
        firecrawl = FireCrawl.Firecrawl(api_key=FIRECRAWL_API_KEY)
        scraped_data = firecrawl.scrape(scrape_site, formats=['html', 'markdown'])

        # Extract string content if scraped_data is a dictionary
        if hasattr(scraped_data, "markdown"):
            save_content = scraped_data.markdown
        elif isinstance(scraped_data, dict):
            save_content = scraped_data.get("markdown", str(scraped_data))
        else:
            save_content = str(scraped_data)

        # Save to file
        with open("firecrawl_scraped_data.txt", "w", encoding="utf-8") as file:
            file.write(save_content)

        st.download_button(label="Download Scraped Data", data=save_content, file_name="firecrawl_scraped_data.txt", mime="text/plain", key="download_firecrawl_txt")
        st.download_button(label="View Scraped Data (Markdown)", data=save_content, file_name="firecrawl_scraped_data.md", mime="text/markdown", key="download_firecrawl_md")

elif selection == "Crawl4Ai 🕷️":
    import crawl4ai
    from crawl4ai import AsyncWebCrawler
    st.subheader("Crawl4Ai 🕷️")
    webbrowser.open("https://docs.crawl4ai.com/core/examples/")
    webbrowser.open("https://docs.crawl4ai.com/")

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    async def crawl(url):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
        return result.markdown

    def crawl_sync(url):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(crawl(url))

    url = st.text_input("Enter URL to scrape", key="crawl4ai_url")

    if st.button("Scrape with Crawl4AI", key="crawl4ai_scrape_btn"):
        with st.spinner("Scraping website..."):
            try:
                content = crawl_sync(url)
                st.text_area("Scraped Content", content, height=400, key="crawl4ai_content")

                st.download_button(
                "Download Markdown",
                content,
                file_name="crawl4ai_output.md",
                mime="text/markdown",
                key="download_crawl4ai"
            )
            except Exception as e:
                st.error(f"Scraping failed: {e}")

elif selection == "Tensorflow":
    st.subheader("Tensorflow 🌊")
    webbrowser.open("https://www.tensorflow.org/agents")
    webbrowser.open("https://www.tensorflow.org/tutorials")

elif selection == "Databricks":
    st.subheader("Databricks 📈🧱")
    webbrowser.open("https://docs.databricks.com/aws/en/sql/language-manual/")
    webbrowser.open("https://docs.databricks.com/aws/en/getting-started/etl-quick-start")
    webbrowser.open("https://community.cloud.databricks.com/login.html?tuuid=991a434b-8db9-4511-a6d9-950cb039cfd0")

elif selection == "Google Cloud":
    st.subheader("Google Cloud ☁️")
    webbrowser.open("https://docs.cloud.google.com/sdk/docs/cheatsheet")
    webbrowser.open("https://cloud.google.com/products/dataflow#real-time-etl-and-data-integration")

elif selection == "AWS Bedrock":
    st.subheader("AWS Bedrock 🏔️")
    webbrowser.open("https://aws.amazon.com/bedrock/getting-started/")
    webbrowser.open("https://www.datacamp.com/tutorial/aws-bedrock")
    webbrowser.open("https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html")

elif selection == "Robocorp":
    st.subheader("Robocorp 🤖🦍")
    webbrowser.open("https://sema4.ai/docs/automation")
    webbrowser.open("https://sema4.ai/docs/automation/visual-studio-code")
    webbrowser.open("https://github.com/robocorp/example-whisper-transcribe")

elif selection == "Langchain 🦜":
    st.subheader("Langchain 🦜")
    webbrowser.open("https://docs.langchain.com/oss/python/deepagents/overview")
    webbrowser.open("https://docs.langchain.com/oss/python/langchain/mcp")
    webbrowser.open("https://docs.langchain.com/oss/python/langgraph/overview")
    webbrowser.open("https://docs.langchain.com/oss/python/langchain/quickstart")
    webbrowser.open("https://smith.langchain.com/")

elif selection == "LlamaIndex":
    st.subheader("LlamaIndex 🦙")
    webbrowser.open("https://developers.llamaindex.ai/python/cloud/cookbooks/")
    webbrowser.open("https://developers.llamaindex.ai/python/framework/use_cases/")
    webbrowser.open("https://developers.llamaindex.ai/python/cloud/")

elif selection == "Automation Anywhere":
    st.subheader("Automation Anywhere 🤖")
    webbrowser.open("https://community.cloud.automationanywhere.digital/#/login?next=/bots/repository")
    webbrowser.open("https://upskill.automationanywhere.com/")
    webbrowser.open("https://docs.automationanywhere.com/bundle/enterprise-v2019/page/enterprise-cloud/topics/bot-insight/user/cloud-configuring-automation-anywhere-connector.html")


# Step 3: Download Button to download PDFs and Material

elif selection == "Tech Learning 🧠":
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    print(current_dir)
    pdf_files = [
        ("spaCy", current_dir / "assets" / "datacamp-spaCy_Cheat_Sheet_final.pdf"),
        ("Azure CLI", current_dir / "assets" / "datacamp-Azure_CLI_Cheat_Sheet.pdf"),
        ("PyTorch", current_dir / "assets" / "datacamp-Deep_Learning_with_PyTorch_1.pdf"),
        ("Descriptive Statistics", current_dir / "assets" / "datacamp-descriptive-statistics---cheat-sheet---25.pdf"),
        ("Docker", current_dir / "assets" / "datacamp-Docker_for_Data_Science_Cheat_Sheet_2.pdf"),
        ("Excel", current_dir / "assets" / "datacamp-Excel_Keyboard_Shortcuts_Cheat_Sheet.pdf"),
        ("AWS", current_dir / "assets" / "datacamp-Infographic_AWS_Azure_GCP_Service_Comparison_for_Data_Science_AI_1.pdf"),
        ("Keras", current_dir / "assets" / "datacamp-Keras_Cheat_Sheet_gssmi8.pdf"),
        ("Markdown", current_dir / "assets" / "datacamp-Markdown_Cheat_Sheet.pdf"),
        ("Machine Learning Cheatsheet", current_dir / "assets" / "datacamp-ML+Cheat+Sheet_2.pdf"),
        ("Pandas", current_dir / "assets" / "datacamp-Pandas_Cheat_Sheet.pdf"),
        ("Probability", current_dir / "assets" / "datacamp-Probability_Cheat_Sheet.pdf"),
        ("PySpark SQL", current_dir / "assets" / "datacamp-PySpark_SQL_Cheat_Sheet.pdf"),
        ("Regular Expressions", current_dir / "assets" / "datacamp-Regular_Expressions_Cheat_Sheet.pdf"),
        ("Reshaping Data with tidyR", current_dir / "assets" / "datacamp-Reshaping_data_with_tidyR_in_R.pdf"),
        ("Supervised Machine Learning Models", current_dir / "assets" / "datacamp-Supervised_Machine_Learning_Models.pdf"),
        ("Unsupervised Machine Learning Models", current_dir / "assets" / "datacamp-Unsupervised_Machine_Learning_Models.pdf"),
        ("Power Query PowerBI", current_dir / "assets" / "datacamp-Working_With_Tables_in_Power_Query_M_in_Power_BI.pdf"),
        ("Git", current_dir / "assets" / "datacampgit_cheat_sheet.pdf"),
        ("Pandas Basics", current_dir / "assets" / "pandas-Reshaping_data_with_Python.pdf"),
        ("Hands On Machine Learning", current_dir / "assets" / "2-Aurélien-Géron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-O'Reilly-Media-2019.pdf"),
        ("SQL Guidebook", current_dir / "assets" / "100+ Pages SQL Guidebook.pdf"),
        ("Statistical Learning with R", current_dir / "assets" / "Introduction to Statistical Learning R.pdf"),
        ("Statistical Learning with Python", current_dir / "assets" / "Introduction to Statistical Learning Python.pdf"),
        ("Data Mining Concepts", current_dir / "assets" / "neural in Data mining book.pdf"),
        ("Python for Data Analysis", current_dir / "assets" / "Python for Data Analysis _ data wrangling with Pandas- NumPy- and IPython.pdf"),]

    col1, col2 = st.columns(2)

    for idx, (label, file_path) in enumerate(pdf_files):
        col = [col1, col2][idx % 2]

        with col:
            try:
                with open(file_path, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()

                st.download_button(
                label=label,
                data=PDFbyte,
                file_name=file_path.name,
                mime="application/octet-stream",
                key=f"download_{idx}" 
            )

            except FileNotFoundError:
                st.error(f"File not found: {file_path.name}")
            
# Step 4: SQL Generation with Mistral AI

st.write("---")
st.subheader("SQL Query Generator with Mistral AI 😸")
MISTRAL_API_KEY = st.text_input("Enter your Mistral AI API Key: ", type="password", key="mistral_api_key_sql")
mistral = Mistral(api_key=MISTRAL_API_KEY)
models_response = mistral.models.list()
st.write("For Vibe Code: pip install mistral-vibe")
selected_model = st.selectbox('Select Mistral Model', [model.id for model in models_response.data], key="model_select_sql")
st.write(f"Model Selected: {selected_model}")

# Add toggle options for SQL generation
show_explanation = st.toggle(
    "Show Query Explanation",
    value=True,
    help="Display an explanation of the generated SQL query",
    key="toggle_explanation"
)

show_examples = st.toggle(
    "Show Example Queries",
    value=True,
    help="Display example queries to choose from",
    key="toggle_examples"
)

# Example queries (only shown if toggle is enabled)
selected_example = ""
if show_examples:
    example_queries = [
        "Show all customers from USA",
        "Get total sales by product category",
        "Find top 10 customers by order value",
        "List employees with salary greater than 50000",
        "Show products with low stock (less than 10 items)"
    ]
    
    selected_example = st.selectbox(
        "Or select an example:",
        [""] + example_queries,
        key="example_select"
    )

user_query = st.text_area(
    "Enter your question:",
    value=selected_example if selected_example else "",
    height=150,
    placeholder="e.g., Kindly enter the SQL Query you would like",
    key="sql_query_input"
)

generate_button = st.button("🚀 Generate SQL", type="primary", use_container_width=True, key="generate_sql_btn")

st.subheader("📝 Generated SQL Query")
sql_output = st.empty()
explanation_output = st.empty()

# Function to generate SQL using Mistral
def generate_sql(user_query, api_key, model):
    try:
        client = Mistral(api_key=api_key)
        prompt = f"""You are an expert SQL query generator. 
        Given the following database schema and a natural language question, generate a valid SQL query.

Natural Language Question: {user_query}

Instructions:
1. Generate only the SQL query without any explanation unless asked
2. Use proper SQL syntax
3. Include appropriate JOINs if multiple tables are needed
4. Add comments in the SQL if the query is complex
5. Ensure the query is optimized and follows best practices

SQL Query:"""

        response = client.chat.complete(
            model=model if model else "mistral-tiny",
            messages=[
                {"role": "user", "content": prompt}])

        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response (remove markdown code blocks if present)
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def generate_explanation(sql_query, api_key):
    try:
        client = Mistral(api_key=api_key)
        prompt = f"""Explain the following SQL query in simple terms:

{sql_query}

Provide a brief, clear explanation of what this query does."""

        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# Generate SQL when button is clicked
if generate_button:
    if not MISTRAL_API_KEY:
        st.error("⚠️ Please enter your Mistral API key")
    elif not user_query:
        st.warning("⚠️ Please enter a natural language query")
    else:
        with st.spinner("Generating SQL query..."):
            # Generate SQL
            sql_result = generate_sql(user_query, MISTRAL_API_KEY, selected_model)
            
            with sql_output.container():
                st.code(sql_result, language="sql")
                
                # Copy button
                st.button("📋 Copy Query", key="copy_btn")
            
            # Generate explanation if toggle is enabled
            if show_explanation:
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(sql_result, MISTRAL_API_KEY)
                    
                    with explanation_output.container():
                        st.info(f"**Explanation:** {explanation}")

# Step 5: Crew AI Code Generator

st.write("---")
st.subheader("Crew AI Code Generator 👥")
code_generated = st.selectbox('Select Topic to understand:', ['Mistral AI 😸', 'HuggingFace 🤗', 'Crewai 👥', 
'FireCrawl 🔥', 'Crawl4Ai 🕷️', 'Tensorflow', 'Databricks', 'Automation Anywhere','Tech Learning 🧠', 
'Google Cloud','AWS Bedrock','Robocorp','Langchain 🦜', 'LlamaIndex'], key = "Final Selection")

# Remove any existing OpenAI environment variables to avoid conflicts
os.environ.pop("OPENAI_MODEL_NAME", None)
os.environ.pop("OPENAI_API_KEY", None)

logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = st.text_input("Enter your Gemini API Key: ", type="password", key="gemini_api_key")
llm = LLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY)

# Code Planner Agent
code_planner = Agent(llm=llm, name="Code Planner", role = "Code Planner", backstory="An expert code " \
"planner who has the knowledge of a PHD Level Researcher in python and " \
"can easily break down complex coding tasks into clear and simple steps from the specified library.",
goal="Break down coding tasks into clear steps using {library} library.", 
allow_delegation=False, verbose=True,)

# Code Writer Agent
code_writer = Agent(llm = llm, name="Code Writer", role="Code Writer", backstory="An expert code writer who " \
"writes factually accurate code and feel free to search the web for getting the accurate code " \
"and feel free to search the web for getting the accurate code" \
"You can use wikipedia and stackoverflow to get the code in case you don't know the answer or " \
"any questions which arise about library: {library}", reasoning = True,max_reasoning_attempts=1,
goal = "Write detailed error-proof code using python only.",
allow_delegation=False, verbose=True, tools = [CodeInterpreterTool()],)

# Code Editor
code_editor = Agent(
    role="Code Editor", name="Code Editor", goal="An expert code editor who write fault-proof and " \
    "production-ready code",backstory = "You are a highly talented code developer who enjoys editing " \
    "code and you are supposed to edit the code generated from the " \
    "Writer Agent about {library} and ensure it is in python only and the code should be " \
    "crisp, simple and highly detailed with an explanation for each line of code and " \
    "you are not afraid to use shortcuts to make the code efficient and streamlined.",
    reasoning = True,max_reasoning_attempts=1,tools = [CodeInterpreterTool()],
    allow_delegation=False, verbose=True, llm=llm)

# Code Planning Task
plan = Task(description=(
        "1. Prioritize the latest code, key lines of the code, "
            "and the current python version of that code.\n"
        "2. Have a deep understanding and reasoning of knowing what code to use and why you are using that code.\n"
        "3. Plan detailed and explained code which is perfect for a downloadable notepad and Microsoft Word documented file.\n"
        "4. While planning include relevant data sources. and articles, links to refer to.\n"
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, analysis, "
        "keywords, and resources.",
    agent=code_planner,)

# Code Writing Task
write = Task(description=(
        "1. Write detailed, error-proof code in python only.\n"
        "2. Ensure the code is well-commented and easy to understand.\n"
        "3. Include relevant data sources, articles, and links to refer to.\n"
        "4. The code should be suitable for a downloadable Notepad (.txt) and Microsoft Word document file (.docx). \n"
    ),
    expected_output="A complete code document "
        "with well-structured code, comments, "
        "and references.",
    agent=code_writer,)

# Code Editing Task
edit = Task(description=(
        "1. Edit the generated code to ensure it is fault-proof and production-ready.\n"
        "2. Optimize the code for efficiency and performance.\n"
        "3. Ensure the code adheres to best practices and coding standards.\n"
        "4. The final code should be suitable for a downloadable Notepad (.txt) and Microsoft Word document file (.docx). \n"
    ),
    expected_output="A polished and optimized code document "
        "ready for deployment., "
        "with well-structured production-ready code, comments.",
    agent=code_editor,)

# Create Crew
crew = Crew(
    agents=[code_planner, code_writer, code_editor],
    tasks=[plan, write, edit],
    verbose=True,
    llm=llm
)

if st.button("Generate Code", key="generate_code_btn"):
    st.write("Generated Code:")
    result = crew.kickoff(inputs={"library": code_generated})
    st.code(result.raw, language="python")
    st.download_button(
    label="Download Generated Code as .txt",
    data=result.raw.encode('utf-8'),
    file_name=f"{code_generated}_generated_code.txt",
    mime="text/plain",
    key="download_generated_code"
)