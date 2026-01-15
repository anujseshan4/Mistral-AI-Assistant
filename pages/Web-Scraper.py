import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

#Step 1: Load Important Libraries
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import webbrowser
import asyncio
import logging

# Step 2.1 - Defining CSS for Streamlit App
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

def matrix_background():
    # This combines the HTML and the necessary CSS to make it a background
    st.markdown(
        """
        <style>
        .matrix-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: black;
            overflow: hidden;
            z-index: -1; /* This puts it BEHIND your buttons/text */
            color: #0F0;
            font-family: monospace;
            font-size: 20px;
            opacity: 0.3; /* Makes it subtle so you can read your app */
        }
        .matrix-column {
            position: absolute;
            top: -1000px;
            writing-mode: vertical-rl;
            text-orientation: upright;
            animation: fall linear infinite;
        }
        @keyframes fall {
            0% { top: -100%; }
            100% { top: 100%; }
        }
        /* Make Streamlit background transparent to see the matrix */
        .stApp {
            background: transparent;
        }
        </style>
        
        <div class="matrix-container">
            """ +
            "".join(
                f"<div class='matrix-column' style='left:{i*4}%; animation-duration:{5 + i%5}s; animation-delay:{-i}s;'>"
                + "".join(str((i*j) % 10) for j in range(40))
                + "</div>"
                for i in range(25)
            ) +
        """
        </div>
        """,
        unsafe_allow_html=True
    )

# Just call this at the top of your app
matrix_background()


# Step 2: Streamlit UI

st.title("Web Scraping 🕷️🔥")
st.write("---")

selected_web_scraper = st.selectbox(
    "Choose your Web Scraping Tool:",
    ["FireCrawl", "Crawl4AI"]
)

# =====================================================
# 🔥 FireCrawl
# =====================================================
if selected_web_scraper == "FireCrawl":
    st.subheader("FireCrawl 🔥")

    if st.button("Open FireCrawl Docs"):
        webbrowser.open("https://docs.firecrawl.dev/introduction")
        webbrowser.open("https://firecrawl.dev/")

    scrape_site = st.text_input("Enter URL to scrape:")

    if scrape_site:
        FIRECRAWL_API_KEY = st.text_input("Enter your FireCrawl API Key: ", type="password", key="firecrawl_api_key")
        firecrawl = FireCrawl.Firecrawl(api_key=FIRECRAWL_API_KEY)
        scraped_data = firecrawl.scrape(scrape_site, formats=["markdown"])

        if hasattr(scraped_data, "markdown"):
            content = scraped_data.markdown
        elif isinstance(scraped_data, dict):
            content = scraped_data.get("markdown", str(scraped_data))
        else:
            content = str(scraped_data)

        st.text_area("Scraped Content", content, height=400)

        st.download_button(
            "Download Markdown",
            content,
            file_name="firecrawl_output.md",
            mime="text/markdown"
        )

# =====================================================
# 🕷️ Crawl4AI
# =====================================================
elif selected_web_scraper == "Crawl4AI":
    st.subheader("Crawl4AI 🕷️")

    from crawl4ai import AsyncWebCrawler

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(
            asyncio.WindowsProactorEventLoopPolicy()
        )

    async def crawl(url):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            return result.markdown

    def crawl_sync(url):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(crawl(url))
        finally:
            loop.close()

    url = st.text_input("Enter URL to scrape")

    if st.button("Scrape with Crawl4AI"):
        with st.spinner("Scraping website..."):
            try:
                content = crawl_sync(url)
                st.text_area("Scraped Content", content, height=400)

                st.download_button(
                    "Download Markdown",
                    content,
                    file_name="crawl4ai_output.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"Scraping failed: {e}")
