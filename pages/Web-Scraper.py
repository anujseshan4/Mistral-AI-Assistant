# Step 1: Imports
import streamlit as st
import asyncio
import sys
import os
import webbrowser
from dotenv import load_dotenv, find_dotenv
import firecrawl as FireCrawl

# Step 2: Load environment variables
env_path = find_dotenv("HOME.env", raise_error_if_not_found=True)
load_dotenv(env_path, override=True)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Step 3: Streamlit UI
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
