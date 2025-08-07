from langchain_community.tools import DuckDuckGoSearchRun
# To get more additional information (e.g. link, source) use DuckDuckGoSearchResults()
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch

# from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

# @tool
def search_tool(query: str) -> TavilySearch:
    """Search in Internet by TavilySearch for a query and return the results."""
    search = TavilySearch(
        max_results=2,
#     # topic="general",
#     # include_answer=False,
#     # include_raw_content=False,
#     # include_images=False,
#     # include_image_descriptions=False,
#     # search_depth="basic",
#     # time_range="day",
#     # include_domains=None,
#     # exclude_domains=None
)
    
    return search.invoke(query)
