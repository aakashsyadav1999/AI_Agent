from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

from dotenv import load_dotenv
from phi.model.azure import AzureOpenAIChat
import os

load_dotenv()


azure_model = AzureOpenAIChat(
    id=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(model='text-davinci-003'),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

prompt = """
"You only answer with information from your RAG database.
You don't use your internal knowledge.
If you can't answer with the database, simple return 'I don't know'"
"""

agent = Agent(
    model=azure_model,
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
    instructions=[prompt],
)
agent.print_response("How do I make Pad Thai Goong Sod", stream=True)