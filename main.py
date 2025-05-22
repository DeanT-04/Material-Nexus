

# main.py
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from tools import extract_literature, predict_properties, rank_candidates, design_experiment, validate_experiments, update_model

# Load environment variables from .env file
load_dotenv()

# Set up the LLM using DeepSeek
api_key = os.getenv("DEEPSEEK_API_KEY")
llm = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# Define the tools
tools = [
    extract_literature,
    predict_properties,
    rank_candidates,
    design_experiment,
    validate_experiments,
    update_model
]

# Initialize the agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Run the agent with a prompt
prompt = """
You are a materials discovery agent. Your task is to discover new materials by following this pipeline:
1. Use extract_literature to get a list of materials based on the query.
2. Use predict_properties to predict band gaps for those materials.
3. Use rank_candidates to select materials with band gap between 1.5 and 2.5.
4. Use design_experiment to create experiment plans for the top candidates.
5. Use validate_experiments to simulate validation of those experiments.
6. Use update_model to update the model with new data.
Start with the query: 'materials for solar cells'
"""

result = agent.run(prompt)
print(result)

# tools.py
from langchain.agents import tool
from models import predict_band_gap, actual_band_gap
import arxiv
import time

@tool
def extract_literature(query: str) -> str:
    """Extract materials from literature based on the query using the arXiv API. Returns a comma-separated list of material names."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=3)
        materials = []
        for result in client.results(search):
            # Mock material extraction: for demo, use simple material names derived from titles
            # In a real system, use NLP to extract actual material names from summaries
            material = f"Material_{result.title[:5].replace(' ', '_')}"
            materials.append(material)
            time.sleep(3)  # Respect arXiv API rate limit (1 request every 3 seconds)
        return ", ".join(materials) if materials else "No materials found"
    except Exception as e:
        return f"Error accessing arXiv API: {str(e)}"

@tool
def predict_properties(materials: str) -> str:
    """Predict band gap properties for the given materials. Input is comma-separated list, output is 'Material: band_gap' pairs."""
    if not materials or materials == "No materials found":
        return "No materials to predict properties for"
    materials_list = materials.split(", ")
    predictions = [f"{m}: {predict_band_gap(m)}" for m in materials_list]
    return ", ".join(predictions)

@tool
def rank_candidates(properties: str) -> str:
    """Rank candidates based on band gap between 1.5 and 2.5. Returns comma-separated list of top candidates."""
    if not properties or "No materials" in properties:
        return "No candidates to rank"
    prop_dict = dict(p.split(": ") for p in properties.split(", "))
    candidates = [m for m, bg in prop_dict.items() if 1.5 <= float(bg) <= 2.5]
    return ", ".join(candidates) if candidates else "No candidates in range"

@tool
def design_experiment(candidates: str) -> str:
    """Design experiments for the top candidates. Returns a list of experiment descriptions."""
    if not candidates or "No candidates" in candidates:
        return "No experiments to design"
    experiments = [f"Synthesize {c} and measure band gap" for c in candidates.split(", ")]
    return "; ".join(experiments)

@tool
def validate_experiments(experiments: str) -> str:
    """Validate the experiments with mock data. Returns validation results."""
    if not experiments or "No experiments" in experiments:
        return "No experiments to validate"
    results = []
    for exp in experiments.split("; "):
        material = exp.split()[1]
        bg = actual_band_gap(material)
        results.append(f"{material}: {bg}")
    return ", ".join(results)

@tool
def update_model(results: str) -> str:
    """Update the model with new validation results. For demo, just returns a message."""
    if not results or "No experiments" in results:
        return "No results to update model with"
    return "Model updated with new data"

# models.py
def predict_band_gap(material: str) -> float:
    """Mock model: Predict band gap based on material name length."""
    return len(material) * 0.5

def actual_band_gap(material: str) -> float:
    """Mock 'experimental' band gap: Slightly different from prediction to simulate error."""
    return len(material) * 0.6

# data.py
sample_materials = ["MaterialA", "MaterialB", "MaterialC"]
sample_properties = {"MaterialA": 2.0, "MaterialB": 1.5, "MaterialC": 3.0}