# Multi-Agent Travel Planning Prototype (Python)

This prototype demonstrates a modular travel-planning system powered by multiple AI agents. Each agent is responsible for a specific task—weather scoring, hotel recommendations, itinerary creation, and summary email generation—using a mix of classical machine learning, sentence embeddings, and a text-generation model.

## Quick start

Create and activate a virtual environment (recommended):

```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the demo:

```
python main.py
```

## Project Breakdown

### Entrypoint

- main.py

### Utility

- safe_generate: A wrapper around a Hugging Face text-generation pipeline to prevent overflow issues.

### Agents

- WeatherAnalysisAgent — Trains a RandomForestRegressor on simple historical data to score travel months.

- HotelRecommenderAgent — Uses SentenceTransformers (all-MiniLM-L6-v2) to embed hotel descriptions and find the closest match to user preferences.

- ItineraryPlannerAgent — Generates a travel itinerary using a small transformer text model (e.g., GPT-2).

- SummaryAgent — Creates a final client-friendly email and estimates total cost via text generation.

### Coordinator

- TravelPlanningSystem — Executes and aggregates results from all agents to produce the final trip plan.

