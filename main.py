"""
Travel Planning System with Multi-Agent Architecture

Agents:
1. WeatherAnalysisAgent:
   Uses RandomForestRegressor to score travel months based on basic weather features.

2. HotelRecommenderAgent:
   Embeds hotel descriptions using SentenceTransformers and retrieves top matches
   via semantic similarity with user preferences.

3. ItineraryPlannerAgent:
   Generates a travel itinerary using a GPT-2 text-generation pipeline.

4. SummaryAgent:
   Produces a client-facing email summarizing the trip and estimating costs.

Coordinated by:
TravelPlanningSystem
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def safe_generate(pipe, prompt: str, requested_new_tokens: int) -> str:
    """
    Generate text while ensuring token limits are respected.

    Args:
        pipe: HF text-generation pipeline.
        prompt (str): Input text prompt.
        requested_new_tokens (int): Desired max number of new tokens.

    Returns:
        str: Model-generated text.
    """
    tokenizer = pipe.tokenizer
    model = pipe.model

    max_pos = getattr(model.config, "max_position_embeddings", None) or \
              getattr(model.config, "n_positions", 1024)

    enc = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_len = enc["input_ids"].shape[1]

    allowed_new = max_pos - input_len - 1
    if allowed_new <= 0:
        trunc_len = max(max_pos - 2, 1)
        enc = tokenizer(prompt, return_tensors="pt",
                        truncation=True, max_length=trunc_len, add_special_tokens=True)
        prompt = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
        input_len = enc["input_ids"].shape[1]
        allowed_new = max(max_pos - input_len - 1, 1)

    max_new = min(requested_new_tokens, allowed_new)
    output = pipe(prompt, max_new_tokens=max_new, truncation=True)

    return output[0].get("generated_text") or output[0].get("text", "")


class WeatherAnalysisAgent:
    """Analyzes weather patterns to determine ideal travel months."""

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, historical_data: List[Dict[str, Any]]) -> None:
        X = np.array([[d['month'], d['latitude'], d['longitude']]
                      for d in historical_data])
        y = np.array([d['weather_score'] for d in historical_data])
        self.model.fit(X, y)

    def predict_best_time(self, location: Dict[str, Any]) -> Dict[str, Any]:
        predictions = [
            {
                'month': month,
                'score': float(
                    self.model.predict([[month, location['latitude'], location['longitude']]]).item()
                )
            }
            for month in range(1, 13)
        ]
        return {
            'best_months': sorted(predictions, key=lambda x: x['score'], reverse=True)[:3],
            'location': location
        }


class HotelRecommenderAgent:
    """Recommends hotels using sentence embedding similarity."""

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.hotels_db: List[Dict[str, Any]] = []
        self.hotels_embeddings: np.ndarray | None = None

    def add_hotels(self, hotels: List[Dict[str, Any]]) -> None:
        self.hotels_db = hotels
        descriptions = [h['description'] for h in hotels]
        self.hotels_embeddings = self.encoder.encode(descriptions)

    def find_hotels(self, preferences: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.hotels_db or self.hotels_embeddings is None:
            return []

        pref_embedding = self.encoder.encode([preferences])
        similarities = np.dot(self.hotels_embeddings, pref_embedding.T).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {**self.hotels_db[i], 'similarity': float(similarities[i])}
            for i in top_indices
        ]


class ItineraryPlannerAgent:
    """Creates travel itineraries using a GPT-2 text-generation model."""

    def __init__(self):
        self.planner = pipeline(
            "text-generation",
            model="gpt2",
            pad_token_id=50256
        )

    def create_itinerary(self, destination_info: Dict[str, Any],
                         weather_info: Dict[str, Any],
                         hotel_info: List[Dict[str, Any]],
                         duration: int) -> Dict[str, Any]:

        prompt = (
            f"Create a {duration}-day itinerary for {destination_info['name']}.\n"
            f"Best month to visit: {weather_info['best_months'][0]['month']}.\n"
            f"Hotel: {hotel_info[0]['name']}.\n"
            f"Attractions: {', '.join(destination_info['attractions'])}."
        )

        response = safe_generate(self.planner, prompt, requested_new_tokens=350)

        return {
            'destination': destination_info['name'],
            'duration': duration,
            'itinerary': response
        }


class SummaryAgent:
    """Generates a final trip summary email and cost estimate."""

    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="gpt2",
            pad_token_id=50256
        )

    @staticmethod
    def calculate_total_price(hotel_info: List[Dict[str, Any]],
                              duration: int) -> float:
        return float(hotel_info[0]['price'] * duration + 100 * duration)

    def create_email(self, trip_data: Dict[str, Any], client_name: str) -> Dict[str, Any]:
        total_price = self.calculate_total_price(
            trip_data['recommended_hotels'],
            trip_data['itinerary']['duration']
        )

        prompt = (
            f"Dear {client_name},\n\n"
            f"Here is your personalized travel plan:\n"
            f"Destination: {trip_data['itinerary']['destination']}\n"
            f"Duration: {trip_data['itinerary']['duration']} days\n"
            f"Best month to visit: {trip_data['weather_analysis']['best_months'][0]['month']}\n"
            f"Recommended hotel: {trip_data['recommended_hotels'][0]['name']}\n\n"
            f"Itinerary:\n{trip_data['itinerary']['itinerary']}\n\n"
            f"Estimated Total Cost: ${total_price}\n\n"
            f"Please reach out if you’d like any changes."
        )

        response = safe_generate(self.llm, prompt, requested_new_tokens=450)

        return {
            'email_content': response,
            'total_price': total_price
        }


class TravelPlanningSystem:
    """Coordinates all agents to generate a complete trip plan."""

    def __init__(self):
        self.weather_agent = WeatherAnalysisAgent()
        self.hotel_agent = HotelRecommenderAgent()
        self.itinerary_agent = ItineraryPlannerAgent()
        self.summary_agent = SummaryAgent()

    def setup(self, weather_data: List[Dict[str, Any]],
              hotels_data: List[Dict[str, Any]]) -> None:
        self.weather_agent.train(weather_data)
        self.hotel_agent.add_hotels(hotels_data)

    def plan_trip(self, destination: Dict[str, Any],
                  preferences: str,
                  duration: int,
                  client_name: str) -> Dict[str, Any]:

        weather_analysis = self.weather_agent.predict_best_time(destination)
        recommended_hotels = self.hotel_agent.find_hotels(preferences)
        itinerary = self.itinerary_agent.create_itinerary(
            destination, weather_analysis, recommended_hotels, duration
        )

        summary = self.summary_agent.create_email(
            {
                'weather_analysis': weather_analysis,
                'recommended_hotels': recommended_hotels,
                'itinerary': itinerary
            },
            client_name
        )

        return {
            'weather_analysis': weather_analysis,
            'recommended_hotels': recommended_hotels,
            'itinerary': itinerary,
            'summary': summary
        }


def main() -> None:
    """Run a simple example travel plan."""

    historical_weather_data = [
        {'month': m, 'latitude': 41.90, 'longitude': 12.49, 'weather_score': s}
        for m, s in zip(range(1, 13),
                        [0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                         0.95, 0.9, 0.85, 0.7, 0.6, 0.5])
    ]

    hotels_database = [
        {'name': 'Grand Hotel', 'description': 'Luxury hotel in city center with spa and restaurant.', 'price': 300},
        {'name': 'Boutique Resort', 'description': 'Intimate boutique hotel with personalized service.', 'price': 250},
        {'name': 'City View Hotel', 'description': 'Modern hotel with panoramic city views.', 'price': 200},
    ]

    destination = {
        'name': 'Rome',
        'latitude': 41.90,
        'longitude': 12.49,
        'attractions': ['Colosseum', 'Vatican Museums', 'Trevi Fountain'],
    }

    preferences = (
        "Looking for a luxury hotel in the city center "
        "with spa facilities and fine dining."
    )

    client_name = "John Smith"

    system = TravelPlanningSystem()
    system.setup(historical_weather_data, hotels_database)

    trip_plan = system.plan_trip(destination, preferences, duration=3, client_name=client_name)

    print("\n===== TRAVEL PLAN =====")
    print(f"Client: {client_name}")
    print(f"Destination: {destination['name']}")
    print("\n----- EMAIL PREVIEW -----\n")
    print(trip_plan['summary']['email_content'])
    print("\nEstimated Total Price: $", trip_plan['summary']['total_price'])


if __name__ == "__main__":
    main()
