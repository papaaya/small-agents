import asyncio
from dataclasses import dataclass
from typing import Any
from openai.types.shared import responses_model
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from devtools import debug
import logfire
import urllib.parse
from httpx import AsyncClient

import dotenv

dotenv.load_dotenv()

@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None

class WeatherToolResponse(BaseModel):
    temperature: str = Field(description="The temperature in the location")
    description: str = Field(description="The description of the weather in the location")

class LocationToolResponse(BaseModel):
    lat: float = Field(description="The latitude of the location")
    lng: float = Field(description="The longitude of the location")


class MultiLocationAgentResponse(BaseModel):
    locations: list[str] = Field(description="The list of locations")
    list_weather: list[WeatherToolResponse] = Field(description="The list of weather for each location")

    def as_dict(self):
        return dict(zip(self.locations, self.list_weather))

class MultiLocationAgentDictResponse(BaseModel):
    weather_by_location: dict[str, WeatherToolResponse] = Field(description="multiple locations with weather data")


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


agent = Agent(
    "google-gla:gemini-2.0-flash",
    system_prompt="""
     You are a helpful assistant that provides accurate weather information for a given location. 
     Also, check if there are multiple locations, and if so, provide the weather for each location.     
     """,
    output_type=MultiLocationAgentDictResponse,
)

@agent.tool
async def get_weather(ctx: RunContext, lat: float, lng: float) -> WeatherToolResponse:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        return WeatherToolResponse(temperature="22°C", description="sunny") #default value

    params = {
        "apikey": ctx.deps.weather_api_key,
        "location": f"{lat},{lng}",
        "units": "metric"
        
    }
    with logfire.span(f"get_weather_at_location", params=params) as span:
        r = await ctx.deps.client.get("https://api.tomorrow.io/v4/weather/realtime", params=params)
        r.raise_for_status()
        data = r.json()

    values = data['data']['values']
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    
    return WeatherToolResponse(
        temperature=f'{values["temperatureApparent"]:0.0f}°C',
        description=code_lookup.get(values['weatherCode'], 'Unknown'),
    )


@agent.tool
async def get_location(ctx: RunContext, location: str = Field(description="The location to get information for")) -> LocationToolResponse:

    if ctx.deps.geo_api_key is None:
        return LocationToolResponse(lat=35.6895, lng=139.6917) #default value
    
    params = {
        "q": location,
        "api_key": ctx.deps.geo_api_key,
    }
    loc = urllib.parse.quote(location)
    
    with logfire.span(f"get_location_coordinates", params=params) as span:
        r = await ctx.deps.client.get(f"https://geocode.maps.co/search", params=params)
        r.raise_for_status()
        data = r.json()
        
        if data and len(data) > 0:
            lat = float(data[0]['lat'])
            lng = float(data[0]['lon'])
            span.set_attribute("location.lat", lat)
            span.set_attribute("location.lng", lng)
            return LocationToolResponse(lat=lat, lng=lng)
        else:
            raise ModelRetry('Could not find the location')

@agent.tool
async def get_thinking(ctx: RunContext, 
                user_query: str = Field(description="The user's query"), 
                list_of_tools: list[str] = Field(description="The list of tools to use")) -> str:
    """Allow the agent to think through problems step by step."""

    think_agent = Agent(
        "google-gla:gemini-2.0-flash",
        system_prompt=f"""
        You can think about the user's query and provide a plan to answer the query.
        You can use the following tools
        {list_of_tools}
        """
    )
    response = await think_agent.run(
        f"User query: {user_query}"
    )
    return response.output

async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        # create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key =  dotenv.get_key(dotenv_path='.env', key_to_get='WEATHER_API_KEY')
        print(weather_api_key)
        # create a free API key at https://www.mapbox.com/

        geo_api_key = dotenv.get_key(dotenv_path='.env', key_to_get='GEO_API_KEY')
        print(geo_api_key)
        mydeps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        
        result = await agent.run(
            'Hi, I am John Doe, What is the weather like in delhi, mumbai and seattle? ', deps=mydeps

        )
        #Also, can you plan a trip to the locations? A detailed plan with the weather and the best time to visit each location.
        debug(result)
        print('Response:', result.output)
        result.new_messages()

        result = await agent.run(
            'what is the name of user?', deps=mydeps, 
            messages=result.new_messages()
        )
        debug(result)
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())
