import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import dotenv
dotenv.load_dotenv()



system_message = """
Follow the steps in the example below to retrieve the weather information requested.

Example:
  Question: What's the weather in Kalamazoo, Michigan?
  Step 1:   The user is asking about Kalamazoo, Michigan.
  Step 2:   Use the latlon_geocoder tool to get the latitude and longitude of Kalamazoo, Michigan.
  Step 3:   latitude, longitude is (42.2917, -85.5872)
  Step 4:   Use the get_weather_from_nws tool to get the weather from the National Weather Service at the latitude, longitude
  Step 5:   The detailed forecast for tonight reads 'Showers and thunderstorms before 8pm, then showers and thunderstorms likely. Some of the storms could produce heavy rain. Mostly cloudy. Low around 68, with temperatures rising to around 70 overnight. West southwest wind 5 to 8 mph. Chance of precipitation is 80%. New rainfall amounts between 1 and 2 inches possible.'
  Answer:   It will rain tonight. Temperature is around 70F.

Question:
"""


async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )

    tools = await client.get_tools()

    agent = create_react_agent(
        "anthropic:claude-sonnet-4-5",
        tools,
        prompt=system_message,
    )

    user_input = "Is it raining now in Chicago?"

    while user_input != "STOP":
        print(f"Invoking agent for {user_input}")

        weather_response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }
        )

        print(weather_response["messages"][-1].content)
        print("Type STOP to exit")
        user_input = input("Q: ")


if __name__ == "__main__":
    asyncio.run(main())