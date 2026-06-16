import asyncio
import os
import requests
from dotenv import load_dotenv
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool, TurnHandlingOptions
from livekit.plugins import groq, deepgram, silero, cartesia
from livekit import agents
from livekit.agents import ChatContext

load_dotenv()

class SmartAssistant(Agent):

    def __init__(self):
        super().__init__(
            instructions="""
            You are a voice assistant.
            Weather tool is available.
            Use it whenever the user asks for weather information.
            Respond in English.
            """
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user in English and user city is Delhi tell the temperature."
        )

    def get_simple_weather(self, city: str, api_key: str) -> str:
        """Get weather from OpenWeatherMap API.
        
        Args:
            city: The name of the city.
            api_key: OpenWeatherMap API key.
        """
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
        try:
            data = requests.get(url).json()
            temp = data['main']['temp']
            condition = data['weather'][0]['description']
            return f"Temperature: {temp}°C | Condition: {condition}"
        except KeyError:
            return "Invalid city name or API key."

    @function_tool()
    async def get_weather(self, context: RunContext, city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: The name of the city to get weather for.
        """
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            return "Weather API key not configured. Please add WEATHER_API_KEY to your .env file."
        
        result = self.get_simple_weather(city, api_key)
        return result


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=deepgram.TTS(model="aura-asteria-en"),
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            interruption={
                "min_words": 2,                    # avoid single-word false triggers
                "resume_false_interruption": True, # recover from coughs/background noise
                "false_interruption_timeout": 2.0,
            }
        ),
    )

    await session.start(
        room=ctx.room,
        agent=SmartAssistant(),
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))