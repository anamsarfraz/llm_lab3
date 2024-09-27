from dotenv import load_dotenv
import os
import json
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket, confirm_ticket_purchase, get_reviews

import traceback

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful movie chatbot that helps people explore movies that are out in \
theaters. 
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_now_playing_movies",
            "description": "Get the movies that are playing now in theaters. Call this whenever the user wants to what are the movies showing in theaters, for example when a customer asks 'What are the movies playing now?'",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_showtimes",
            "description": "Get the showtimes for a movie. Call this whenever the user wants to know the showtimes for a movie, for example when a customer asks 'What are the showtimes for Avengers: Endgame? near San Francisco'",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the movie.",
                    },
                    "location": {
                        "type": "string",
                        "description": "The location of the theater, for example 'San Francisco'.",
                    },
                },
                "required": ["title", "location"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buy_ticket",
            "description": "Buy a ticket for a movie. Call this whenever the user wants to buy a ticket for a movie, for example when a customer asks 'I want to buy a ticket for Avengers: Endgame at 7pm. Is that available?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {   
                        "type": "string",
                        "description": "The name of the theater, for example 'AMC Metreon 16'",
                    },
                    "movie": {
                        "type": "string",
                        "description": "The title of the movie, for example 'Avengers: Endgame'",
                    },
                    "showtime": {
                        "type": "string",   
                        "description": "The showtime for the movie, for example '7pm'",
                    },
                },
                "required": ["theater", "movie", "showtime"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_ticket_purchase",
            "description": "Confirm a ticket purchase. Call this whenever the user wants to confirm a ticket purchase",
            "parameters": {
                "type": "object",
                "properties": {
                    "theater": {
                        "type": "string",
                        "description": "The name of the theater, for example 'AMC Metreon 16'",
                    },
                    "movie": {
                        "type": "string",
                        "description": "The title of the movie, for example 'Avengers: Endgame'",
                    },
                    "showtime": {
                        "type": "string",   
                        "description": "The showtime for the movie, for example '7pm'",
                    },
                },
                "required": ["theater", "movie", "showtime"],
                "additionalProperties": False,
            }
        }
    },
]

REVIEW_PROMPT = """\
Based on the conversation, determine if the topic is about a specific movie. Determine if the user is asking a question that would be aided by knowing what critics are saying about the movie. Determine if the reviews for that movie have already been provided in the conversation. If so, do not fetch reviews.
"""
review_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_reviews",
            "description": "Evaluate the conversation, and determine if the user is asking for reviews for a movie. If so, provide the ID of the movie. If not, return null.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "ID of the movie to fetch reviews for"
                    },
                    "movie_title": {
                        "type": "string",
                        "description": "Title of the movie to fetch reviews for"
                    }
                },
                "required": ["movie_id"],
                "additionalProperties": False,
            }
        }
    },
]

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def handle_tool_calls(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    function_data = {"name": [], "arguments": []}

    # Commenting out the send() call to handle sending empty 
    #await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, tools=tools, stream=True, **gen_kwargs)
    async for part in stream:
        if part.choices[0].delta.tool_calls:
            tool_call = part.choices[0].delta.tool_calls[0]
            function_name = tool_call.function.name or ""
            arguments = tool_call.function.arguments or ""
            function_data["name"].append(function_name)
            function_data["arguments"].append(arguments)
        
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    function_data["name"] = ''.join(function_data["name"])
    function_data["arguments"] = ''.join(function_data["arguments"])
    return response_message, function_data


@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@observe
async def check_for_review_call(client, message_history, gen_kwargs):
    print("Checking for review call")
    response = await client.chat.completions.create(
        messages=[{"role": "system", "content": REVIEW_PROMPT}]+message_history[1:],
        tools=review_tools,
        **gen_kwargs
    )
    
    context_response = response.choices[0].message.content
    print("Response text for review call: ", context_response)
    print("Response: ", response)
    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        print("Tool call: ", tool_call)
        arguments = json.loads(tool_call.function.arguments)
        function_name = tool_call.function.name
        if function_name == "get_reviews":
            reviews = get_reviews(arguments.get('movie_id'))
            reviews_content = f"Reviews for {arguments.get('movie_title', '')} (ID: {arguments.get('movie_id', '')}):\n\n{reviews}"
            context_message = {"role": "system", "content": f"CONTEXT: {reviews_content}"}
            message_history.append(context_message)
       

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    # Check for review call first
    await check_for_review_call(client, message_history, gen_kwargs)
    
    response_message, function_data = await handle_tool_calls(client, message_history, gen_kwargs)
    print("Function data: ", function_data)
    print("Response text: ", response_message.content)
    if response_message.content:
        message_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("message_history", message_history)

    while function_data["name"]:
        function_name = function_data["name"]
        if arguments := function_data["arguments"]:
            arguments = json.loads(arguments)
        print("Function name: ", function_name)
        print("Arguments: ", arguments)
        if function_name == "get_now_playing_movies":
            now_playing_movies = get_now_playing_movies()
            print("Now playing movies: added to message history")
            message_history.append({"role": "system", "content": now_playing_movies})
        elif function_name == "get_showtimes":
            showtimes = get_showtimes(**arguments)
            message_history.append({"role": "system", "content": showtimes})
            print("Showtimes: added to message history")
        elif function_name == "buy_ticket":
            purchase_ticket = buy_ticket(**arguments)
            message_history.append({"role": "system", "content": purchase_ticket})
            print("Purchase ticket: added to message history")
        elif function_name == "confirm_ticket_purchase":
            confirm_purchase = confirm_ticket_purchase(**arguments)
            message_history.append({"role": "system", "content": confirm_purchase})
            print("Confirm purchase: added to message history")
        else:
            break
        # Generate a response to the user with the added system messages 
        response_message, function_data = await handle_tool_calls(client, message_history, gen_kwargs)
        print("Function data in loop: ", function_data)
        print("Response text in loop: ", response_message.content)
        if response_message.content:
            message_history.append({"role": "assistant", "content": response_message.content})
            cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()
