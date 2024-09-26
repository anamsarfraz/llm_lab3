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
You are a helpful assistant that can sometimes answer with a list of movies and ticket purchases.

If you encounter errors, report the issue to the user.

If you need a list of movies, generate a function call in JSON format, as shown below. If you are generating a json response for a function call, do not include any other text.

{
    "function_name": "get_now_playing_movies",
    "rationale": "Explain why you are calling the function"
}

If you need to get showtimes for a movie, generate a function call in JSON format, as shown below. 
{
    "function_name": "get_showtimes",
    "rationale": "Explain why you are calling the function",
    "title": "title of the movie",
    "location": "location of the movie"
}

If you need to purchase a ticket, generate a function call in JSON format, as shown below. 
{
    "function_name": "buy_ticket",
    "rationale": "Explain why you are calling the function",
    "theater": "theater name",
    "movie": "title of the movie",
    "showtime": "showtime of the movie"
}

If you need to confirm a ticket purchase, generate a function call in JSON format, as shown below.
{
    "function_name": "confirm_ticket_purchase",
    "rationale": "Explain why you are calling the function",
    "theater": "theater name",
    "movie": "title of the movie",
    "showtime": "showtime of the movie"
}

"""

REVIEW_PROMPT = """\
Based on the conversation, determine if the topic is about a specific movie. Determine if the user is asking a question that would be aided by knowing what critics are saying about the movie. Determine if the reviews for that movie have already been provided in the conversation. If so, do not fetch reviews.

Your only role is to evaluate the conversation, and decide whether to fetch reviews.

Output the current movie, id, a boolean to fetch reviews in JSON format, and your
rationale. Do not output as a code block.

{
    "movie": "title",
    "id": 123,
    "fetch_reviews": true
    "rationale": "reasoning"
}
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

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
        **gen_kwargs
    )
    
    response_content = response.choices[0].message.content
    print("Response text for review call: ", response_content)
    try:
        context_json = json.loads(response_content)
        if context_json.get("fetch_reviews", False):
            movie_id = context_json.get("id")
            reviews = get_reviews(movie_id)
            reviews = f"Reviews for {context_json.get('movie')} (ID: {movie_id}):\n\n{reviews}"
            context_message = {"role": "system", "content": f"CONTEXT: {reviews}"}
            message_history.append(context_message)
    except json.JSONDecodeError:
        print(f"Error parsing review call: {response_content}")

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    # Check for review call first
    await check_for_review_call(client, message_history, gen_kwargs)
    response_message = await generate_response(client, message_history, gen_kwargs)

    response_message_content = response_message.content.strip()
    print(f"response_message_chat: {response_message_content[:300]}...")
    while response_message_content.startswith('{'):
        try:
            function_call = json.loads(response_message_content)
            if "function_name" in function_call and "rationale" in function_call:
                function_name = function_call.get("function_name")
                rationale = function_call.get("rationale")
                if function_name == "get_now_playing_movies":
                    movies = get_now_playing_movies()
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\nMovies: {movies}"})
                    # Generate a response to the user
                    response_message = await generate_response(client, message_history, gen_kwargs)
                    response_message_content = response_message.content.strip()
                    print(f"response_message_chat: {response_message_content[:300]}...")
                elif function_name == "get_showtimes":
                    title = function_call.get("title")
                    location = function_call.get("location")
                    showtimes = get_showtimes(title, location)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\nShowtimes: {showtimes}"})
                    # Generate a response to the user
                    response_message = await generate_response(client, message_history, gen_kwargs)
                    response_message_content = response_message.content.strip()
                    print(f"response_message_chat: {response_message_content[:300]}...")
                elif function_name == "buy_ticket":
                    print("Ticket purchase in queue: ", message_history)
                    theater = function_call.get("theater")
                    movie = function_call.get("movie")
                    showtime = function_call.get("showtime")
                    ticket_purchase = buy_ticket(theater, movie, showtime)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\nTicket purchase in queue: {ticket_purchase} Please conirm if you want to purchase the ticket."})
                    # Generate a response to the user
                    response_message = await generate_response(client, message_history, gen_kwargs)
                    response_message_content = response_message.content.strip()
                    print(f"response_message_chat: {response_message_content[:300]}...")
                elif function_name == "confirm_ticket_purchase":
                    print("Confirm ticket purchase: ", message_history)
                    theater = function_call.get("theater")
                    movie = function_call.get("movie")
                    showtime = function_call.get("showtime")
                    ticket_purchase = confirm_ticket_purchase(theater, movie, showtime)
                    message_history.append({"role": "system", "content": f"Function call rationale: {rationale}\n\nTicket purchase confirmed: {ticket_purchase}"})
                    # Generate a response to the user
                    response_message = await generate_response(client, message_history, gen_kwargs)
                    response_message_content = response_message.content.strip()
                    print(f"response_message_chat: {response_message_content[:300]}...")
                else:
                    error_message = f"Function {function_name} not found"
                    message_history.append({"role": "system", "content": error_message})
                    # Generate a response to the user
                    response_message = await cl.Message(content=error_message).send()
            else:
                error_message = "Invalid function call format"
                message_history.append({"role": "system", "content": error_message})
                # Generate a response to the user
                response_message = await cl.Message(content=error_message).send()
        except json.JSONDecodeError as e:
            error_message = f"Invalid JSON format: {response_message.content}"
            print(f"error_message: {error_message} Traceback: {traceback.format_exc()}")
            message_history.append({"role": "system", "content": error_message})
            # Generate a response to the user
            response_message = await cl.Message(content=error_message).send()
            
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
