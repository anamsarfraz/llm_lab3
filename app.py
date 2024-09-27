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
theaters. If a user asks for recent information, output a function call and \
the system add to the context. If you need to call a function, only output the \
function call. Call functions using Python syntax in plain text, no code blocks.

You have access to the following functions:

get_now_playing_movies()
get_showtimes(title, location)
buy_ticket(theater, movie, showtime)
confirm_ticket_purchase(theater, movie, showtime)
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

    context_response = response.choices[0].message.content
    print("Response text for review call: ", context_response)
    try:
        context_json = json.loads(context_response)
        if context_json.get("fetch_reviews", False):
            movie_id = context_json.get("id")
            reviews = get_reviews(movie_id)
            reviews = f"Reviews for {context_json.get('movie')} (ID: {movie_id}):\n\n{reviews}"
            context_message = {"role": "system", "content": f"CONTEXT: {reviews}"}
            message_history.append(context_message)
    except json.JSONDecodeError:
        print(f"Error parsing review call: {context_response}")

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    # Check for review call first
    await check_for_review_call(client, message_history, gen_kwargs)

    response_message = await generate_response(client, message_history, gen_kwargs)

    response_message_content = response_message.content
    message_history.append({"role": "assistant", "content": response_message_content})
    cl.user_session.set("message_history", message_history)

    while True:
        if "get_now_playing_movies()" in response_message_content:
            now_playing_movies = get_now_playing_movies()
            message_history.append({"role": "system", "content": now_playing_movies})
        elif "get_showtimes(" in response_message_content:
            try:
                start_index = response_message_content.find("get_showtimes(") + len("get_showtimes(")
                end_index = response_message_content.find(")", start_index)
                args = response_message_content[start_index:end_index].split(",")
                print(f"Received args for get_showtimes: {args}")
                title = args[0].strip().strip("\"")
                location = args[1].strip().strip("\"")
                print(f"Extracted title: {title}, location: {location}")
                showtimes = get_showtimes(title, location)
            except Exception as e:
                error = f"Error processing get_showtimes: {str(e)}"
                print(error)
                showtimes = error

            message_history.append({"role": "system", "content": showtimes})

        elif "buy_ticket(" in response_message_content:
            try:
                start_index = response_message_content.find("buy_ticket(") + len("buy_ticket(")
                end_index = response_message_content.find(")", start_index)
                args = response_message_content[start_index:end_index].split(",")
                print("Received args for buy_ticket: ", args)
                theater = args[0].strip().strip("\"")
                movie = args[1].strip().strip("\"")
                showtime = args[2].strip().strip("\"")
                print(f"Extracted theater: {theater}, movie: {movie}, showtime: {showtime}")

                purchase_ticket = buy_ticket(theater, movie, showtime)
            except Exception as e:
                error = f"Error processing buy_ticket: {str(e)}"
                print(error)
                purchase_ticket = error

            message_history.append({"role": "system", "content": purchase_ticket})
        elif "confirm_ticket_purchase(" in response_message_content:
            try:
                start_index = response_message_content.find("confirm_ticket_purchase(") + len("confirm_ticket_purchase(")
                end_index = response_message_content.find(")", start_index)
                args = response_message_content[start_index:end_index].split(",")
                print("Received args for confirm_ticket_purchase: ", args)
                theater = args[0].strip().strip("\"")
                movie = args[1].strip().strip("\"")
                showtime = args[2].strip().strip("\"")
                print(f"Extracted theater: {theater}, movie: {movie}, showtime: {showtime}")
                confirm_purchase = confirm_ticket_purchase(theater, movie, showtime)
            except Exception as e:
                error = f"Error processing confirm_ticket_purchase: {str(e)}"
                print(error)
                confirm_purchase = error

            message_history.append({"role": "system", "content": confirm_purchase})
        else:
            break

        # Generate a response to the user with the added system messages
        response_message = await generate_response(client, message_history, gen_kwargs)
        response_message_content = response_message.content

        message_history.append({"role": "assistant", "content": response_message.content})
        cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
