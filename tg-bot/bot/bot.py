import os
import asyncio
import re
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import requests

from dotenv import load_dotenv
load_dotenv()

# from langgraph_agent import build_agent

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL")

# for test only all beckend from API
LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")


def llm_chat_tool(messages):
    response = requests.post(
        url=LLM_API_SERVER_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3:0.6b",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 500
        }
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    
    # Remove <think>...</think> blocks
    clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
    
    return clean_content.strip()


bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Отправь волшебное слово '???' чтобы получить доступ к боту!")

@dp.message(Command("start", "начали"))
async def access(message: types.Message):
    await message.answer("Добро пожаловать! Теперь вы можете отправлять мне текст, и я сгенерирую ответ.")

@dp.message()
async def handle_message(message: types.Message):
    try:
        # response = requests.post(API_URL, json={"prompt": message.text})
        # await message.answer(response.json()["generated_text"])
        response = llm_chat_tool([{"role": "user", "content": message.text}])
        await message.answer(response)
    except Exception as e:
        await message.answer("Ошибка: " + str(e))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())  # Properly run the async main function