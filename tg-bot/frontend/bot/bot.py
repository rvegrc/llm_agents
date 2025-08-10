import asyncio
import os
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("API_URL")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def access(message: types.Message):
    await message.answer("Добро пожаловать! Теперь вы можете отправлять мне текст, и я сгенерирую ответ.")

@dp.message()
async def handle_message(message: types.Message):
    try:
        payload = {
            "prompt": message.text,
            "user_id": str(message.from_user.id),
            "thread_id": str(message.message_thread_id or message.message_id)
        }

        # Send request to FastAPI
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        generated_text = result.get("generated_text", "Нет ответа от API.")

        await message.answer(generated_text)

    except Exception as e:
        await message.answer("Ошибка: " + str(e))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
