import asyncio
from datetime import timezone
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import aiohttp
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")


# Validate required environment variables early
if not all([BOT_TOKEN, API_URL, API_TOKEN]):
    logger.error("Missing required environment variables BOT_TOKEN, API_URL or API_TOKEN")
    raise ValueError("Missing required environment variables BOT_TOKEN, API_URL or API_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

#  Healthchek  API

async def check_api_connection(retries: int = 20, delay: int = 5) -> bool:
    """Check if API is reachable and token is valid, retrying if needed."""
    health_url = f"{API_URL}/health" if not API_URL.endswith("/health") else API_URL

    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("API connection test successful")
                        return True
                    else:
                        logger.error(f"API connection test failed with status: {resp.status}")
        except aiohttp.ClientError as e:
            logger.error(f"[Attempt {attempt}/{retries}] Connection failed to {health_url}: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"[Attempt {attempt}/{retries}] Timeout while connecting to API at {health_url}")
        except Exception as e:
            logger.error(f"[Attempt {attempt}/{retries}] Unexpected API test error: {str(e)}")

        logger.info(f"Waiting {delay} seconds before retrying...")
        await asyncio.sleep(delay)

    logger.critical(f"API did not become available after {retries} attempts.")
    return False

@dp.message(Command("start"))
async def start_command(message: types.Message):
    """Handle /start command."""
    logger.info(f"Start command received from user {message.from_user.id}")
    await message.answer("Добро пожаловать! Отправьте мне ваш вопрос, и я постараюсь помочь.")

@dp.message()
async def handle_message(message: types.Message):
    """Handle user messages."""
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Determine thread ID
    if hasattr(message, "message_thread_id") and message.message_thread_id:
        # Forum thread (topic) exists
        thread_id = f"forum_{message.message_thread_id}"
    else:
        # Normal chat → consistent per user
        thread_id = f"chat_{chat_id}_{user_id}"

     # created_at in ISO8601 UTC
    created_at = message.date.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"created_at: {created_at} Received message from user {user_id}: {message.text}")

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": message.text,
        "user_id": str(user_id),
        "thread_id": str(thread_id),
        "created_at": str(created_at)
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{API_URL}/generate',
                json=payload,
                headers=headers,
                timeout=180
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                response_text = data.get("generated_text", "Не удалось получить ответ от API.")
                await message.answer(response_text)
                logger.info(f"Sent response to user {user_id}")
    except aiohttp.ClientError as e:
        error_msg = f"Ошибка соединения с API: {e}"
        await message.answer(error_msg)
        logger.error(f"API connection error for user {user_id}: {e}")
    except Exception as e:
        error_msg = "Произошла внутренняя ошибка. Пожалуйста, попробуйте позже."
        await message.answer(error_msg)
        logger.error(f"Unexpected error for user {user_id}: {e}", exc_info=True)

async def main():
    """Main entry point: check API and start polling."""
    logger.info("Initializing bot...")
    
    if not await check_api_connection():
        logger.error("API token validation failed or API unreachable. Shutting down bot.")
        return
    
    logger.info("API token valid. Starting bot polling.")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.critical(f"Bot crashed: {e}", exc_info=True)
    finally:
        logger.info("Bot stopped")
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
