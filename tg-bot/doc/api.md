## Документация по использованию API (на русском)

### Проверка работоспособности API

Для проверки статуса сервера используйте эндпоинт `/health`:

```python
import requests
api_url = 'http://localhost:8000'
health = requests.get(f'{api_url}/health')
print(health.status_code, health.text)
```

### Генерация текста через API

Для генерации текста используйте POST-запрос к `/generate`:

- **URL:** `http://localhost:8000/generate`
- **Метод:** POST
- **Заголовки:**
    - `Content-Type: application/json`
    - `Authorization: Bearer <API_TOKEN>`
- **Тело запроса (JSON):**
    - `prompt` — строка с запросом пользователя
    - `user_id` — идентификатор пользователя
    - `thread_id` — идентификатор сессии/диалога

Пример запроса:
```python
import requests
import os
API_TOKEN = os.getenv('API_TOKEN')
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}
payload = {
    "prompt": "Привет! Расскажи о погоде в Москве.",
    "user_id": "user_123",
    "thread_id": "thread_123"
}
response = requests.post('http://localhost:8000/generate', json=payload, headers=headers)
print(response.status_code)
print(response.json())
```

### Ответ API
В ответе ожидается JSON с ключом `generated_text`, содержащим сгенерированный текст.

### Пример функции для тестирования
Ниже реализована функция `test_llm`, которую можно использовать для отправки запросов к API и вывода результата.


```python
# def for test llm
def test_llm(prompt: str, user_id: str, thread_id: str) -> None:
    api_url = "http://localhost:8000"
    endpoint = "/generate"

       
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"  # add token
    }

    payload = {
        "prompt": prompt,
        "user_id": user_id,
        "thread_id": thread_id
    }

    response = requests.post(f'{api_url}{endpoint}', json=payload, headers=headers)
    print("Status code:", response.status_code)
    try:
        data = response.json()
        print("Response:", data.get("generated_text", data))
    except Exception:
        print("Response Text:", response.text)


user_id = "user_123"
thread_id = "thread_123"

test_llm("Какая погода сегодня в Москве", user_id, thread_id)
```

подробнее в ноутбуке [api-test.ipynb](../dev/api-test.ipynb)