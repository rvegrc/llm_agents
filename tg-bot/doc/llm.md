### Загрузка новых моделей через Ollama API

Для загрузки (pull) новых моделей на сервер Ollama используйте функцию `pull_ollama_model`. Эта функция отправляет POST-запрос к Ollama API и позволяет загрузить нужную модель по её имени.

**Пример использования:**
```python
pull_ollama_model("qwen3:4b")
```

- `model_name` — строка с названием модели, которую нужно загрузить (например, "qwen3:4b").
- После выполнения функция выведет статус и текст ответа от Ollama.

> ⚠️ Перед использованием убедитесь, что Ollama сервер запущен и доступен по адресу, указанному в переменной `LLM_API_SERVER_URL`.

```python
def pull_ollama_model(model_name: str) -> None:
    """
    Pulls a model to the Ollama server.
    
    Args:
        model_name (str): The name of the model to pull.
        
    Returns:
        dict: The response from the Ollama server if model is successfully pulled.
    """
    url = f"{LLM_API_SERVER_URL}/api/pull"
    payload = {
    "name": model_name
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    print("Status code:", response.status_code)
    print("Response:", response.text)

# use def
pull_ollama_model("qwen3:4b")
```

После загрузке новое модели протестировать ее работу возможно указав в переменной окружения 


подробнее в ноутбуке [api-test.ipynb](../dev/api-test.ipynb)