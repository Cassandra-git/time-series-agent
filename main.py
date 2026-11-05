from ollama import chat

response = chat (
    model = "qwen3:8b",
    messages=[
        {
            "role":"user",
            "content": "Donne moi la définition d'une analyse de données"
        }
    ]


)

print(response.message.content)