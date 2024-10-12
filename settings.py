INSTALLED_APPS = [
    # ...
    'sentiment',
    'rest_framework',  
    'corsheaders',  
]

MIDDLEWARE = [
    # ...
    'corsheaders.middleware.CorsMiddleware',
    # ...
]

# Дополнительные настройки
CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000"
    "http://localhost:8000",
    # другие домены будут тут
]
