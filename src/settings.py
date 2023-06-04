INSTALLED_APPS = [
    "db",
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": "semantra.sqlite",
    }
}
USE_TZ = True
