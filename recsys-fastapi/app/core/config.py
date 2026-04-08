from pydantic import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    LOGGING_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()