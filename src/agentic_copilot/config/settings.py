import os
from pathlib import Path

from pydantic_settings import SettingsConfigDict
from pydantic_settings_yaml import YamlBaseSettings


def base_path() -> Path:
    return Path(__file__).parent.parent.parent.parent


def yaml_config_location() -> str:
    env = os.getenv("ENVIRONMENT", "local").lower()

    config_filename = base_path().joinpath(f"config/{env}.yaml")

    return config_filename


class Settings(YamlBaseSettings):
    log_level: str
    azure_endpoint: str
    azure_api_key: str
    azure_temperature: float
    embedding_model: str
    embedding_deployment: str
    embedding_api_version: str
    max_function_calls: int

    model_config = SettingsConfigDict(yaml_file=yaml_config_location())
