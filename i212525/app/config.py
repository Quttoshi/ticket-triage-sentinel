from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
ROOT = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    host: str = '127.0.0.1'
    port: int = 8000
    model_provider: str = 'openai'
    model_name: str = 'gpt-4.1-mini'
    allowed_models: str = 'gpt-4.1-mini,gpt-4.1-nano'
    openai_api_key: str = ''
    max_steps: int = Field(default=6, ge=1, le=6)
    max_tool_retries: int = Field(default=2, ge=0, le=2)
    max_repairs: int = Field(default=2, ge=0, le=2)
    max_output_tokens: int = Field(default=512, ge=1, le=4096)
    run_timeout_seconds: float = Field(default=40, gt=0, le=40)
    model_timeout_seconds: float = Field(default=15, gt=0, le=40)
    tool_timeout_seconds: float = Field(default=5, gt=0, le=20)
    max_external_chars: int = Field(default=12000, ge=500)
    max_concurrent_runs: int = Field(default=4, ge=1, le=32)
    max_model_calls_per_day: int = Field(default=3000, ge=1)
    chat_requests_per_minute: int = Field(default=20, ge=1)

    @property
    def api_key(self) -> str:
        return self.openai_api_key

    @property
    def model_list(self) -> list[str]:
        names = [m.strip() for m in self.allowed_models.split(',') if m.strip()]
        if self.model_name and self.model_name not in names:
            names.insert(0, self.model_name)
        return names
settings = Settings()
