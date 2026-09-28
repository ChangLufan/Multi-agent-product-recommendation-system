from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # env中不生效时启用默认配置
    # LLM
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Redis（本地 Redis 3.2.100：带密码、需 RESP2 协议——老版本不认识 HELLO 命令）
    redis_url: str = "redis://:123456@127.0.0.1:6379/0"
    redis_protocol: int = 2  # redis-py 6.x 默认 RESP3 会发 HELLO，Redis<6.0 必须显式降为 2
    feature_ttl_seconds: int = 86400

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "product_embeddings"

    # Database
    database_url: str = "sqlite:///./ecommerce.db"

    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds)
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_marketing_copy: float = 10.0
    agent_timeout_inventory: float = 5.0

    model_config = {"env_file": ".env", "env_prefix": "ECOM_"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
