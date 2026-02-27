from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Target app URLs
    conduit_url: str = "https://conduit-testbench-production.up.railway.app"
    plane_url: str = "https://plane-testbench-production.up.railway.app"
    json_api_url: str = "https://jsonapi-testbench-production.up.railway.app"
    chaos_app_url: str = "https://chaos-testbench-production.up.railway.app"

    # Argus API
    argus_url: str = "https://argus-brain-production.up.railway.app"
    argus_api_key: str = ""
    argus_org_id: str = ""
    argus_project_id: str = ""

    # Controller
    port: int = 8000

    model_config = {"env_prefix": "TESTBENCH_"}


settings = Settings()
