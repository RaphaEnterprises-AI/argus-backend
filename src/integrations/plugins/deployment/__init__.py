"""Deployment integration plugins.

This package contains integrations for deployment platforms:
- Vercel: Frontend and full-stack deployment
- Netlify: Modern web project deployment
- Heroku: PaaS for application deployment
- Railway: Infrastructure platform with instant provisioning
- Render: Unified cloud platform for web services
- Fly.io: Global application deployment on Fly Machines
- AWS: Amazon Web Services (ECS, Lambda, CloudWatch)
"""

from src.integrations.plugins.deployment.aws_plugin import AWSPlugin
from src.integrations.plugins.deployment.flyio_plugin import FlyioPlugin
from src.integrations.plugins.deployment.heroku_plugin import HerokuPlugin
from src.integrations.plugins.deployment.netlify_plugin import NetlifyPlugin
from src.integrations.plugins.deployment.railway_plugin import RailwayPlugin
from src.integrations.plugins.deployment.render_plugin import RenderPlugin
from src.integrations.plugins.deployment.vercel_plugin import VercelPlugin

__all__ = [
    "VercelPlugin",
    "NetlifyPlugin",
    "HerokuPlugin",
    "RailwayPlugin",
    "RenderPlugin",
    "FlyioPlugin",
    "AWSPlugin",
]
