"""Tests for the backend analyzer module."""

import pytest
from pathlib import Path

from src.analyzers.backend import BackendAnalyzer
from src.analyzers.base import ComponentType, Severity


class TestBackendAnalyzer:
    """Test BackendAnalyzer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with backend files."""
        # Create FastAPI routes
        api_dir = tmp_path / "src" / "api"
        api_dir.mkdir(parents=True)

        (api_dir / "users.py").write_text('''
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter()

class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str

def get_current_user():
    pass

@router.get("/users", response_model=list[UserResponse])
async def list_users():
    """List all users."""
    return []

@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get a specific user."""
    return {"id": user_id}

@router.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, current_user = Depends(get_current_user)):
    """Create a new user."""
    return {"id": 1, "name": user.name}

@router.delete("/admin/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user - admin only."""
    return {"deleted": True}
''')

        # Create Express routes
        express_dir = tmp_path / "src" / "routes"
        express_dir.mkdir(parents=True)

        (express_dir / "products.ts").write_text('''
import express from "express";
import { authenticate } from "../middleware/auth";

const router = express.Router();

router.get("/products", async (req, res) => {
    res.json([]);
});

router.get("/products/:id", async (req, res) => {
    res.json({ id: req.params.id });
});

router.post("/products", authenticate, async (req, res) => {
    res.json({ created: true });
});

export default router;
''')

        # Create Flask routes
        flask_dir = tmp_path / "src" / "flask_app"
        flask_dir.mkdir(parents=True)

        (flask_dir / "views.py").write_text('''
from flask import Blueprint, jsonify, request

bp = Blueprint("main", __name__)

@bp.route("/items")
def list_items():
    return jsonify([])

@bp.route("/items/<int:item_id>")
def get_item(item_id):
    return jsonify({"id": item_id})

@bp.route("/items", methods=["POST"])
def create_item():
    return jsonify({"created": True})
''')

        return tmp_path

    @pytest.fixture
    def analyzer(self, temp_repo):
        """Create a BackendAnalyzer for the temp repo."""
        return BackendAnalyzer(str(temp_repo))

    def test_analyzer_type(self, analyzer):
        """Test analyzer type property."""
        assert analyzer.analyzer_type == "backend"

    def test_file_patterns(self, analyzer):
        """Test that file patterns include backend files."""
        patterns = analyzer.get_file_patterns()

        assert any("routes" in p for p in patterns)
        assert any("api" in p for p in patterns)
        assert any(".py" in p for p in patterns)
        assert any(".ts" in p for p in patterns)

    def test_analyze_finds_fastapi_routes(self, analyzer):
        """Test that analysis finds FastAPI routes."""
        result = analyzer.analyze()

        route_names = [c.name for c in result.components if c.component_type == ComponentType.ROUTE]

        # Should find GET /users, POST /users, etc.
        assert any("/users" in n for n in route_names if n)

    def test_analyze_finds_express_routes(self, analyzer):
        """Test that analysis finds Express routes."""
        result = analyzer.analyze()

        route_names = [c.name for c in result.components if c.component_type == ComponentType.ROUTE]

        # Should find /products routes
        assert any("/products" in n for n in route_names if n)

    @pytest.mark.requires_tree_sitter
    def test_analyze_finds_flask_routes(self, analyzer):
        """Test that analysis finds Flask routes."""
        result = analyzer.analyze()

        route_names = [c.name for c in result.components if c.component_type == ComponentType.ROUTE]

        # Should find /items routes
        assert any("/items" in n for n in route_names if n)

    def test_analyze_detects_unprotected_admin_route(self, analyzer):
        """Test that analysis detects unprotected sensitive routes."""
        result = analyzer.analyze()

        # Find the admin delete route
        admin_route = next(
            (c for c in result.components if c.name and "/admin" in c.name),
            None
        )

        if admin_route:
            has_auth_warning = any(
                "auth" in i.message.lower() or "sensitive" in i.message.lower()
                for i in admin_route.issues
            )
            # Should have a warning about missing auth


class TestBackendAnalyzerFrameworkDetection:
    """Test framework detection in BackendAnalyzer."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create analyzer with minimal repo."""
        (tmp_path / "dummy.py").write_text("")
        return BackendAnalyzer(str(tmp_path))

    @pytest.mark.requires_tree_sitter
    def test_detect_fastapi(self, analyzer):
        """Test detecting FastAPI framework."""
        code = 'from fastapi import APIRouter\nrouter = APIRouter()'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "routes.py")

        framework = analyzer._detect_framework(parsed)
        assert framework == "fastapi"

    @pytest.mark.requires_tree_sitter
    def test_detect_flask(self, analyzer):
        """Test detecting Flask framework."""
        code = 'from flask import Flask\napp = Flask(__name__)'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "app.py")

        framework = analyzer._detect_framework(parsed)
        assert framework == "flask"

    @pytest.mark.requires_tree_sitter
    def test_detect_express(self, analyzer):
        """Test detecting Express framework."""
        code = 'const express = require("express");\nconst router = express.Router();'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "routes.js")

        framework = analyzer._detect_framework(parsed)
        assert framework == "express"

    @pytest.mark.requires_tree_sitter
    def test_detect_django(self, analyzer):
        """Test detecting Django framework."""
        code = 'from django.shortcuts import render\nfrom rest_framework.views import APIView'
        from src.indexer import TreeSitterParser
        parser = TreeSitterParser()
        parsed = parser.parse_content(code, "views.py")

        framework = analyzer._detect_framework(parsed)
        assert framework == "django"
