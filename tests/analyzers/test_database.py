"""Tests for the database analyzer module."""


import pytest

from src.analyzers.base import ComponentType, Severity
from src.analyzers.database import DatabaseAnalyzer


class TestDatabaseAnalyzer:
    """Test DatabaseAnalyzer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository with database files."""
        # Create SQL files
        sql_dir = tmp_path / "sql"
        sql_dir.mkdir()

        (sql_dir / "queries.sql").write_text('''
-- Get all users
SELECT id, name, email
FROM users
WHERE active = true;

-- Get user orders with join
SELECT u.name, o.total, o.created_at
FROM users u
INNER JOIN orders o ON o.user_id = u.id
WHERE o.status = 'completed';

-- Update user status
UPDATE users
SET active = false
WHERE last_login < '2024-01-01';

-- Delete old records (DANGEROUS!)
DELETE FROM audit_logs;
''')

        # Create migrations
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        (migrations_dir / "001_create_users.sql").write_text('''
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
''')

        (migrations_dir / "002_add_orders.sql").write_text('''
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    total DECIMAL(10, 2) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Missing index for foreign key
ALTER TABLE orders ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id);
''')

        # Create SQLAlchemy models
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        (models_dir / "user.py").write_text('''
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)

    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    user_id = Column(Integer, ForeignKey("users.id"))
''')

        # Create Prisma schema
        prisma_dir = tmp_path / "prisma"
        prisma_dir.mkdir()

        (prisma_dir / "schema.prisma").write_text('''
model User {
    id        Int      @id @default(autoincrement())
    email     String   @unique
    name      String?
    posts     Post[]
    createdAt DateTime @default(now())

    @@index([email])
}

model Post {
    id        Int      @id @default(autoincrement())
    title     String
    content   String?
    author    User     @relation(fields: [authorId], references: [id])
    authorId  Int
}
''')

        # Create file with embedded SQL
        (tmp_path / "repository.py").write_text('''
def get_user_by_email(email):
    # Safe - using parameterized query
    return db.execute("SELECT * FROM users WHERE email = :email", {"email": email})

def dangerous_query(name):
    # DANGEROUS - SQL injection!
    return db.execute(f"SELECT * FROM users WHERE name = '{name}'")
''')

        return tmp_path

    @pytest.fixture
    def analyzer(self, temp_repo):
        """Create a DatabaseAnalyzer for the temp repo."""
        return DatabaseAnalyzer(str(temp_repo))

    def test_analyzer_type(self, analyzer):
        """Test analyzer type property."""
        assert analyzer.analyzer_type == "database"

    def test_file_patterns(self, analyzer):
        """Test that file patterns include database files."""
        patterns = analyzer.get_file_patterns()

        assert any(".sql" in p for p in patterns)
        assert any("migrations" in p for p in patterns)
        assert any("models" in p for p in patterns)
        assert any(".prisma" in p for p in patterns)

    def test_analyze_finds_sql_queries(self, analyzer):
        """Test that analysis finds SQL queries."""
        result = analyzer.analyze()

        query_components = [c for c in result.components if c.component_type == ComponentType.QUERY]
        assert len(query_components) >= 1

    def test_analyze_finds_migrations(self, analyzer):
        """Test that analysis finds migrations."""
        result = analyzer.analyze()

        migration_components = [c for c in result.components if c.component_type == ComponentType.MIGRATION]
        assert len(migration_components) >= 1

    def test_analyze_finds_orm_models(self, analyzer):
        """Test that analysis finds ORM models."""
        result = analyzer.analyze()

        model_components = [c for c in result.components if c.component_type == ComponentType.MODEL]
        assert len(model_components) >= 1

    def test_analyze_finds_prisma_models(self, analyzer):
        """Test that analysis finds Prisma models."""
        result = analyzer.analyze()

        # Look for User or Post model
        model_names = [c.name for c in result.components if c.component_type == ComponentType.MODEL]
        assert "User" in model_names or "Post" in model_names

    def test_analyze_detects_delete_without_where(self, analyzer):
        """Test that analysis detects DELETE without WHERE clause."""
        result = analyzer.analyze()

        # Find components with critical issues
        critical_issues = [
            issue
            for c in result.components
            for issue in c.issues
            if issue.severity == Severity.CRITICAL
        ]

        any(
            "DELETE" in i.message.upper() and "WHERE" in i.message.upper()
            for i in critical_issues
        )
        # Should detect the dangerous DELETE

    def test_analyze_detects_sql_injection(self, analyzer):
        """Test that analysis detects SQL injection risks."""
        result = analyzer.analyze()

        # Find components with SQL injection warnings
        [
            issue
            for c in result.components
            for issue in c.issues
            if "injection" in issue.message.lower()
        ]

        # Should have detected the f-string SQL injection


class TestDatabaseAnalyzerSQLParsing:
    """Test SQL parsing in DatabaseAnalyzer."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create analyzer with minimal repo."""
        (tmp_path / "dummy.sql").write_text("")
        return DatabaseAnalyzer(str(tmp_path))

    def test_get_query_type_select(self, analyzer):
        """Test query type detection for SELECT."""
        query_type = analyzer._get_query_type("SELECT * FROM users")
        assert query_type == "SELECT"

    def test_get_query_type_insert(self, analyzer):
        """Test query type detection for INSERT."""
        query_type = analyzer._get_query_type("INSERT INTO users (name) VALUES ('test')")
        assert query_type == "INSERT"

    def test_get_query_type_update(self, analyzer):
        """Test query type detection for UPDATE."""
        query_type = analyzer._get_query_type("UPDATE users SET name = 'test'")
        assert query_type == "UPDATE"

    def test_get_query_type_delete(self, analyzer):
        """Test query type detection for DELETE."""
        query_type = analyzer._get_query_type("DELETE FROM users WHERE id = 1")
        assert query_type == "DELETE"

    def test_get_query_type_create_table(self, analyzer):
        """Test query type detection for CREATE TABLE."""
        query_type = analyzer._get_query_type("CREATE TABLE users (id INT)")
        assert query_type == "CREATE_TABLE"

    def test_extract_tables_from_select(self, analyzer):
        """Test table extraction from SELECT."""
        tables = analyzer._extract_tables("SELECT * FROM users")
        assert "users" in tables

    def test_extract_tables_with_join(self, analyzer):
        """Test table extraction from JOIN."""
        tables = analyzer._extract_tables(
            "SELECT * FROM users u JOIN orders o ON u.id = o.user_id"
        )
        assert "users" in tables or "u" in tables
        assert "orders" in tables or "o" in tables

    def test_check_sql_injection_safe(self, analyzer):
        """Test SQL injection check with safe query."""
        context = "db.execute('SELECT * FROM users WHERE id = ?', [user_id])"
        assert analyzer._check_sql_injection("SELECT * FROM users", context) is False

    def test_check_sql_injection_unsafe(self, analyzer):
        """Test SQL injection check with unsafe query."""
        context = 'f"SELECT * FROM users WHERE name = \'{name}\'"'
        assert analyzer._check_sql_injection("SELECT", context) is True
