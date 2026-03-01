"""
Production-Grade Database Connection Pool Manager
Supports PostgreSQL, SQLite with automatic failover and health monitoring
"""

import logging
import threading
import time
from typing import Optional, Any, Dict, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import database drivers
try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import RealDictCursor

    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("psycopg2 not available")

try:
    import sqlite3

    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


class DatabaseType(Enum):
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"


@dataclass
class DatabaseConfig:
    """Database configuration"""

    db_type: DatabaseType
    host: str = "localhost"
    port: int = 5432
    database: str = "alice"
    user: str = "alice"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: int = 30
    idle_timeout: int = 300
    max_retries: int = 3


class ConnectionPool:
    """
    Database connection pool with:
    - Automatic connection recycling
    - Health monitoring
    - Retry logic
    - Thread-safe operation
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.lock = threading.Lock()
        self.stats = {
            "connections_created": 0,
            "connections_used": 0,
            "queries_executed": 0,
            "errors": 0,
        }

        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize connection pool based on database type"""
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL and POSTGRES_AVAILABLE:
                self.pool = pg_pool.ThreadedConnectionPool(
                    minconn=self.config.min_connections,
                    maxconn=self.config.max_connections,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    connect_timeout=self.config.connection_timeout,
                )
                logger.info(
                    f"[DB] PostgreSQL pool initialized ({self.config.min_connections}-{self.config.max_connections} connections)"
                )

            elif self.config.db_type == DatabaseType.SQLITE and SQLITE_AVAILABLE:
                # SQLite doesn't have pool library, we'll implement basic pooling
                self.pool = SQLitePool(
                    database=self.config.database,
                    max_connections=self.config.max_connections,
                )
                logger.info(f"[DB] SQLite pool initialized ({self.config.database})")
            else:
                raise Exception(f"Database type {self.config.db_type} not available")

        except Exception as e:
            logger.error(f"[DB] Failed to initialize pool: {e}")
            raise

    @contextmanager
    def get_connection(self, retries: int = None):
        """
        Get database connection from pool (context manager)

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        retries = retries or self.config.max_retries
        conn = None

        for attempt in range(retries):
            try:
                if self.config.db_type == DatabaseType.POSTGRESQL:
                    conn = self.pool.getconn()
                    self.stats["connections_used"] += 1
                elif self.config.db_type == DatabaseType.SQLITE:
                    conn = self.pool.get_connection()

                try:
                    yield conn
                finally:
                    # Cleanup: return connection to pool
                    if conn:
                        if self.config.db_type == DatabaseType.POSTGRESQL:
                            self.pool.putconn(conn)
                        elif self.config.db_type == DatabaseType.SQLITE:
                            self.pool.return_connection(conn)
                return  # Success, exit

            except Exception as e:
                self.stats["errors"] += 1
                logger.warning(f"[DB] Connection attempt {attempt + 1} failed: {e}")

                if conn and self.config.db_type == DatabaseType.POSTGRESQL:
                    try:
                        self.pool.putconn(conn, close=True)
                    except:
                        pass

                if attempt < retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    def execute_query(
        self, query: str, params: tuple = None, fetch: bool = True
    ) -> Any:
        """Execute query with automatic connection management"""
        with self.get_connection() as conn:
            if self.config.db_type == DatabaseType.POSTGRESQL:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()

            try:
                cursor.execute(query, params or ())
                self.stats["queries_executed"] += 1

                if fetch:
                    if self.config.db_type == DatabaseType.SQLITE:
                        # Convert SQLite rows to dicts
                        columns = [desc[0] for desc in cursor.description]
                        return [dict(zip(columns, row)) for row in cursor.fetchall()]
                    else:
                        return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount

            except Exception as e:
                conn.rollback()
                logger.error(f"[DB] Query failed: {e}\nQuery: {query}")
                raise
            finally:
                cursor.close()

    def execute_many(self, query: str, params_list: list) -> int:
        """Execute query with multiple parameter sets (bulk insert)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.executemany(query, params_list)
                conn.commit()
                self.stats["queries_executed"] += len(params_list)
                return cursor.rowcount

            except Exception as e:
                conn.rollback()
                logger.error(f"[DB] Bulk query failed: {e}")
                raise
            finally:
                cursor.close()

    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            result = self.execute_query("SELECT 1 as health", fetch=True)
            return len(result) > 0 and result[0].get("health") == 1
        except:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = self.stats.copy()

        if self.config.db_type == DatabaseType.POSTGRESQL and self.pool:
            # Get active connection count - not directly available in psycopg2
            stats["pool_size"] = (
                f"{self.config.min_connections}-{self.config.max_connections}"
            )

        stats["healthy"] = self.health_check()
        return stats

    def close(self):
        """Close all connections in pool"""
        try:
            if self.config.db_type == DatabaseType.POSTGRESQL and self.pool:
                self.pool.closeall()
            elif self.config.db_type == DatabaseType.SQLITE and self.pool:
                self.pool.close_all()

            logger.info("[DB] Connection pool closed")
        except Exception as e:
            logger.error(f"[DB] Error closing pool: {e}")


class SQLitePool:
    """Simple connection pool for SQLite"""

    def __init__(self, database: str, max_connections: int = 10):
        self.database = database
        self.max_connections = max_connections
        self.available = []
        self.in_use = set()
        self.lock = threading.Lock()

    def get_connection(self):
        """Get connection from pool"""
        with self.lock:
            if self.available:
                conn = self.available.pop()
            elif len(self.in_use) < self.max_connections:
                conn = sqlite3.connect(self.database, check_same_thread=False)
                conn.row_factory = sqlite3.Row
            else:
                raise Exception("Connection pool exhausted")

            self.in_use.add(conn)
            return conn

    def return_connection(self, conn):
        """Return connection to pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.available.append(conn)

    def close_all(self):
        """Close all connections"""
        with self.lock:
            for conn in list(self.available) + list(self.in_use):
                try:
                    conn.close()
                except:
                    pass
            self.available.clear()
            self.in_use.clear()


# Global connection pool
_connection_pool = None


def get_connection_pool() -> ConnectionPool:
    """Get global connection pool"""
    global _connection_pool
    if _connection_pool is None:
        # Default to SQLite for development
        config = DatabaseConfig(db_type=DatabaseType.SQLITE, database="data/alice.db")
        _connection_pool = ConnectionPool(config)
    return _connection_pool


def initialize_database(config: DatabaseConfig) -> ConnectionPool:
    """Initialize global database connection pool"""
    global _connection_pool
    _connection_pool = ConnectionPool(config)
    return _connection_pool


def create_tables():
    """Create database tables for A.L.I.C.E"""
    pool = get_connection_pool()

    # Notes table
    pool.execute_query(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
        fetch=False,
    )

    # Memories table
    pool.execute_query(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            context TEXT,
            importance REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP
        )
    """,
        fetch=False,
    )

    # Conversations table
    pool.execute_query(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_input TEXT NOT NULL,
            alice_response TEXT NOT NULL,
            intent TEXT,
            entities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
        fetch=False,
    )

    # Learning examples table
    pool.execute_query(
        """
        CREATE TABLE IF NOT EXISTS learning_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            intent TEXT,
            quality_score REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
        fetch=False,
    )

    logger.info("[DB] Database tables created successfully")
