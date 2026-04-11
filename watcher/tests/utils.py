"""
Test utilities — PostgreSQL with transaction rollback isolation.

Each test wraps its work in a transaction that is rolled back in tearDown,
including any DDL (Postgres supports transactional DDL). Nothing ever
persists to the database.

Default test URL: postgresql+psycopg2://watcher:...@mira.local/watcher_test
Override by setting TEST_DATABASE_URL in the environment.

Prerequisites (one-time, run as a Postgres superuser):
    sudo -u postgres createdb -O watcher watcher_test
"""

import os

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session

from watcher.model import WatcherBase
from watcher.remote import APIUser, Base as RemoteBase

_DEFAULT_TEST_URL = (
    'postgresql+psycopg2://watcher:iquuvoaLi4woh3o@mira.local/watcher_test'
)

def test_db_url() -> str:
    return os.environ.get('TEST_DATABASE_URL', _DEFAULT_TEST_URL)


def make_test_engine():
    """Return an engine pointing at the test database."""
    return create_engine(test_db_url(), echo=False)


class TransactionalTestCase:
    """
    Mixin for unittest.TestCase subclasses.

    setUpClass  — creates all tables once per class
    setUp       — opens a connection and begins a transaction + savepoint
    tearDown    — rolls back to the savepoint (all writes undone)
    tearDownClass — drops all tables

    Usage:
        class MyTest(TransactionalTestCase, unittest.TestCase):
            def test_something(self):
                self.session.add(...)
    """

    _engine = None

    @classmethod
    def setUpClass(cls):
        cls._engine = make_test_engine()
        WatcherBase.metadata.create_all(cls._engine)
        RemoteBase.metadata.create_all(cls._engine)

    @classmethod
    def tearDownClass(cls):
        WatcherBase.metadata.drop_all(cls._engine)
        RemoteBase.metadata.drop_all(cls._engine)
        cls._engine.dispose()

    def setUp(self):
        self._conn = self._engine.connect()
        self._outer_tx = self._conn.begin()
        self.session = Session(bind=self._conn)
        # Nested transaction = SAVEPOINT; tearDown rolls back to it
        self._conn.execute(text('SAVEPOINT test_sp'))

    def tearDown(self):
        self.session.close()
        self._conn.execute(text('ROLLBACK TO SAVEPOINT test_sp'))
        self._outer_tx.rollback()
        self._conn.close()


# ── helpers used by existing tests ───────────────────────────────────────────

def create_db_from_sql(engine=None) -> tuple:
    """
    Create schema from db/watcher.sql and add a test API user.
    Returns (session, apiuser).

    When called from a TransactionalTestCase.setUp, pass cls._engine
    so the session shares the same connection and transaction.
    """
    if engine is None:
        engine = make_test_engine()
        WatcherBase.metadata.create_all(engine)
        RemoteBase.metadata.create_all(engine)

    session = sessionmaker(bind=engine)()

    testuser = APIUser(username='testuser')
    session.add(testuser)
    session.commit()

    return session, testuser


def create_db_from_object_model(base, engine=None) -> Session:
    """
    Create schema from SQLAlchemy models.
    Returns a session. Caller is responsible for teardown.
    """
    if engine is None:
        engine = make_test_engine()

    base.metadata.create_all(engine)
    RemoteBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
