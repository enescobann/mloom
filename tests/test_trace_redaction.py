import json
import sys

import pytest

import mloom
from mloom.sdk.trace import (
    capture_exception_context,
    is_internal_frame,
    redact_source_line,
    redact_source_text,
)

R = '"[REDACTED]"'


@pytest.mark.parametrize("line, expected", [
    # plain and annotated assignments
    ('vector_db_api_key = "vdb-live-7f3a9c2e"', f"vector_db_api_key = {R}"),
    ("    password: str = 'hunter2'  # comment", f"    password: str = {R}  # comment"),
    ('api_key: Optional[str] = "sk-4"', f"api_key: Optional[str] = {R}"),
    ('self.client_secret = "abc"', f"self.client_secret = {R}"),
    ('TOKEN="abc"', f"TOKEN={R}"),
    # keyword arguments
    ('client = Client(api_key="sk-1", base_url="http://x")',
     f'client = Client(api_key={R}, base_url="http://x")'),
    # dict keys
    ('cfg = {"api_key": "sk-2", "region": "eu"}', f'cfg = {{"api_key": {R}, "region": "eu"}}'),
    ("headers = {'Authorization': 'Bearer abc'}", f"headers = {{'Authorization': {R}}}"),
    # subscript assignment
    ("os.environ['OPENAI_API_KEY'] = 'sk-3'", f"os.environ['OPENAI_API_KEY'] = {R}"),
    # string prefixes, f-strings, escapes, triple quotes
    ('token = f"Bearer {raw}"', f"token = {R}"),
    ('pwd = r"raw\\path"', f"pwd = {R}"),
    ("secret = 'it\\'s'", f"secret = {R}"),
    ('self.secret = """abc"""', f"self.secret = {R}"),
    # several secrets on one line
    ('connect(user="bob", password="pw", token="t")', f'connect(user="bob", password={R}, token={R})'),
])
def test_secret_literals_are_redacted(line, expected):
    assert redact_source_line(line) == expected


@pytest.mark.parametrize("line", [
    'name = "bob"',
    'url = "https://example.com"',
    'api_key = os.environ["API_KEY"]',          # not a literal
    'if token == "abc":',                        # comparison, not assignment
    'if password != "":',
    'print("api_key = sk-123")',                 # inside a plain string
    'log.info({"user": "bob"})',
    "a = b = c",
    "",
])
def test_other_lines_are_untouched(line):
    assert redact_source_line(line) == line


def test_redact_source_text_handles_multiple_lines():
    text = 'Traceback:\n    api_key = "sk-1"\n    user = "bob"\n'
    assert redact_source_text(text) == f'Traceback:\n    api_key = {R}\n    user = "bob"\n'


def test_captured_trace_never_stores_literal_secrets():
    def connect(host):
        db_password = "pa55-hunter2"
        config = {"api_key": "sk-from-dict"}
        raise ConnectionError(f"cannot reach {host}")

    try:
        connect("db.local")
    except ConnectionError as e:
        trace = capture_exception_context(e, sys.exc_info()[2])

    dumped = json.dumps(trace)
    assert "pa55-hunter2" not in dumped
    assert "sk-from-dict" not in dumped
    assert any('db_password = "[REDACTED]"' in line for line in trace["source_context"]["lines"])
    assert '"api_key": "[REDACTED]"' in trace["traceback_formatted"] or \
        any('"api_key": "[REDACTED]"' in line for line in trace["source_context"]["lines"])
    # Non-secret code is still readable.
    assert 'raise ConnectionError(f"cannot reach {host}")' in trace["frames"][-1]["code_line"]


def test_is_internal_frame():
    import mloom.sdk.decorators as decorators

    assert is_internal_frame(decorators.__file__)
    assert is_internal_frame("/srv/venv/lib/python3.12/site-packages/mloom/sdk/patcher.py")
    assert not is_internal_frame(__file__)
    assert not is_internal_frame("/srv/app/mloom_bot/main.py")
    assert not is_internal_frame("<string>")
    assert not is_internal_frame("")
    assert mloom  # package import keeps the path check meaningful


def test_secrets_inside_local_values_are_redacted():
    class Client:
        def __init__(self, api_key):
            self.api_key = api_key

        def __repr__(self):
            return f"Client(api_key={self.api_key!r}, region='eu')"

    def call():
        settings = {"user": "bob", "password": "pw-in-dict"}
        client = Client("sk-in-object")
        raise RuntimeError("boom")

    try:
        call()
    except RuntimeError as e:
        trace = capture_exception_context(e, sys.exc_info()[2])

    local_vars = trace["frames"][-1]["locals"]
    assert local_vars["settings"] == "{'user': 'bob', 'password': \"[REDACTED]\"}"
    assert local_vars["client"] == "Client(api_key=\"[REDACTED]\", region='eu')"
