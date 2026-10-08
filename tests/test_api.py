

def test_cors_default_is_frontend_not_wildcard():
    from app.main import Settings
    origins = Settings(_env_file=None).get_allowed_origins()
    assert "*" not in origins
    assert "https://frontend-vert-eight-61.vercel.app" in origins


def test_chat_stream_emits_status_then_answer(monkeypatch):
    """/chat/stream must surface node progress as `status` events before the
    single `answer` event, and the answer payload must match /chat's schema."""
    import json

    from fastapi.testclient import TestClient

    import agent.graph as graph_module
    import app.main as main_module

    monkeypatch.setattr(main_module.settings, "database_url", "")

    async def fake_stream_agent(user_query, session_id):
        yield {"type": "node", "node": "planner", "update": {"plan": [{"tool": "sql_analytics"}]}}
        yield {"type": "node", "node": "reporter", "update": {}}
        yield {"type": "result", "data": {"final_report": "## Answer", "citations": [], "provider": "groq"}}

    monkeypatch.setattr(graph_module, "stream_agent", fake_stream_agent)

    with TestClient(main_module.app).stream("POST", "/chat/stream", json={"query": "hi"}) as res:
        body = "".join(res.iter_text())

    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1].removeprefix("data: ")))
        for block in body.strip().split("\n\n")
    ]
    names = [name for name, _ in events]
    assert names == ["status", "status", "answer", "done"]
    assert events[0][1]["message"] == "Planned: sql_analytics"
    assert events[2][1]["answer"] == "## Answer"
    assert events[2][1]["session_id"]
