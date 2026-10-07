

def test_cors_default_is_frontend_not_wildcard():
    from app.main import Settings
    origins = Settings(_env_file=None).get_allowed_origins()
    assert "*" not in origins
    assert "https://frontend-vert-eight-61.vercel.app" in origins
