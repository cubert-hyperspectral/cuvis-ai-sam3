"""T1: ServerConfig loads with sane defaults."""

from rest_api.config import ServerConfig


def test_config_defaults():
    config = ServerConfig()
    assert config.port == 8100
    assert config.host == "0.0.0.0"
    assert config.device == "cuda"
    assert config.session_timeout_seconds == 3600
    assert config.max_sessions == 10
    assert config.compile_model is False
    assert config.checkpoint_path is None
    assert config.bpe_path is None
