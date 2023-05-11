from diotima.world import UniverseConfig
import pytest


@pytest.fixture
def universe_config():
    return UniverseConfig()


def test_init_universe_config(universe_config: UniverseConfig):
    pass


def test_validate_physics(universe_config: UniverseConfig):
    universe_config.validate_physics_config(universe_config.physics_config)
