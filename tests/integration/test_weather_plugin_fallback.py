from ai.plugins.plugin_system import WeatherPlugin


def test_weather_plugin_strips_wake_word_from_location_candidate():
    plugin = WeatherPlugin()
    assert plugin._clean_location_candidate("seattle alice") == "seattle"
    assert (
        plugin._clean_location_candidate("Austin, Texas assistant") == "Austin, Texas"
    )


def test_weather_plugin_detect_location_fallback_uses_ip_data(monkeypatch):
    plugin = WeatherPlugin()

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "status": "success",
                "city": "Toronto",
                "country": "Canada",
            }

    class _Requests:
        @staticmethod
        def get(url, timeout=4):
            return _Resp()

    monkeypatch.setitem(__import__("sys").modules, "requests", _Requests)
    loc = plugin._detect_location_fallback()
    assert loc == "Toronto, Canada"
