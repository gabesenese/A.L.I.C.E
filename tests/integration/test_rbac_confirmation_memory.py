from ai.infrastructure.rbac import AccessRequest, RBACEngine, UserRole


def test_rbac_confirmation_grace_allows_followup_high_risk_action():
    engine = RBACEngine(default_role=UserRole.TRUSTED, confirmation_grace_seconds=180)

    first = engine.authorize(
        AccessRequest(
            intent="system:execute",
            user_input="confirm run it",
            entities={},
        )
    )
    assert first.allowed is True
    assert first.requires_confirmation is False

    second = engine.authorize(
        AccessRequest(
            intent="system:execute",
            user_input="run deployment",
            entities={},
        )
    )
    assert second.allowed is True
    assert second.requires_confirmation is False
