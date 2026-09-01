import json

import pytest

from pyspatialml import operator_cli


def test_discover_operators_includes_arithmetic_and_model():
    operators = operator_cli.discover_operators()
    by_name = {item.enum_name: item for item in operators}

    assert "ARITHMETIC_COMPOSE" in by_name
    assert by_name["ARITHMETIC_COMPOSE"].creator == "arithmetic"
    assert by_name["ARITHMETIC_COMPOSE"].supported
    assert "expression" in by_name["ARITHMETIC_COMPOSE"].signature
    assert by_name["RUN_MODEL_INFERENCE"].creator == "run_model_inference"
    assert by_name["RUN_MODEL_INFERENCE"].supported


def test_discover_operators_marks_custom_handler_only_features():
    operators = {item.enum_name: item for item in operator_cli.discover_operators()}
    for name in (
        "CAMERA_SPACE_TO_WORLD", "LOAD_TEXTURE", "SWITCH_GLTF_RENDER_STATUS",
        "UPDATE_GLTF", "RENDER_TEXT",
    ):
        assert operators[name].supported
        assert not operators[name].native_default_loader_supported
        assert operators[name].requires_custom_handler


def test_find_operator_accepts_enum_type_and_creator_names():
    assert operator_cli.find_operator("ARITHMETIC_COMPOSE").creator == "arithmetic"
    assert operator_cli.find_operator("XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO").creator == "arithmetic"
    assert operator_cli.find_operator("arithmetic").enum_name == "ARITHMETIC_COMPOSE"
    assert operator_cli.find_operator("missing") is None


def test_list_operators_human_output(capsys):
    assert operator_cli.list_operators() == 0

    captured = capsys.readouterr()
    assert "Operators:" in captured.out
    assert "ARITHMETIC_COMPOSE" in captured.out
    assert "creator=arithmetic" in captured.out


def test_list_operators_json_output(capsys):
    assert operator_cli.list_operators(as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    arithmetic = next(item for item in payload if item["enum_name"] == "ARITHMETIC_COMPOSE")
    assert arithmetic["creator"] == "arithmetic"
    assert arithmetic["supported"] is True


def test_describe_operator_outputs_details(capsys):
    assert operator_cli.describe_operator("assignment") == 0

    captured = capsys.readouterr()
    assert "Operator: ASSIGNMENT" in captured.out
    assert "Creator: assignment" in captured.out
    assert "Signature: assignment" in captured.out


def test_describe_operator_reports_package_portability(capsys):
    assert operator_cli.describe_operator("render_text") == 0
    captured = capsys.readouterr()
    assert "Native default loader supported: no" in captured.out
    assert "Requires downstream custom handler: yes" in captured.out


def test_describe_operator_json_output(capsys):
    assert operator_cli.describe_operator("run_model_inference", as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["enum_name"] == "RUN_MODEL_INFERENCE"
    assert payload["creator"] == "run_model_inference"


def test_describe_operator_rejects_unknown():
    with pytest.raises(operator_cli.OperatorCliError, match="Unknown operator"):
        operator_cli.describe_operator("does_not_exist")


def test_enum_names_fallback_filters_non_uppercase_attrs(monkeypatch):
    class FakeOperatorType:
        FOO = 2
        BAR = 1
        value = 3
        name = 4

    monkeypatch.setattr(operator_cli, "EOperatorType", FakeOperatorType)

    assert operator_cli._enum_names() == ["BAR", "FOO"]
