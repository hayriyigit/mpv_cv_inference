from __future__ import annotations

import pytest

pytest.importorskip("zmq")

from src.server.inference_server import BrightnessHotspotModel, InferenceServer
from src.shared.models import ControlCommandType, ModelControlCommand


def make_server() -> InferenceServer:
    return InferenceServer(
        video_uri="sample.mp4",
        pub_endpoint="inproc://pub-test",
        ctrl_endpoint="inproc://ctrl-test",
    )


def test_update_parameters_command() -> None:
    server = make_server()
    command = ModelControlCommand(
        command=ControlCommandType.UPDATE_PARAMS,
        model_name="brightness_hotspot",
        parameters={"sensitivity": 0.9},
    )
    response = server._handle_control(command)
    assert response["status"] == "ok"
    model = server.models["brightness_hotspot"]
    assert isinstance(model.runner, BrightnessHotspotModel)
    assert model.runner.sensitivity == 0.9


def test_request_status_all_models() -> None:
    server = make_server()
    command = ModelControlCommand(
        command=ControlCommandType.REQUEST_STATUS,
        model_name="*",
    )
    response = server._handle_control(command)
    assert response["status"] == "ok"
    assert "models" in response
    assert response["models"][0]["model_name"] == "brightness_hotspot"
