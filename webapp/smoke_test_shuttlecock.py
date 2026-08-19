"""
Smoke test for the Roboflow "Shuttlecock Detection" workflow integration.

Run manually (needs ROBOFLOW_API_KEY set):

    export ROBOFLOW_API_KEY='rf_xxxxxxxxxxxx'
    python webapp/smoke_test_shuttlecock.py path/to/badminton_photo.jpg

It calls the real workflow once, prints the raw output keys it actually
got back, and asserts the response has the shape run_shuttlecock_workflow
promises (a dict). It does NOT assert specific output key names, since
those were never verified against a live response while building this
integration — use the printed keys to sanity-check / refine
split_workflow_outputs's assumptions in shuttlecock_client.py if needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shuttlecock_client import (  # noqa: E402
    ShuttlecockConfigError,
    ShuttlecockDetectionError,
    run_shuttlecock_workflow,
    split_workflow_outputs,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python webapp/smoke_test_shuttlecock.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    try:
        raw = run_shuttlecock_workflow(image_path)
    except ShuttlecockConfigError as e:
        print(f"CONFIG ERROR: {e}")
        sys.exit(1)
    except ShuttlecockDetectionError as e:
        print(f"REQUEST FAILED: {e}")
        sys.exit(1)

    assert isinstance(raw, dict), f"Expected a dict result, got {type(raw)}"
    assert raw, "Workflow returned an empty dict"

    parsed = split_workflow_outputs(raw)

    print("Raw output keys:", list(raw.keys()))
    print("Classified as images:", list(parsed["images"].keys()))
    print("Classified as detections:", list(parsed["detections"].keys()))
    print("Classified as other:", {k: type(v).__name__ for k, v in parsed["other"].items()})
    print("\nSMOKE TEST PASSED — response received and has the expected top-level shape.")
    print("Now check the 'Classified as ...' lines above against what you expect from your")
    print("workflow definition, to confirm split_workflow_outputs guessed right.")


if __name__ == "__main__":
    main()
