import json

from cleantables.episodes import prepare, segment

RAW = {
    "videos": [
        {
            "video": "V1",
            "fps_used": 5,
            "tasks": [
                {
                    "id": "V1_t01",
                    "instruction": "Put the tools into the toolbox.",
                    "start_s": 0.4,
                    "end_s": 9.5,
                    "objects": ["wrench"],
                    "destination": "toolbox",
                    "complete": True,
                    "note": "",
                },
            ],
        },
        {
            "video": "V2",
            "fps_used": 5,
            "tasks": [
                {
                    "id": "V2_t01",
                    "instruction": "Stack the bowls.",
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "objects": ["bowl"],
                    "destination": "table",
                    "complete": True,
                    "note": "",
                },
            ],
        },
    ]
}


def config():
    return prepare.PrepareConfig(
        margin_s=1.0, videos=["V1"], task_keywords={"fruit": "fruit", "tools": "tool"}, excluded_videos={"V2": "bowls"}
    )


def test_margin_is_clipped_to_the_video():
    out = prepare.prepare(RAW, config(), video_end_s={"V1": 10.0})
    (e,) = out["tasks"]
    assert (e["start_s"], e["end_s"], e["core_start_s"], e["core_end_s"]) == (0.0, 10.0, 0.4, 9.5)
    assert e["task"] == "tools" and e["video"] == "V1"
    assert list(e) == [
        "id",
        "video",
        "task",
        "instruction",
        "start_s",
        "end_s",
        "core_start_s",
        "core_end_s",
        "objects",
        "destination",
        "complete",
        "note",
    ]


def test_margin_rounds_to_tenths():
    raw = json.loads(json.dumps(RAW))
    raw["videos"][0]["tasks"][0].update(start_s=14.4, end_s=20.8)
    (e,) = prepare.prepare(raw, config(), video_end_s={"V1": 100.0})["tasks"]
    assert (e["start_s"], e["end_s"]) == (13.4, 21.8)


def test_parse_answer_sorts_and_numbers_tasks():
    answer = json.dumps(
        {
            "tasks": [
                {
                    "start_s": 5.04,
                    "end_s": 7,
                    "instruction": "b",
                    "objects": ["x"],
                    "destination": "d",
                    "complete": True,
                    "note": "",
                },
                {
                    "start_s": 1,
                    "end_s": 3,
                    "instruction": "a",
                    "objects": [],
                    "destination": "d",
                    "complete": False,
                    "note": "",
                },
            ]
        }
    )
    tasks = segment.parse_answer("V9", answer)
    assert [t["id"] for t in tasks] == ["V9_t01", "V9_t02"]
    assert tasks[0]["instruction"] == "a" and tasks[1]["start_s"] == 5.0
    assert list(tasks[0]) == ["id", *segment.TASK_FIELDS]


def test_segment_skips_done_videos_and_writes_json(tmp_path, monkeypatch):
    calls = []

    def fake_ask(client, video_file, cfg):
        calls.append(video_file)
        return json.dumps({"tasks": []})

    monkeypatch.setattr(segment, "ask_gemini", fake_ask)
    monkeypatch.setattr(segment.paths, "video_path", lambda v: tmp_path / f"{v}.MOV")
    out = tmp_path / "episodes_raw.json"
    out.write_text(json.dumps({"videos": [{"video": "A", "fps_used": 5, "tasks": []}]}))
    data = segment.segment_videos(["A", "B"], segment.SegmentConfig(), out, client=object())
    assert [v["video"] for v in data["videos"]] == ["A", "B"]
    assert len(calls) == 1 and json.loads(out.read_text()) == data
