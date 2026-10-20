"""Person ReID models: OSNet-AIN (512) and FastReID MSMT17 (2048), same checks
for both. Model tests are skipped when torch or the weights aren't installed;
the same-person test also needs YOLO and the waiting-room clip."""

import importlib.util
import pathlib

import numpy as np
import pytest

from services.worker.app.domain.models import CropSample, ObjectType
from services.worker.app.reid import fastreid, osnet

ROOT = pathlib.Path(__file__).resolve().parents[3]
YOLO = ROOT / "models" / "yolo" / "yolo11m.pt"
CLIP = ROOT / "data" / "recordings" / "test-waiting" / "2018-03-13_15-50-00.mp4"

MODELS = {
    "osnet": (osnet.OsnetAinReID, osnet.OSNET_AIN_X1_0_MSMT17),
    "fastreid": (fastreid.FastReIDPersonReID, fastreid.FASTREID_MSMT17_SBS_R50_IBN),
}


def test_osnet_preprocess_matches_training():
    x = osnet.preprocess(np.full((300, 90, 3), 128, np.uint8))
    assert x.shape == (3, 256, 128) and x.dtype == np.float32
    card = osnet.OSNET_AIN_X1_0_MSMT17
    assert np.allclose(x[:, 0, 0], (128 / 255 - np.array(card.mean)) / np.array(card.std), atol=1e-6)


def test_fastreid_preprocess_keeps_0_255_and_is_384_by_128():
    x = fastreid.preprocess(np.full((300, 90, 3), 128, np.uint8))
    assert x.shape == (3, 384, 128) and x.dtype == np.float32
    assert np.allclose(x, 128)  # the network itself subtracts the mean and divides by the std


@pytest.mark.parametrize("prep", [osnet.preprocess, fastreid.preprocess])
def test_preprocess_keeps_red_as_red(prep):
    red = np.zeros((256, 128, 3), np.uint8)
    red[..., 0] = 255
    x = prep(red)
    assert x[0].mean() > x[2].mean()  # channel 0 is red, not blue


_loaded = {}


def load(name):
    cls, card = MODELS[name]
    if importlib.util.find_spec("torch") is None or not (ROOT / card.weights).is_file():
        pytest.skip(f"needs torch and {card.weights}")
    if name not in _loaded:
        _loaded[name] = cls(ROOT / card.weights, device="cpu")
    return _loaded[name]


@pytest.fixture(params=list(MODELS), scope="module")
def model(request):
    return load(request.param)


def crop(seed: int, object_type=ObjectType.PERSON) -> CropSample:
    image = np.random.default_rng(seed).integers(0, 256, (200, 80, 3), dtype=np.uint8)
    return CropSample("cam01-1", object_type, image, 0.8)


def test_barcode_has_the_cards_length_and_is_length_one(model):
    e = model.embed(crop(1))
    v = np.array(e.values)
    assert v.shape == (model.card.feature_dim,)
    assert abs(np.linalg.norm(v) - 1) < 1e-5
    assert e.vector_space == model.card.model_key  # = an HNSW index in 003_reid_embeddings.sql
    assert e.model_version == model.card.sha256


def test_osnet_barcodes_are_never_negative():
    assert load("osnet").vectors([crop(2).image]).min() >= 0


def test_a_vehicle_crop_gets_no_person_barcode(model):
    assert model.embed(crop(1, ObjectType.VEHICLE)) is None


def test_batch_gives_the_same_barcodes_as_one_by_one(model):
    crops = [crop(s) for s in range(4)]
    batch = np.array([e.values for e in model.embed_batch(crops)])
    single = np.array([model.embed(c).values for c in crops])
    assert np.allclose(batch, single, atol=1e-4)


@pytest.mark.parametrize("name", list(MODELS))
def test_wrong_weights_are_refused(tmp_path, name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("needs torch")
    cls, card = MODELS[name]
    fake = tmp_path / card.weights.name
    fake.write_bytes(b"not these weights")
    with pytest.raises(ValueError, match="SHA-256"):
        cls(fake, device="cpu")


@pytest.fixture(scope="module")
def two_frames_of_people():
    """Accepted person crops from the first two sampled frames (0.5 s apart)."""
    if importlib.util.find_spec("ultralytics") is None or not YOLO.is_file() or not CLIP.is_file():
        pytest.skip("needs ultralytics, YOLO weights and the waiting-room clip")
    from services.worker.app.detection.yolo import YoloDetector
    from services.worker.app.indexing.person_crops import PersonCropRules, judge_person_crop
    from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader

    detector = YoloDetector(YOLO, device="cpu", object_types={ObjectType.PERSON})
    people = []
    for frame in list(PyAVSegmentReader().frames(str(CLIP), 2))[:2]:
        kept = []
        for d in detector.detect(frame):
            verdict = judge_person_crop(frame.image, d.bbox, d.confidence, PersonCropRules())
            if verdict.reason is None:
                x1, y1, x2, y2 = verdict.box
                kept.append((d.bbox, frame.image[y1:y2, x1:x2]))
        people.append(kept)
    return people


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def test_same_person_half_a_second_later_is_the_most_similar(model, two_frames_of_people):
    """Real people: each person in frame 1 must be closer to themself in frame 2
    than to anybody else there -- the whole point of a ReID barcode."""
    first, second = two_frames_of_people
    assert len(first) >= 3 and len(second) >= 3
    similarity = model.vectors([im for _, im in first]) @ model.vectors([im for _, im in second]).T
    for i, (box, _) in enumerate(first):
        same = max(range(len(second)), key=lambda j: iou(box, second[j][0]))
        assert iou(box, second[same][0]) > 0.5, "people barely move in half a second"
        assert similarity[i].argmax() == same, f"person {i}: {similarity[i].round(2)}"
