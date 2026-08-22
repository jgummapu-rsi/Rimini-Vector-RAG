"""CrossEncoderReranker: the real ONNX cross-encoder runs for real (model is
cached after first load, same as MiniLMEmbedder in test_embedder.py)."""
from app.adapters.rerankers.cross_encoder import CrossEncoderReranker


def test_cross_encoder_scores_relevant_doc_higher():
    rr = CrossEncoderReranker()
    scores = rr.score(
        "What transaction code shows update terminations after COMMIT WORK?",
        [
            "SM13 shows update terminations that occurred after COMMIT WORK.",
            "A cat sat on a sofa in the afternoon sun.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_cross_encoder_empty_documents():
    assert CrossEncoderReranker().score("anything", []) == []
