"""Unit tests for models/wellbeing_scorer.py::combine_scores() -- the one
deterministic step in the wellbeing pipeline (everything else calls an LLM).
"""
import pytest
from models.wellbeing_scorer import combine_scores


def test_core_only_averages_sound_and_emotion_50_50():
    # No transcript/scene supplied -> pure 50/50 average, no modifier.
    assert combine_scores(sound_score=0.8, emotion_score=0.4) == 0.60


def test_core_only_symmetric():
    assert combine_scores(sound_score=1.0, emotion_score=0.0) == 0.50


def test_neutral_scene_score_applies_no_modifier():
    # scene_score of exactly 0.5 is neutral by design -> no shift from core.
    core = combine_scores(sound_score=0.6, emotion_score=0.6)
    with_neutral_scene = combine_scores(sound_score=0.6, emotion_score=0.6, scene_score=0.5)
    assert with_neutral_scene == core


def test_maximally_positive_scene_score_adds_full_modifier_at_full_confidence():
    # scene_score=1.0, confidence=1.0 -> full +0.10 nudge.
    result = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=1.0, scene_confidence=1.0)
    assert result == 0.60


def test_maximally_negative_scene_score_subtracts_full_modifier_at_full_confidence():
    # scene_score=0.0, confidence=1.0 -> full -0.10 nudge.
    result = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=0.0, scene_confidence=1.0)
    assert result == 0.40


def test_scene_modifier_is_scaled_by_confidence():
    # Same extreme scene_score, but only 50% confidence -> half the nudge.
    result = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=1.0, scene_confidence=0.5)
    assert result == 0.55


def test_zero_confidence_fully_suppresses_scene_modifier():
    # A scene reading the classifier itself had zero confidence in should
    # not move the score at all, however extreme scene_score is.
    result = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=1.0, scene_confidence=0.0)
    assert result == 0.50


def test_missing_scene_confidence_defaults_to_full_weight():
    # scene_confidence omitted entirely (not just 1.0) must behave the
    # same as scene_confidence=1.0, per the module's own docstring.
    with_default = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=1.0)
    with_explicit_full = combine_scores(sound_score=0.5, emotion_score=0.5, scene_score=1.0, scene_confidence=1.0)
    assert with_default == with_explicit_full == 0.60


def test_transcript_and_scene_modifiers_stack():
    # Both an extreme positive transcript_score and scene_score together
    # should stack to a full +0.20 nudge on top of the 0.50 core.
    result = combine_scores(
        sound_score=0.5, emotion_score=0.5,
        transcript_score=1.0, scene_score=1.0, scene_confidence=1.0,
    )
    assert result == 0.70


def test_result_is_clamped_to_1_when_core_and_modifiers_would_exceed_it():
    result = combine_scores(
        sound_score=1.0, emotion_score=1.0,
        transcript_score=1.0, scene_score=1.0, scene_confidence=1.0,
    )
    assert result == 1.0


def test_result_is_clamped_to_0_when_core_and_modifiers_would_go_negative():
    result = combine_scores(
        sound_score=0.0, emotion_score=0.0,
        transcript_score=0.0, scene_score=0.0, scene_confidence=1.0,
    )
    assert result == 0.0


def test_result_is_rounded_to_two_decimal_places():
    result = combine_scores(sound_score=0.821, emotion_score=0.821)
    assert result == 0.82
