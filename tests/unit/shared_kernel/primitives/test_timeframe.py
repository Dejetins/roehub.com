from datetime import datetime, timezone

from trading.shared_kernel.primitives import Timeframe, UtcTimestamp


def test_timeframe_normalizes_code_and_accepts_supported() -> None:
    tf = Timeframe(" 1M ")
    assert tf.code == "1m"
    assert str(tf) == "1m"

    assert Timeframe("5m").code == "5m"
    assert Timeframe("1h").code == "1h"


def test_timeframe_rejects_unsupported() -> None:
    try:
        Timeframe("2m")
        assert False, "Expected ValueError for unsupported timeframe"
    except ValueError:
        assert True


def test_timeframe_bucket_alignment_epoch_aligned_utc() -> None:
    # 2026-02-04 12:34:56.789Z
    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 34, 56, 789000, tzinfo=timezone.utc))

    tf_15m = Timeframe("15m")
    bucket_open = tf_15m.bucket_open(ts)
    bucket_close = tf_15m.bucket_close(ts)

    assert str(bucket_open) == "2026-02-04T12:30:00.000Z"
    assert str(bucket_close) == "2026-02-04T12:45:00.000Z"

    tf_1h = Timeframe("1h")
    assert str(tf_1h.bucket_open(ts)) == "2026-02-04T12:00:00.000Z"
    assert str(tf_1h.bucket_close(ts)) == "2026-02-04T13:00:00.000Z"
