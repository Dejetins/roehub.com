"""Execute explicitly imported immutable legacy demand without current catalog defaults."""

from uuid import NAMESPACE_URL, uuid5

from trading.contexts.backtest_artifacts.application.services.management import (
    Grid,
    Indicator,
    Profile,
    digest,
)


def legacy_profiles(snapshot):
    catalog = tuple(
        Indicator(
            indicator_id=name,
            sources=tuple(value["sources"]),
            params={k: Grid(mode="explicit", values=tuple(v)) for k, v in value["params"].items()},
            signal_params={
                k: Grid(mode="explicit", values=tuple(v)) for k, v in value["signal_params"].items()
            },
        )
        for name, value in sorted(snapshot["indicator_defaults"].items())
    )
    identity = digest([x.model_dump(mode="json") for x in catalog])
    groups = {}
    if snapshot["signals"] == "precompute":
        for item in snapshot["signal_artifacts"]:
            groups.setdefault(item["timeframe"], set()).add(item["indicator_id"])
    if not groups:
        groups["15m"] = set()
    result = []
    for index, (tf, names) in enumerate(sorted(groups.items())):
        result.append(
            Profile(
                profile_id=uuid5(NAMESPACE_URL, digest(snapshot) + tf),
                revision=1,
                catalog_revision=identity,
                mode="precompute",
                derivative_timeframes=(tf,),
                indicators=tuple(x for x in catalog if x.indicator_id in names),
                catalog_indicators=catalog,
                tp_levels_pct=tuple(snapshot["tp_levels_pct"])
                if index == 0 and snapshot["hit_times"] == "precompute"
                else (),
                sl_levels_pct=tuple(snapshot["sl_levels_pct"])
                if index == 0 and snapshot["hit_times"] == "precompute"
                else (),
            )
        )
    return tuple(result)
