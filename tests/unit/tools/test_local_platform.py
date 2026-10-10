import json
from xml.etree import ElementTree

import pytest
import yaml

from apps.control_agent.docker_backend import DockerComposeControlBackend
from tools.installation.local_platform import WORKERS, render
from trading.contexts.operations import ControlOperationError


def _required_text(element: ElementTree.Element, path: str) -> str:
    value = element.findtext(path)
    assert value is not None, path
    return value


def test_adoption_bundle_preserves_storage_and_binds_controller_policy(tmp_path):
    config = dict(root=str(tmp_path), runtime_image='local-runtime:test', runtime_user='501:20',
        control_image='sha256:'+'b'*64,
        artifact_root='/owned/artifacts', network='local-platform', postgres_volume='existing-pg',
        clickhouse_volume='existing-ch', socket_volume='control-socket')
    release = dict(schema='io.roehub.release/v1alpha1', version='0.1.0', images={
        key: {'reference': key+'@sha256:'+'a'*64}
        for key in ('runtime','postgresql','clickhouse')})
    source, release_source = json.dumps(config).encode(), json.dumps(release).encode()
    outputs = render(config, release, source=source, release_source=release_source)
    capacity = ElementTree.fromstring(outputs['config/clickhouse-capacity.xml'])
    assert int(_required_text(capacity, 'max_server_memory_usage')) == 1073741824
    assert int(_required_text(capacity, 'query_condition_cache_size')) == 1048576
    logs = ElementTree.fromstring(outputs['config/clickhouse-observability.xml'])
    for name in ('metric_log', 'asynchronous_metric_log', 'query_metric_log',
                 'trace_log', 'processors_profile_log', 'query_views_log', 'query_thread_log'):
        item = logs.find(name)
        assert item is not None
        assert item.attrib == {'remove': '1'}
    for name in ('query_log', 'part_log', 'text_log', 'error_log'):
        log = logs.find(name)
        assert log is not None
        assert int(_required_text(log, 'reserved_size_rows')) < int(
            _required_text(log, 'buffer_size_rows_flush_threshold')
        ) < int(_required_text(log, 'max_size_rows'))
        assert _required_text(log, 'ttl') == 'event_date + INTERVAL 1 DAY DELETE'
    for path in ('config/clickhouse-profiles.xml', 'config/clickhouse-query-limits.xml'):
        profile = ElementTree.fromstring(outputs[path])
        assert profile.findtext('profiles/default/use_query_condition_cache') == '0'
    pool = int(_required_text(capacity, 'background_pool_size'))
    concurrency = int(_required_text(capacity, 'background_merges_mutations_concurrency_ratio'))
    assert pool * concurrency <= 4
    assert int(_required_text(capacity, 'merge_tree/merge_max_block_size')) <= 1024
    assert int(_required_text(capacity, 'merge_tree/merge_max_block_size_bytes')) <= 1048576
    assert int(_required_text(
        capacity, 'merge_tree/vertical_merge_algorithm_min_rows_to_activate'
    )) == 1
    # ClickHouse rejects startup if any admission threshold exceeds its merge pool.
    for setting in (
        'number_of_free_entries_in_pool_to_execute_mutation',
        'number_of_free_entries_in_pool_to_lower_max_size_of_merge',
        'number_of_free_entries_in_pool_to_execute_optimize_entire_partition',
    ):
        assert int(_required_text(capacity, 'merge_tree/' + setting)) < pool * concurrency
    for name, value in outputs.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (tmp_path/'release.json').write_bytes(release_source)
    DockerComposeControlBackend(profile_root=tmp_path, project='local-test',
                                trusted_release_manifest=tmp_path/'release.json')
    controller = yaml.safe_load(outputs['control-compose.yaml'])['services']['control-agent']
    assert controller['user'] == '501:20'
    assert controller['image'] == config['control_image']
    assert '/var/run/docker.sock:/var/run/docker.sock' in controller['volumes']
    assert controller['healthcheck']
    compose = yaml.safe_load(outputs['compose.yaml'])
    health_query = compose['services']['clickhouse']['healthcheck']['test'][-1]
    assert 'canonical_candles_1m_stats' in health_query
    assert compose['volumes']['postgres-data'] == {'external': True, 'name': 'existing-pg'}
    assert compose['volumes']['clickhouse-data'] == {'external': True, 'name': 'existing-ch'}
    assert 'control-agent' not in compose['services']
    assert '--requests-only' in compose['services']['market-data-scheduler']['command']
    for name, service in compose['services'].items():
        assert not any('docker.sock' in mount for mount in service['volumes'])
        assert service['deploy']['resources']['limits']['memory']
        if name in WORKERS:
            assert service['stop_grace_period'] == '180s'
            assert '/owned/artifacts:/owned/artifacts' in service['volumes']
            assert not any('control-socket' in mount for mount in service['volumes'])
    # A post-generation override cannot silently broaden runtime authority.
    compose['services']['api']['volumes'].append('/var/run/docker.sock:/var/run/docker.sock')
    (tmp_path/'compose.yaml').write_text(yaml.safe_dump(compose))
    with pytest.raises(ControlOperationError):
        DockerComposeControlBackend(profile_root=tmp_path, project='local-test',
                                    trusted_release_manifest=tmp_path/'release.json')


@pytest.mark.parametrize('mode', ['provisioned', 'active'])
def test_workload_activation_and_rollback_keep_additive_server_identities(tmp_path, mode):
    config=dict(root=str(tmp_path),runtime_image='local-runtime:test',runtime_user='501:20',
                artifact_root='/owned/artifacts', network='test',postgres_volume='existing-pg',
                clickhouse_volume='existing-ch',socket_volume='socket',
                clickhouse_workload_budgets=mode)
    release=dict(schema='io.roehub.release/v1alpha1',version='0.1.0',images={
        key:{'reference':key+'@sha256:'+'a'*64} for key in ('runtime','postgresql','clickhouse')})
    outputs=render(config,release,source=json.dumps(config).encode(),release_source=json.dumps(release).encode())
    services=yaml.safe_load(outputs['compose.yaml'])['services']
    assert any('/users.xml:' in m for m in services['clickhouse']['volumes'])
    for name,workload in {'api':'interactive','market-data-ws':'stream',
                          'market-data-scheduler':'backfill',
                          'backtest-artifact-publisher':'artifacts'}.items():
        service=services[name]
        if mode=='active':
            assert service['environment']['ROEHUB_CH_WORKLOADS_REQUIRED']=='1'
            assert service['environment']['CH_PASSWORD']==''
            assert service['environment']['CLICKHOUSE_PASSWORD']==''
            assert any(f'/{workload}.password:' in m for m in service['volumes'])
            assert not any('/private:' in m or '/private/clickhouse-workloads:' in m
                           for m in service['volumes'])
        else:
            assert 'environment' not in service
