"""Disposable two-stage tests; never claim or change the real implementation ledger."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

PACK = Path(__file__).resolve().parent


def read_record(path):
    return json.loads(re.search(r'<!-- prompt-pack-ledger:v1 -->\s*```json\n(.*?)\n```',
                               path.read_text(), re.S)[1])


class LedgerUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='roehub-ledger-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = self.root / '.codex/agents/generated/fixture'
        self.pack.mkdir(parents=True)
        self.ledger = self.pack / 'iteration-ledger.md'
        shutil.copyfile(PACK / 'ledger_update.py', self.pack / 'ledger_update.py')
        (self.pack / 'implementation-plan.md').write_text('Test-only plan; no implementation proof.\n')
        record = read_record(PACK / 'iteration-ledger.md')
        record['stages'] = record['stages'][:2]
        record['ledger_status'] = 'draft'
        record['current_stage'] = None
        record.pop('execution_history', None)
        for index, row in enumerate(record['stages']):
            contract = row['contract']
            contract['expected_touches'] = []
            contract['entry_inputs'] = [{'path': 'implementation-plan.md', 'producer_stage': None}]
            if index:
                contract['entry_inputs'].append({'path': 'reports/S01.md', 'producer_stage': 'S01'})
            row.update(status='pending', execution_allowed=index == 0, current_authority=True,
                       executor_claim=None, claimed_at=None, transition_receipt=None, decision_packet=None)
            prompt = {'schema_version': 'stage-prompt/v1',
                      'prompt_pack_execution': record['prompt_pack_execution'], 'stage_contract': contract}
            (self.pack / contract['prompt_path']).write_text('---\n'+json.dumps(prompt)+'\n---\nTest only.\n')
        record['claim_capability'] = {'mechanism': 'pack directory flock + atomic replace',
            'evidence': {'path': 'ledger_update.py', 'sha256': hashlib.sha256(
                (self.pack/'ledger_update.py').read_bytes()).hexdigest()}}
        self.ledger.write_text('Test history is preserved.\n<!-- prompt-pack-ledger:v1 -->\n```json\n'
                               + json.dumps(record) + '\n```\n')

    def command(self, *args):
        return [sys.executable, str(self.pack / 'ledger_update.py'), *args, '--root', str(self.root)]

    def run_tool(self, *args, ok=True):
        result = subprocess.run(self.command(*args), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if ok else 1, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def claim(self, stage='S01', owner='executor-a', ok=True):
        return self.run_tool('claim', '--stage', stage, '--executor', owner, ok=ok)

    def evidence(self, stage):
        directory = self.pack / 'reports'
        directory.mkdir(exist_ok=True)
        (directory / f'{stage}.md').write_text('Synthetic receipt test report, not runtime evidence.\n')
        (directory / f'{stage}-checks.txt').write_text('Synthetic check fixture for updater tests only.\n')
        return f'reports/{stage}-checks.txt'

    def accept(self, stage, owner='executor-a', ok=True):
        return self.run_tool('accept', '--stage', stage, '--executor', owner,
                             '--checks-passed', '--evidence', self.evidence(stage), ok=ok)

    def test_sequential_receipts_and_final_completion(self):
        self.claim()
        accepted = self.accept('S01')
        self.assertEqual(accepted['next_allowed'], ['S02'])
        record = read_record(self.ledger)
        receipt = self.pack / record['stages'][0]['transition_receipt']
        saved = receipt.read_bytes()
        self.claim('S02', 'executor-b')
        self.accept('S02', 'executor-b')
        self.assertEqual(read_record(self.ledger)['ledger_status'], 'completed')
        self.assertEqual(receipt.read_bytes(), saved)
        self.assertIn('Test history is preserved.', self.ledger.read_text())
        self.claim(ok=False)

    def test_two_competing_claims_have_one_winner(self):
        processes = [subprocess.Popen(self.command('claim', '--stage', 'S01', '--executor', name),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for name in ('a', 'b')]
        outputs = [p.communicate() for p in processes]
        self.assertEqual(sorted(p.returncode for p in processes), [0, 1], outputs)
        self.assertIn(read_record(self.ledger)['stages'][0]['executor_claim'], ('a', 'b'))

    def test_directory_lock_rejects_concurrent_transaction(self):
        before = self.ledger.read_bytes()
        fd = os.open(self.pack, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.claim(ok=False)
        finally:
            os.close(fd)
        self.assertEqual(self.ledger.read_bytes(), before)
        self.claim()

    def test_foreign_owner_and_out_of_order_fail_without_mutation(self):
        before = self.ledger.read_bytes()
        self.claim('S02', ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)
        self.claim()
        before = self.ledger.read_bytes()
        self.accept('S01', 'foreign', ok=False)
        self.run_tool('resume', '--stage', 'S01', '--executor', 'foreign', ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_accept_requires_real_readable_evidence_and_report(self):
        self.claim()
        before = self.ledger.read_bytes()
        self.run_tool('accept', '--stage', 'S01', '--executor', 'executor-a', ok=False)
        self.run_tool('accept', '--stage', 'S01', '--executor', 'executor-a', '--checks-passed',
                      '--evidence', 'missing.txt', ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_changed_plan_invalidates_accepted_dependency(self):
        self.claim()
        self.accept('S01')
        (self.pack/'implementation-plan.md').write_text('Changed after receipt.\n')
        before = self.ledger.read_bytes()
        self.claim('S02', ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_need_input_resume_and_terminal_block(self):
        self.claim()
        evidence = self.evidence('S01')
        self.run_tool('needs-input', '--stage', 'S01', '--executor', 'executor-a',
                      '--reason', 'Synthetic new owner question', '--resume-condition', 'Recorded answer',
                      '--evidence', evidence)
        self.run_tool('resume', '--stage', 'S01', '--executor', 'executor-a', ok=False)
        self.run_tool('resume', '--stage', 'S01', '--executor', 'executor-a', '--resolution', evidence)
        self.run_tool('block', '--stage', 'S01', '--executor', 'executor-a',
                      '--reason', 'Synthetic hard blocker', '--evidence', evidence)
        self.run_tool('resume', '--stage', 'S01', '--executor', 'executor-a', ok=False)

    def test_changed_updater_and_missing_entry_fail(self):
        with (self.pack/'ledger_update.py').open('a') as stream:
            stream.write('\n# unbound modification\n')
        before = self.ledger.read_bytes()
        self.claim(ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_missing_future_input_preserves_executing_stage(self):
        self.claim()
        record = read_record(self.ledger)
        future = self.pack / record['stages'][1]['contract']['prompt_path']
        future.unlink()
        before = self.ledger.read_bytes()
        self.accept('S01', ok=False)
        self.assertEqual(self.ledger.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
