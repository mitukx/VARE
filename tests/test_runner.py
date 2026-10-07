import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest

from benchmarks.scheduler.fixtures import make_fixture
from vare.runner import audit, encode, run_campaign, strict_json, write_manifest


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vare-test-')
        self.parent = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def campaign(self, behaviors):
        jobs = []
        for behavior, expected in behaviors:
            root, job = make_fixture(self.parent, behavior, timeout=.25 if behavior == 'timeout' else 3,
                                     limit=1024 if behavior == 'flood' else 1048576)
            jobs.append(job)
        output = self.parent / 'evidence'
        summary = asyncio.run(run_campaign(jobs, output, 3, root))
        for job, (_, expected) in zip(jobs, behaviors):
            record = json.loads((output / 'jobs' / job.id / 'record.json').read_text())
            self.assertEqual(expected, record['status'], job.id)
        self.assertEqual(len(jobs), audit(output)['recorded_jobs'])
        return output, summary

    def test_fault_matrix(self):
        cases = [('valid', 'candidate_passed'), ('reject', 'candidate_rejected'),
                 ('malformed', 'invalid_output'), ('duplicate', 'invalid_output'),
                 ('nonfinite', 'invalid_output'), ('crash', 'grader_error'),
                 ('wrong_task', 'provenance_mismatch'), ('fake_hash', 'provenance_mismatch'),
                 ('wrong_exit', 'invalid_output'), ('mutate', 'workspace_changed'),
                 ('timeout', 'timeout'), ('flood', 'output_limit')]
        output, _ = self.campaign(cases)
        self.assertLessEqual((output / 'jobs/flood/stdout.bin').stat().st_size, 1024)
        import subprocess
        cli = subprocess.run([__import__('sys').executable, '-m', 'vare', 'audit', str(output)], capture_output=True)
        self.assertEqual(0, cli.returncode, cli.stderr.decode())

    def test_protocol_tamper(self):
        root, job = make_fixture(self.parent, 'valid')
        (job.task_root / 'TASK.md').write_text('tampered')
        output = self.parent / 'evidence'
        summary = asyncio.run(run_campaign([job], output, 1, root))
        self.assertEqual({'protocol_error': 1}, summary['outcomes'])
        audit(output)

    def test_missing_source(self):
        root, job = make_fixture(self.parent, 'valid')
        (job.workspace / 'source.txt').unlink()
        output = self.parent / 'evidence'
        summary = asyncio.run(run_campaign([job], output, 1, root))
        self.assertEqual({'source_error': 1}, summary['outcomes'])

    def test_child_cleanup(self):
        output, _ = self.campaign([('child', 'candidate_passed')])
        pid = int((output / 'jobs/child/stderr.bin').read_text().split('=')[1])
        # Reparenting/reaping can lag the signal. A zombie has no running resources.
        for _ in range(30):
            proc = __import__('subprocess').run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
            if not proc.stdout.strip() or proc.stdout.strip().startswith('Z'):
                break
            time.sleep(.02)
        self.assertTrue(not proc.stdout.strip() or proc.stdout.strip().startswith('Z'))

    def test_inherited_pipe_times_out(self):
        root, job = make_fixture(self.parent, 'inherited_pipe', timeout=.25)
        output = self.parent / 'evidence'
        summary = asyncio.run(run_campaign([job], output, 1, root))
        self.assertEqual({'timeout': 1}, summary['outcomes'])
        audit(output)

    def test_snapshot_binding_with_rebuilt_manifest(self):
        output, _ = self.campaign([('valid', 'candidate_passed')])
        brief = next(output.glob('snapshot/*/TASK.md'))
        brief.write_text('tampered snapshot')
        (output / 'manifest.json').unlink()
        write_manifest(output)
        with self.assertRaises(ValueError):
            audit(output)

    def test_audit_tampering(self):
        output, _ = self.campaign([('valid', 'candidate_passed')])
        path = output / 'jobs/valid/stdout.bin'
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            audit(output)
        path.write_bytes(original)
        (output / 'extra.txt').write_text('extra')
        with self.assertRaises(ValueError):
            audit(output)
        (output / 'extra.txt').unlink()
        summary_path = output / 'summary.json'
        summary = json.loads(summary_path.read_text())
        summary['outcomes'] = {'candidate_rejected': 1}
        summary_path.write_bytes(encode(summary))
        # Even rebuilding the manifest cannot hide inconsistency with raw records.
        (output / 'manifest.json').unlink()
        write_manifest(output)
        with self.assertRaises(ValueError):
            audit(output)

    def test_ledger_tamper_with_rebuilt_manifest(self):
        output, _ = self.campaign([('valid', 'candidate_passed')])
        path = output / 'ledger.json'
        ledger = json.loads(path.read_text())
        ledger[0]['previous_hash'] = '1' * 64
        path.write_bytes(encode(ledger))
        (output / 'manifest.json').unlink()
        write_manifest(output)
        with self.assertRaises(ValueError):
            audit(output)

    def test_concurrency_and_output_preservation(self):
        root, job = make_fixture(self.parent, 'valid', delay=.2)
        jobs = [replace(job, id='j' + str(i)) for i in range(4)]
        output = self.parent / 'evidence'
        summary = asyncio.run(run_campaign(jobs, output, 2, root))
        self.assertEqual({'candidate_passed': 4}, summary['outcomes'])
        records = [json.loads((output / 'jobs' / j.id / 'record.json').read_text()) for j in jobs]
        self.assertGreater(min(r['queue_seconds'] for r in records[2:]), .1)
        with self.assertRaises(FileExistsError):
            asyncio.run(run_campaign(jobs, output, 2, root))
        audit(output)

    def test_cancellation_retains_partial_bundle(self):
        root, job = make_fixture(self.parent, 'timeout', timeout=10)
        output = self.parent / 'evidence'
        async def interrupt():
            task = asyncio.create_task(run_campaign([job], output, 1, root))
            await asyncio.sleep(.2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(interrupt())
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual('interrupted', summary['status'])
        self.assertEqual({'cancelled': 1}, summary['outcomes'])
        audit(output)

    def test_validation(self):
        root, job = make_fixture(self.parent, 'valid')
        for jobs, workers in [([job, job], 1), ([replace(job, id='../escape')], 1),
                              ([job], 0), ([replace(job, timeout_seconds=float('nan'))], 1),
                              ([replace(job, workspace=root)], 1)]:
            with self.assertRaises(ValueError):
                asyncio.run(run_campaign(jobs, self.parent / 'evidence', workers, root))
        self.assertFalse((self.parent / 'evidence').exists())

    def test_json_and_source_symlink(self):
        for raw in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}']:
            with self.assertRaises(ValueError):
                strict_json(raw)
        root, job = make_fixture(self.parent, 'valid')
        (job.workspace / 'source.txt').unlink()
        (job.workspace / 'source.txt').symlink_to(job.task_root / 'TASK.md')
        result = asyncio.run(run_campaign([job], self.parent / 'evidence', 1, root))
        self.assertEqual({'source_error': 1}, result['outcomes'])

    def test_cli_plan_operational_failure_and_audit(self):
        import subprocess
        import sys
        from vare.runner import ROOT
        workspace = self.parent / 'empty-candidate'
        workspace.mkdir()
        plan = self.parent / 'input-plan.json'
        plan.write_bytes(encode({'jobs': [{'id': 'missing',
            'task_root': str(ROOT / 'benchmarks/historical/rvl_behavior_policy_parity'),
            'workspace': 'empty-candidate'}]}))
        output = self.parent / 'cli-evidence'
        result = subprocess.run([sys.executable, '-m', 'vare', 'run', '--plan', str(plan),
                                 '--output', str(output), '--workers', '2'], capture_output=True)
        self.assertEqual(1, result.returncode, result.stderr.decode())
        self.assertEqual({'source_error': 1}, json.loads(result.stdout)['outcomes'])
        checked = subprocess.run([sys.executable, '-m', 'vare', 'audit', str(output)], capture_output=True)
        self.assertEqual(0, checked.returncode, checked.stderr.decode())


if __name__ == '__main__':
    unittest.main()
