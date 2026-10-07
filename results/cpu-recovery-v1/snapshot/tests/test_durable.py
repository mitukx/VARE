import asyncio
from dataclasses import replace
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest

from benchmarks.scheduler.fixtures import make_fixture
from vare.durable import Store, FenceError, audit_durable, run_workers, worker
from vare.runner import encode, evaluate, strict_json, write_manifest


def die_after_claim(path):
    import os
    Store(path).claim('dead', .2)
    os._exit(23)


def die_in_transaction(path):
    import os
    with Store(path).connection(True) as db:
        db.execute("UPDATE jobs SET state='stale',effective='input_stale'")
        os._exit(24)


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vare-durable-test-')
        self.parent = Path(self.temporary.name)
        self.root, self.job = make_fixture(self.parent, 'valid')
        self.store = Store.create(self.parent / 'state.db', [self.job], self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def outcome(self, claim):
        return asyncio.run(evaluate(claim['job'], self.root, time.monotonic()))

    def exported(self, name='evidence'):
        output = self.parent / name
        self.store.export(output)
        return output, audit_durable(output)

    def test_idempotent_commit(self):
        claim = self.store.claim('one', 30)
        result = self.outcome(claim)
        self.assertEqual({'committed': True, 'duplicate': False}, self.store.complete(claim, result))
        self.assertEqual({'committed': True, 'duplicate': True}, self.store.complete(claim, result))
        self.assertIsNone(self.store.claim('two'))
        _, audit = self.exported()
        self.assertEqual({'candidate_passed': 1}, audit['outcomes'])
        self.assertEqual(1, audit['attempts'])

    def test_expired_attempt_is_fenced(self):
        first = self.store.claim('old', 1, now=10)
        old_result = self.outcome(first)
        second = self.store.claim('new', 30, now=12)
        new_result = self.outcome(second)
        self.assertFalse(self.store.renew(first, 30, now=12))
        self.assertTrue(self.store.complete(second, new_result, now=13)['committed'])
        self.assertFalse(self.store.complete(first, old_result, now=13)['committed'])
        self.assertTrue(self.store.complete(first, old_result, now=13)['duplicate'])
        _, checked = self.exported()
        self.assertEqual(2, checked['attempts'])
        self.assertEqual({'candidate_passed': 1}, checked['outcomes'])

    def test_expiry_without_replacement_cannot_commit(self):
        claim = self.store.claim('old', 1, now=10)
        self.assertFalse(self.store.complete(claim, self.outcome(claim), now=12)['committed'])
        self.assertEqual('running', self.store.jobs()[0]['state'])
        # Expiry later transitions a fenced attempt to expired, preserving its raw outcome.
        self.store.claim('new', 5, now=13)
        self.exported()

    def test_stale_completed_result_is_retained_and_disabled(self):
        asyncio.run(worker(self.store, 'one'))
        (self.job.workspace / 'source.txt').write_text('new input')
        asyncio.run(worker(self.store, 'two'))
        self.assertEqual('stale', self.store.jobs()[0]['state'])
        output, checked = self.exported()
        self.assertEqual({'input_stale': 1}, checked['outcomes'])
        historical = strict_json((output / 'attempts/valid/1/record.json').read_bytes())
        self.assertEqual('candidate_passed', historical['status'])

    def test_stale_pending_input_never_executes(self):
        (self.job.workspace / 'source.txt').write_text('new input')
        asyncio.run(worker(self.store, 'one'))
        _, checked = self.exported()
        self.assertEqual(0, checked['attempts'])
        self.assertEqual({'input_stale': 1}, checked['outcomes'])

    def test_changed_protocol_invalidates(self):
        (self.job.task_root / 'TASK.md').write_text('changed')
        asyncio.run(worker(self.store, 'one'))
        self.assertEqual({'input_stale': 1}, self.exported()[1]['outcomes'])

    def test_change_between_evaluation_and_commit(self):
        claim = self.store.claim('one', 30)
        result = self.outcome(claim)
        (self.job.workspace / 'source.txt').write_text('changed after evaluation')
        self.assertFalse(self.store.complete(claim, result)['committed'])
        output, checked = self.exported()
        self.assertEqual({'input_stale': 1}, checked['outcomes'])
        self.assertEqual('candidate_passed', strict_json((output / 'attempts/valid/1/record.json').read_bytes())['status'])

    def test_worker_heartbeat_during_slow_evaluation(self):
        parent = self.parent / 'slow'
        parent.mkdir()
        root, job = make_fixture(parent, 'valid', delay=.7)
        store = Store.create(parent / 'state.db', [job], root)
        async def work():
            await asyncio.gather(worker(store, 'one', .3), worker(store, 'two', .3))
        asyncio.run(work())
        self.assertEqual(1, store.jobs()[0]['token'])
        self.assertEqual('candidate_passed', store.jobs()[0]['effective'])

    def test_multiprocess_claims(self):
        path = self.parent / 'many.db'
        jobs = [replace(self.job, id='job-%02d' % i) for i in range(12)]
        store = Store.create(path, jobs, self.root)
        run_workers(path, 4, 2)
        self.assertTrue(all(r['token'] == 1 and r['effective'] == 'candidate_passed' for r in store.jobs()))
        output = self.parent / 'many-evidence'
        store.export(output)
        self.assertEqual(12, audit_durable(output)['attempts'])

    def test_process_crash_after_claim_recovers(self):
        child = multiprocessing.get_context('spawn').Process(target=die_after_claim, args=(self.store.path,))
        child.start()
        child.join(5)
        self.assertEqual(23, child.exitcode)
        time.sleep(.25)
        run_workers(self.store.path, 2, 2)
        self.assertEqual(2, self.store.jobs()[0]['token'])
        self.assertEqual({'candidate_passed': 1}, self.exported()[1]['outcomes'])

    def test_process_death_rolls_back_uncommitted_state(self):
        child = multiprocessing.get_context('spawn').Process(target=die_in_transaction, args=(self.store.path,))
        child.start()
        child.join(5)
        self.assertEqual(24, child.exitcode)
        self.assertEqual('pending', self.store.jobs()[0]['state'])
        run_workers(self.store.path)
        self.assertEqual({'candidate_passed': 1}, self.exported()[1]['outcomes'])

    def test_retry_budget(self):
        for token in range(1, 4):
            claim = self.store.claim('lost', 1, now=token * 2)
            self.assertEqual(token, claim['token'])
        self.assertIsNone(self.store.claim('next', 1, now=10))
        self.assertEqual({'retry_exhausted': 1}, self.exported()[1]['outcomes'])

    def test_conflicting_duplicate_is_rejected(self):
        claim = self.store.claim('one', 30)
        outcome = self.outcome(claim)
        self.store.complete(claim, outcome)
        record = dict(outcome[0])
        record['execution_seconds'] += 1
        with self.assertRaises(FenceError):
            self.store.complete(claim, (record, *outcome[1:]))

    def test_invalid_score_cannot_be_committed(self):
        claim = self.store.claim('one', 30)
        outcome = self.outcome(claim)
        record = dict(outcome[0])
        record['input'] = dict(record['input'], candidate_diff_sha256='0' * 64)
        with self.assertRaises(ValueError):
            self.store.complete(claim, (record, *outcome[1:]))

    def test_audit_replays_state(self):
        asyncio.run(worker(self.store, 'one'))
        output, _ = self.exported()
        path = output / 'state.json'
        state = strict_json(path.read_bytes())
        state['jobs'][0]['effective'] = 'candidate_rejected'
        path.write_bytes(encode(state))
        (output / 'manifest.json').unlink()
        write_manifest(output)
        with self.assertRaises(ValueError):
            audit_durable(output)

    def test_forged_claim_expected_cannot_commit(self):
        import copy
        from vare.runner import digest
        claim = self.store.claim('one', 30)
        record, stdout, stderr, data = self.outcome(claim)
        fake = copy.deepcopy(claim)
        fake['expected']['input']['candidate_diff_sha256'] = '0' * 64
        record['input']['candidate_diff_sha256'] = '0' * 64
        data['candidate_diff_sha256'] = '0' * 64
        with self.assertRaises(FenceError):
            self.store.complete(fake, (record, encode(data), stderr, data))
        self.assertEqual('running', self.store.jobs()[0]['state'])

    def test_export_recomputes_freshness(self):
        asyncio.run(worker(self.store, 'one'))
        (self.job.workspace / 'source.txt').write_text('changed')
        self.assertEqual({'input_stale': 1}, self.exported()[1]['outcomes'])

    def test_duplicate_after_invalidation_retains_identity(self):
        claim = self.store.claim('one', 30)
        outcome = self.outcome(claim)
        self.store.complete(claim, outcome)
        (self.job.workspace / 'source.txt').write_text('changed')
        self.assertTrue(self.store.complete(claim, outcome)['duplicate'])
        self.assertEqual({'input_stale': 1}, self.exported()[1]['outcomes'])

    def test_linked_state_path_rejected(self):
        link = self.parent / 'linked.db'
        link.symlink_to(self.store.path)
        with self.assertRaises(ValueError):
            Store(link)

    def test_audit_rejects_completion_after_terminal(self):
        from vare.runner import digest
        asyncio.run(worker(self.store, 'one'))
        output, _ = self.exported()
        path = output / 'state.json'
        state = strict_json(path.read_bytes())
        previous = state['events'][-1]
        event = {key: value for key, value in previous.items() if key != 'event_sha256'}
        event.update(kind='fenced', sequence=event['sequence'] + 1, previous_hash=previous['event_sha256'])
        event['event_sha256'] = digest(encode(event))
        state['events'].append(event)
        path.write_bytes(encode(state))
        attempt_path = output / 'attempts/valid/1/attempt.json'
        attempt = strict_json(attempt_path.read_bytes())
        attempt['state'] = 'fenced'
        attempt_path.write_bytes(encode(attempt))
        (output / 'manifest.json').unlink()
        write_manifest(output)
        with self.assertRaises(ValueError):
            audit_durable(output)


if __name__ == '__main__':
    unittest.main()
