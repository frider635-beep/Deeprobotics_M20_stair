#!/usr/bin/env python3
"""Check evaluation selection/reporting without loading or rolling out policies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('m20_eval_cli_under_test', ROOT/'scripts/tools/evaluate_m20_refinement.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix='m20_eval_cli_') as directory:
    p=Path(directory)
    for name in ('model.xml','candidate.pt','baseline.pt','flat.xml'):
        (p/name).write_text('CLI fixture only: no policy is loaded')
    common=['evaluate_m20_refinement.py','--model',str(p/'model.xml'),'--candidate',str(p/'candidate.pt'),
            '--terrain-dir',str(p),'--case','flat_zero','--output',str(p/'report.json')]
    def fake_case(*args):
        return {'metrics':{'action_jitter':{k:{'mean':.01} for k in
                ('leg_action_second_difference_rms','wheel_action_second_difference_rms')}},
                'gates':{'fixture_gate':True}}
    with patch.object(sys,'argv',common), patch.object(module.d,'load_actor',return_value=object()) as load, \
         patch.object(module,'run_case',side_effect=fake_case),contextlib.redirect_stdout(io.StringIO()):
        module.main()
        assert load.call_count==1
        assert load.call_args.args[0]==p/'candidate.pt'
    result=json.loads((p/'report.json').read_text())
    assert list(result['runs'])==['candidate'] and result['candidate_all_gates_pass']
    with patch.object(sys,'argv',common+['--baseline',str(p/'baseline.pt')]), \
         patch.object(module.d,'load_actor',return_value=object()) as load, \
         patch.object(module,'run_case',side_effect=fake_case),contextlib.redirect_stdout(io.StringIO()):
        module.main()
        assert load.call_count==2
    assert set(json.loads((p/'report.json').read_text())['runs'])=={'baseline','candidate'}
    with patch.object(sys,'argv',common+['--require-pass']),patch.object(module.d,'load_actor',return_value=object()), \
         patch.object(module,'run_case',return_value={'metrics':{},'gates':{'fixture_gate':False}}), \
         contextlib.redirect_stdout(io.StringIO()):
        try:
            module.main()
        except SystemExit as error:
            assert error.code==2
        else:
            raise AssertionError('A failed gate must return exit status 2')
    assert not json.loads((p/'report.json').read_text())['candidate_all_gates_pass']
print(json.dumps({'no_policy_loading_or_rollout':True,'candidate_only_is_default':True,
                  'baseline_runs_only_when_requested':True,'failed_gate_preserves_report_and_exit_2':True},indent=2))
