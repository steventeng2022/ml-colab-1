"""Run end-to-end regression checks using generated data, without competition credentials."""
from pathlib import Path
import contextlib
import io
import json
import os
import argparse
import tempfile
import hashlib
import zipfile
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--notebook', type=Path, help='Notebook path; defaults to the repository notebook')
parser.add_argument('--artifacts', type=Path, help='Keep generated fixtures and charts in this directory')
args = parser.parse_args()
temporary = tempfile.TemporaryDirectory(prefix='bdb_notebook_qa_') if args.artifacts is None else None
QA = Path(temporary.name) if temporary else args.artifacts.resolve()
QA.mkdir(parents=True, exist_ok=True)
NOTEBOOK = args.notebook.resolve() if args.notebook else ROOT / 'nfl_big_data_bowl_2027.ipynb'
os.environ['MPLCONFIGDIR'] = str(QA / 'matplotlib_config')
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

DATA = QA / 'data'
DATA.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1427)
players, results, combine, pp, games = [], [], [], [], []
game_rows = {year: [] for year in (2023, 2024, 2025)}
for year in (2023, 2024, 2025):
    for game_index in range(6):
        games.append({'game_id': year * 10000 + game_index + 1, 'season': year, 'season_type': 'REG'})
    for player in range(14):
        pid = year * 100 + player
        name = f'Fixture Receiver {year}-{player}'
        players.append({'nfl_id': pid, 'draft_year': year, 'nfl_position': 'WR', 'display_name': name})
        forty = rng.uniform(4.3, 4.7)
        latent = 10.5 - 2 * (forty - 4.3) + rng.normal(0, .25)
        results.append({'nfl_id': pid, 'draft_year': year, 'forty': forty,
                        'ten_yd_split': np.nan if player % 5 == 0 else forty / 2.8,
                        'combine_height': rng.normal(73, 2), 'combine_weight': rng.normal(198, 8)})
        start = pd.Timestamp(f'{year}-03-01T12:00:00Z') + pd.Timedelta(seconds=player * 100)
        for attempt in (1, 2):
            for f in range(30):
                combine.append({'nfl_id': pid, 'draft_year': year, 'event_id': f'{pid}-{attempt}',
                                'attempt': attempt, 'entity_type': 'PLAYER',
                                'time': start + pd.Timedelta(seconds=attempt * 10 + f / 10),
                                'drill_type': 'FORTY_YARD_DASH',
                                's': 1 + (latent - 1) * min(f / 20, 1) + rng.normal(0, .02)})
        ball = dict(combine[-1], entity_type='BALL', s=25)
        combine.append(ball)
        for route in range(60):
            game_id = year * 10000 + route // 10 + 1
            play_id = player * 100 + route
            label = ['GO', 'SLANT', 'CURL'][route % 3]
            route_effect = {'GO': .7, 'SLANT': -.25, 'CURL': -.6}[label]
            target = route % 4 == 0
            caught = target and route % 12 != 0
            yards = max(-3, rng.normal(12, 4)) if caught else np.nan
            if year == 2023 and player == 1 and route == 4:
                yards = np.nan  # Unresolved targeted completion must not become zero.
            pp.append({'game_id': game_id, 'play_id': play_id, 'nfl_id': pid,
                       'route_ran': label, 'target': target, 'rec_yards': yards,
                       'pass_result': 'C' if caught else 'I',
                       'separation_at_pass_forward': rng.uniform(1, 4),
                       'play_nullified_by_penalty': 'N'})
            snap = pd.Timestamp(f'{year}-09-10T12:00:00Z') + pd.Timedelta(seconds=player * 200 + route * 2)
            speed_top = latent * .68 + route_effect + rng.normal(0, .06)
            for f in range(14):
                # Enough frames for an old check to pass, but an incomplete window.
                if year == 2023 and player == 0 and route == 5 and f >= 8:
                    continue
                # Both edges are present, but an internal gap is too large.
                if year == 2023 and player == 0 and route == 6 and 6 <= f <= 9:
                    continue
                game_rows[year].append({'game_id': game_id, 'play_id': play_id, 'nfl_id': pid,
                                        'time': snap + pd.Timedelta(seconds=f / 10), 'event': '',
                                        's': 2 + (speed_top - 2) * min(f / 9, 1)})
            # Events are carried only by a different tracked player.
            for event, offset in [('ball_snap', 0), ('pass_forward', 1.3)]:
                if year == 2023 and player == 0 and route == 0 and event == 'pass_forward':
                    continue
                game_rows[year].append({'game_id': game_id, 'play_id': play_id, 'nfl_id': 999999,
                                        'time': snap + pd.Timedelta(seconds=offset), 'event': event, 's': 0})

pd.DataFrame(players + [players[0]]).to_csv(DATA / 'players.csv', index=False)
pd.DataFrame(results).to_csv(DATA / 'combine_results.csv', index=False)
pd.DataFrame(combine).to_csv(DATA / 'combine_tracking.csv', index=False)
pd.DataFrame(pp).to_csv(DATA / 'player_play.csv', index=False)
pd.DataFrame(games).to_csv(DATA / 'games.csv', index=False)
for year, rows in game_rows.items():
    if year == 2023:
        # Identical repeated frame across a 2,000-row reader boundary.
        rows.insert(2000, dict(rows[1999]))
        conflict = next(row for row in rows if row['play_id'] == 3 and row['nfl_id'] == 202300)
        rows.append(dict(conflict, s=conflict['s'] + .75))
    pd.DataFrame(rows).to_csv(DATA / f'game_tracking_{year}.csv', index=False)

nb = nbformat.read(NOTEBOOK, as_version=4)
nbformat.validate(nb)
code_cells = [cell.source for cell in nb.cells if cell.cell_type == 'code']

def execute(label, *, chunk_size=2000, test_min=5, use_cache=True, cache_suffix='normal'):
    output = QA / label
    output.mkdir(parents=True, exist_ok=True)
    ns = {'__name__': '__main__'}
    started = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for index, source in enumerate(code_cells):
            try:
                exec(compile(source, f'{label}_cell_{index}', 'exec'), ns)
            except Exception:
                (output / 'execution.log').write_text(log.getvalue(), encoding='utf-8')
                raise
            if index == 0:
                ns.update(DATA_DIR_OVERRIDE=DATA, OUTPUT_DIR=output,
                          CACHE_DIR=QA / ('cache_' + cache_suffix),
                          CHUNK_SIZE=chunk_size, BOOTSTRAPS=100, REFIT_BOOTSTRAPS=30,
                          MIN_TEST_PLAYERS=test_min, USE_CACHE=use_cache,
                          display=lambda *args, **kwargs: None)
                ns['CACHE_DIR'].mkdir(exist_ok=True)
                ns['plt'].show = lambda: plt.close('all')
    (output / 'execution.log').write_text(log.getvalue(), encoding='utf-8')
    print(f'{label}: completed {len(code_cells)} code cells in {time.time() - started:.1f}s', flush=True)
    return ns, log.getvalue()

os.chdir(QA)
baseline, _ = execute('baseline')
assert len(baseline['analysis']) == 42
assert len(baseline['route_measurements']) == 42 * 60 - 4
assert len(baseline['metrics']) == 4 * 4
assert len(baseline['development_metrics']) == 4 * 4
assert set(baseline['development_metrics']['train_classes']) == {'2023'}
assert set(baseline['development_metrics']['test_class']) == {2024}
assert len(baseline['refit_uncertainty']) == 1
refit = baseline['refit_uncertainty'].iloc[0]
assert np.isfinite(refit['lower_95']) and refit['lower_95'] <= refit['upper_95']
pooled = baseline['repeatability'].loc[baseline['repeatability']['draft_class'].eq('pooled')]
assert set(pooled['n_pairs']) == {42}
assert pooled['spearman_rho'].notna().all()
with zipfile.ZipFile(baseline['archive_path']) as archive:
    manifest = json.loads(archive.read('run_manifest.json'))
    for filename, digest in manifest['output_sha256'].items():
        assert hashlib.sha256(archive.read(filename)).hexdigest() == digest
    assert 'analysis_cohort.csv' not in archive.namelist()
    assert 'route_measurements.csv' not in archive.namelist()

# Changing holdout outcomes must not change fitted primary predictions.
changed = baseline['analysis'].copy()
changed.loc[changed['draft_year'].eq(2025), 'rookie_route_speed'] += 1.0
_, predictions = baseline['evaluate_split'](changed, 'rookie_route_speed', 2025, (2023, 2024), 'leakage_check')
original = baseline['holdout'].loc[baseline['holdout']['target'].eq('rookie_route_speed')]
for model_name in ('mean', 'forty_only', 'official', 'with_tracking'):
    np.testing.assert_allclose(predictions[model_name], original[model_name], rtol=1e-10, atol=1e-10)

# Context expectations for a training player cannot fit that player's own speed.
train_ids = set(baseline['analysis'].loc[baseline['analysis']['draft_year'].isin([2023, 2024]), 'nfl_id'])
test_ids = set(baseline['analysis'].loc[baseline['analysis']['draft_year'].eq(2025), 'nfl_id'])
context = baseline['context_outcomes']
before_context = context(train_ids, test_ids, ('regression', 'original'))
original_routes = baseline['route_measurements']
baseline['route_measurements'] = original_routes.copy()
selected_player = min(train_ids)
selected_mask = baseline['route_measurements']['nfl_id'].eq(selected_player)
baseline['route_measurements'].loc[selected_mask, 'peak_speed'] += 1.0
after_context = context(train_ids, test_ids, ('regression', 'training_perturbed'))
np.testing.assert_allclose(after_context.loc[selected_player] - before_context.loc[selected_player], 1.0, atol=1e-8)

# Holdout speeds cannot fit the training context reference either.
baseline['route_measurements'] = original_routes.copy()
test_mask = baseline['route_measurements']['nfl_id'].isin(test_ids)
baseline['route_measurements'].loc[test_mask, 'peak_speed'] += 1.0
after_holdout = context(train_ids, test_ids, ('regression', 'holdout_perturbed'))
np.testing.assert_allclose(after_holdout.loc[sorted(train_ids)], before_context.loc[sorted(train_ids)], atol=1e-8)
baseline['route_measurements'] = original_routes
assert set(baseline['matched_drill_columns']) == {'drill_type'}
unresolved = baseline['cohort'].set_index('nfl_id').loc[202301, 'rookie_yards_per_route']
assert pd.isna(unresolved)
audit = baseline['quality_audit']
assert audit.loc[audit['check'].eq('identical repeated timestamps removed'), 'value'].sum() == 1
assert audit.loc[audit['check'].eq('units excluded for conflicting duplicate speeds'), 'value'].sum() == 1

assert audit.loc[audit['check'].eq('routes rejected for incomplete snap/release coverage'), 'value'].sum() == 1
assert audit.loc[audit['check'].eq('routes rejected for internal frame gaps'), 'value'].sum() == 1

cached, cache_log = execute('cached')
assert 'Reused validated tracking cache' in cache_log
pd.testing.assert_frame_equal(baseline['metrics'], cached['metrics'], check_exact=False, rtol=1e-10, atol=1e-10)
pd.testing.assert_frame_equal(baseline['refit_uncertainty'], cached['refit_uncertainty'], check_exact=False, rtol=1e-10, atol=1e-10)

# Exercise the alternative label schema and compare different chunk boundaries.
ct = pd.read_csv(DATA / 'combine_tracking.csv')
ct['drill_name'] = ct['drill_type']
ct['drill_type'] = 'STANDARD'
ct.to_csv(DATA / 'combine_tracking.csv', index=False)
alternate, _ = execute('alternative_schema', chunk_size=100000, cache_suffix='alternative')
assert alternate['matched_drill_columns'] == ['drill_name']
pd.testing.assert_frame_equal(baseline['metrics'], alternate['metrics'], check_exact=False, rtol=1e-10, atol=1e-10)
pd.testing.assert_frame_equal(baseline['route_measurements'].sort_values(['game_id', 'play_id', 'nfl_id']).reset_index(drop=True),
                              alternate['route_measurements'].sort_values(['game_id', 'play_id', 'nfl_id']).reset_index(drop=True),
                              check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)

small, _ = execute('insufficient_holdout', test_min=999, cache_suffix='alternative')
assert small['metrics'].empty and small['uncertainty'].empty
assert set(small['model_status']['status']) == {'insufficient players'}
assert 'Primary comparison unavailable' in small['summary_text']

# A year with no release tags must be reported and omitted, rather than crash.
track_path = DATA / 'game_tracking_2024.csv'
tracking = pd.read_csv(track_path)
tracking.loc[tracking['event'].eq('pass_forward'), 'event'] = ''
tracking.to_csv(track_path, index=False)
missing, _ = execute('missing_release_year', chunk_size=100000, cache_suffix='missing')
assert 2024 not in set(missing['route_measurements']['season'])
assert len(missing['analysis']) == 28
assert not missing['metrics'].empty

for year in (2023, 2025):
    track_path = DATA / f'game_tracking_{year}.csv'
    tracking = pd.read_csv(track_path)
    tracking.loc[tracking['event'].eq('pass_forward'), 'event'] = ''
    tracking.to_csv(track_path, index=False)
empty, _ = execute('no_route_coverage', chunk_size=100000, cache_suffix='empty')
assert empty['analysis'].empty and empty['metrics'].empty
assert 'Primary comparison unavailable' in empty['summary_text']

checks = ['All notebook code cells execute on generated fixtures',
          'Identical duplicate at a chunk boundary is counted once',
          'Conflicting duplicate excludes the affected route',
          'Event tags on another tracked player define the route window',
          'Unresolved completion yards stay missing',
          'Cache reuse preserves model metrics',
          'drill_type-only and drill_name label schemas agree',
          'Small and large CSV chunks give the same results',
          'Insufficient holdout sample completes with explicit model status',
          'An entire season without release tags is reported without crashing',
          'Zero usable route coverage completes with diagnostic outputs',
          'Incomplete route edges and internal gaps are rejected separately',
          'Primary refit bootstrap produces finite paired uncertainty',
          'Earlier chronological evaluation uses only 2023 to predict 2024',
          'Combine and disjoint-game repeatability each have the expected player pairs',
          'Changing holdout outcomes leaves fitted primary predictions unchanged',
          'Training context expectations exclude the player being adjusted',
          'Changing holdout tracking speeds leaves training context outcomes unchanged',
          'The results ZIP matches manifest hashes and excludes optional player tables']
(QA / 'checks.json').write_text(json.dumps({'passed': checks, 'fixtures_only': True}, indent=2), encoding='utf-8')
os.chdir(ROOT)
print(f'PASS: {len(checks)} behavioral checks using generated fixtures.', flush=True)
if args.artifacts:
    print(f'Generated artifacts: {QA}', flush=True)
