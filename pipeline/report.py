import json
import os

RESULTS_FILE = 'output/results.json'

def run():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    
    winner = data['winner']
    orig = data['original_mb']
    all_r = data['all_results']
    hardware = data.get('hardware', 'cpu')
    priority = data.get('priority', 'balanced')
    
    size_reduction = round((1 - winner['size_mb'] / orig) * 100, 1) if orig > 0 else 0
    
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY', 'summary.md')
    
    lines = []
    lines.append('# 🤖 Universal ML Deploy Report\n')
    lines.append(f'## 🎯 Hardware: `{hardware.upper()}` | Priority: `{priority.upper()}`\n')
    lines.append(f'## 🏆 Winner: `{winner["technique"].upper()}`\n')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.append(f'| Original Size | {orig} MB |')
    lines.append(f'| Optimized Size | {winner["size_mb"]} MB |')
    lines.append(f'| Size Reduction | **{size_reduction}%** |')
    lines.append(f'| Accuracy Drop | {winner["acc_drop"]}% |')
    lines.append(f'| Inference Time | {winner["infer_ms"]} ms |')
    lines.append(f'| Passes Gates | {"✅ Yes" if winner.get("gate_pass") else "⚠️ No"} |')
    lines.append('')
    lines.append('## 📊 All Techniques\n')
    lines.append('| Technique | Size (MB) | Acc Drop | Inference (ms) | Gates | Status |')
    lines.append('|-----------|-----------|----------|----------------|-------|--------|')
    
    for r in all_r:
        if r.get('status') == 'ok':
            trophy = ' 🏆' if r['technique'] == winner['technique'] else ''
            gates = '✅' if r.get('gate_pass') else '⚠️'
            lines.append(
                f'| {r["technique"]}{trophy} | {r["size_mb"]} | '
                f'{r["acc_drop"]}% | {r["infer_ms"]} ms | {gates} | ✅ |'
            )
        else:
            lines.append(f'| {r["technique"]} | - | - | - | - | ❌ {r.get("error", "Failed")[:30]} |')
    
    lines.append('')
    lines.append('## 📦 Deploy Package')
    lines.append('Download `deploy-package` artifact for ready-to-use model.')
    
    with open(summary_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print('📋 Report written!')
    print('\n'.join(lines))

if __name__ == '__main__':
    run()