import json
import sys
import os

RESULTS_FILE = 'output/results.json'
HARDWARE = os.environ.get('HARDWARE_TARGET', 'cpu')

GATES = {
    'cpu': {'size': 200, 'infer': 1000, 'acc': 5.0},
    'raspberry_pi': {'size': 50, 'infer': 2000, 'acc': 10.0},
    'nvidia_gpu': {'size': 500, 'infer': 100, 'acc': 2.0},
    'mobile_android': {'size': 50, 'infer': 500, 'acc': 5.0},
    'mobile_ios': {'size': 50, 'infer': 500, 'acc': 5.0},
    'edge_tpu': {'size': 8, 'infer': 100, 'acc': 10.0}
}

def run():
    gates = GATES.get(HARDWARE, GATES['cpu'])
    
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    
    winner = data['winner']
    orig = data['original_mb']
    
    print('\n' + '='*60)
    print(f' 🔍 QUALITY GATES — {HARDWARE.upper().replace("_", " ")}')
    print('='*60)
    print(f' Technique : {winner["technique"]}')
    print(f' Original  : {orig} MB')
    print(f' Optimized : {winner["size_mb"]} MB')
    print(f' Acc drop  : {winner["acc_drop"]}%')
    print(f' Inference : {winner["infer_ms"]} ms')
    print('='*60)
    
    failures = []
    
    if winner['acc_drop'] > gates['acc']:
        failures.append(f'❌ Acc drop {winner["acc_drop"]}% > {gates["acc"]}%')
    else:
        print(f' ✅ Acc drop {winner["acc_drop"]}% ≤ {gates["acc"]}%')
    
    if winner['size_mb'] > gates['size']:
        failures.append(f'❌ Size {winner["size_mb"]}MB > {gates["size"]}MB')
    else:
        print(f' ✅ Size {winner["size_mb"]}MB ≤ {gates["size"]}MB')
    
    if winner['infer_ms'] > gates['infer']:
        failures.append(f'❌ Inference {winner["infer_ms"]}ms > {gates["infer"]}ms')
    else:
        print(f' ✅ Inference {winner["infer_ms"]}ms ≤ {gates["infer"]}ms')
    
    print('='*60)
    
    if failures:
        print('\n🚫 GATE FAILED:')
        for f in failures:
            print(f'   {f}')
        sys.exit(1)
    else:
        print('\n✅ ALL GATES PASSED — Ready for deployment!')

if __name__ == '__main__':
    run()