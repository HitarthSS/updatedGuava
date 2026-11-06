"""Lossless tool-decision transport over HarnessStudio's numeric action interface.

One encoded action represents one Guava model decision. It is never a robot
control vector. Physics is advanced and counted by the benchmark tool bridge.
"""
import json
import math

CAPACITY = 8192
ACTION_DIM = CAPACITY + 1


def encode(value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > CAPACITY:
        raise ValueError('Tool decision exceeds the transport capacity')
    return [len(data), *data, *([0] * (CAPACITY - len(data)))]


def decode(action):
    values = list(action)
    if len(values) != ACTION_DIM or any(not math.isfinite(float(x)) or float(x) != int(x) for x in values):
        raise ValueError('Invalid tool envelope shape or numeric values')
    length = int(values[0])
    if not 0 < length <= CAPACITY:
        raise ValueError('Invalid tool envelope length')
    if any(not 0 <= int(x) <= 255 for x in values[1:length + 1]) or any(values[length + 1:]):
        raise ValueError('Invalid tool envelope bytes or padding')
    value = json.loads(bytes(int(x) for x in values[1:length + 1]).decode('utf-8'))
    if not isinstance(value, dict) or value.get('kind') not in ('tool', 'terminal', 'invalid'):
        raise ValueError('Invalid decision kind')
    if value['kind'] == 'tool' and (not isinstance(value.get('name'), str) or not isinstance(value.get('arguments'), dict)):
        raise ValueError('Invalid tool call')
    return value
