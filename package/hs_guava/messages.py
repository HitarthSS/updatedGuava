"""Request construction shared by the native and bounded-context server paths."""
from copy import deepcopy
import hashlib
import json

from .feedback import HISTORY_MODE


def prepare_messages(history, request):
    mode = request.get('history_mode', 'native')
    if mode == 'native':
        history.append(deepcopy(request['message']))
        return history
    if mode != HISTORY_MODE:
        raise ValueError('Unknown Guava history mode')
    message = request['message']
    content = message.get('content', [])
    images = [part for part in content if part.get('type') == 'image_url']
    texts = [part.get('text', '') for part in content if part.get('type') == 'text']
    if message.get('role') != 'user' or len(images) != 1 or len(texts) != 1 or len(content) != 2:
        raise ValueError('Bounded-context request must contain one current image and one text block')
    if len(texts[0]) > 24000:
        raise ValueError('Bounded-context text exceeds the reviewed limit')
    return [{'role': 'system', 'content': request['system_prompt']}, deepcopy(message)]


def request_audit(messages):
    image_hashes = []
    text_chars = 0
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, str):
            text_chars += len(content)
        else:
            for part in content:
                if part.get('type') == 'image_url':
                    image_hashes.append(hashlib.sha256(part['image_url']['url'].encode()).hexdigest())
                elif part.get('type') == 'text':
                    text_chars += len(part.get('text', ''))
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return {'request_sha256': hashlib.sha256(raw).hexdigest(), 'image_sha256': image_hashes,
            'image_count': len(image_hashes), 'text_chars': text_chars, 'message_count': len(messages)}
