import os

with open('finetune_lora.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"cuda"', '"cpu"')
content = content.replace('.cuda()', '.cpu()')
content = content.replace('torch.cuda.set_device', '#torch.cuda.set_device')

with open('finetune_lora.py', 'w', encoding='utf-8') as f:
    f.write(content)
