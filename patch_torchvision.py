#!/usr/bin/env python3
"""
patch_torchvision.py — Fix torchvision::nms import crash on CPU-only builds.

Known bug: pytorch/vision#8985, pytorch/vision#9085
Affected:  torchvision==0.21.0+cpu and later CPU builds (unfixed upstream)
Symptom:   RuntimeError: operator torchvision::nms does not exist

Fix: wrap decorator + function body in try/except at module level.
"""

import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("torchvision")
if spec is None:
    print("torchvision not found — nothing to patch")
    sys.exit(0)

f = pathlib.Path(spec.origin).parent / "_meta_registrations.py"
print(f"Patching: {f}")

src = f.read_text()

TARGETS = [
    '@torch.library.register_fake("torchvision::nms")',
    "@torch.library.register_fake('torchvision::nms')",
]

target_line = None
for t in TARGETS:
    if t in src:
        target_line = t
        break

if target_line is None:
    print("Pattern not found — already patched or layout changed, skipping")
    sys.exit(0)

print(f"Found: {repr(target_line)}")

lines = src.splitlines(keepends=True)

# Find the decorator line index
decorator_idx = None
for i, line in enumerate(lines):
    if target_line in line:
        decorator_idx = i
        print(f"Decorator at line {i+1}: {repr(line.rstrip())}")
        break

if decorator_idx is None:
    print("Could not locate decorator — aborting")
    sys.exit(1)

# The structure is:
#   @torch.library.register_fake("torchvision::nms")   <- decorator_idx
#   def fake_nms(...):                                  <- decorator_idx + 1
#       <body indented>                                 <- decorator_idx + 2 ...
#
# We must capture: decorator + def line + indented body
# Block ends when we hit a non-empty line that is NOT indented,
# AFTER we have passed the def line.

block_start = decorator_idx
# Skip the decorator line itself, then find the end of the function body
block_end = decorator_idx + 1  # start after decorator

# First non-blank line after decorator must be the def — skip it and its body
found_def = False
while block_end < len(lines):
    line = lines[block_end]
    stripped = line.strip()

    if not found_def:
        # Looking for the def line
        if stripped.startswith("def "):
            found_def = True
            block_end += 1
            continue
        elif stripped == "":
            block_end += 1
            continue
        else:
            # Unexpected — stop
            break
    else:
        # Inside function body — stop at next top-level non-empty line
        if stripped and not line[0].isspace():
            break
        block_end += 1

print(f"Block: lines {block_start+1} to {block_end} ({block_end - block_start} lines)")

if not found_def:
    print("Could not find def line after decorator — aborting")
    sys.exit(1)

# Indent the entire block by 4 spaces and wrap in try/except
block = lines[block_start:block_end]
indented = ["    " + l for l in block]
replacement = (
    ["try:\n"]
    + indented
    + ["except Exception:\n",
       "    pass  # torchvision::nms not available in this CPU-only build\n"]
)

new_src = "".join(lines[:block_start] + replacement + lines[block_end:])
f.write_text(new_src)
print(f"Successfully patched {f}")