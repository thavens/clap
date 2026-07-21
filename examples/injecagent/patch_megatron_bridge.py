# Copyright 2026 Individual Contributor: Michael Glaese
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make Megatron Bridge's transformers Auto* registrations idempotent."""

from __future__ import annotations

import re
import site
from pathlib import Path


def main() -> None:
    patched = 0
    for site_dir in map(Path, site.getsitepackages()):
        root = site_dir / "megatron" / "bridge" / "models"
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            source = path.read_text()
            updated = re.sub(
                r"(Auto\w+\.register\([^()]*?)\)",
                lambda match: (
                    match.group(1) + ", exist_ok=True)" if "exist_ok" not in match.group(1) else match.group(0)
                ),
                source,
            )
            if updated != source:
                path.write_text(updated)
                patched += 1
    print(f"transformers-register patch: files patched = {patched}")


if __name__ == "__main__":
    main()
