#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилита для генерации лицензионных ключей для SAED Suite.
Храните этот файл в секрете и не распространяйте его.
"""

import argparse
import hashlib
import hmac
import secrets
import textwrap
from typing import Optional

# Этот секрет ДОЛЖЕН совпадать с секретом в основном приложении (pipeline_app.py)
LICENSE_SECRET = "ChangeMeToAPrivateSecret"

def generate_license_key(label: Optional[str] = None) -> str:
    """Генерирует новый лицензионный ключ, опционально используя метку."""
    label_text = (label or "").strip()
    base_seed = secrets.token_hex(8).upper()
    if label_text:
        label_digest = hashlib.sha256(label_text.upper().encode("utf-8")).hexdigest().upper()
        random_part = (label_digest[:8] + base_seed)[:16]
    else:
        random_part = base_seed[:16]
    checksum = hmac.new(
        LICENSE_SECRET.encode("utf-8"),
        random_part.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:8].upper()
    raw_key = random_part + checksum
    return "-".join(textwrap.wrap(raw_key, 4))

def _build_cli_parser() -> argparse.ArgumentParser:
    """Создает интерфейс командной строки для генератора ключей."""
    parser = argparse.ArgumentParser(description="Генератор лицензионных ключей SAED Symmetry")
    parser.add_argument(
        "label",
        metavar="LABEL",
        nargs="?",
        help=(
            "Опциональная метка (например, имя клиента) для генерации ключа."
        ),
    )
    return parser

if __name__ == "__main__":
    parser = _build_cli_parser()
    args = parser.parse_args()
    key = generate_license_key(args.label or None)
    print("Сгенерированный лицензионный ключ:")
    print(key)