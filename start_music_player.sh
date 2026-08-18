#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
exec python3 music_player.py "$@"