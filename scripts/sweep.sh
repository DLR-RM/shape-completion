#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/activate_env.sh"

cd "$(dirname "$0")" || exit
wandb agent --count 1 "$1"