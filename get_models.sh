#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE_MODEL file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

mkdir -p checkpoints
wget https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip -P checkpoints

# Extract models
cd checkpoints
unzip -qq llava-fastvithd_0.5b_stage3.zip

# Clean up
rm llava-fastvithd_0.5b_stage3.zip
cd -