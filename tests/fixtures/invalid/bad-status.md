---
id: pr-fixture-bad-status
repo: example/repo
pr: 1
title: Fixture — Invalid PR Status
author: tester
date: '2026-06-30'
url: https://example.com/pr/1
source_category: upstream-code
architectures: [sm75]
tags: [tensor-core]
captured_at: '2026-06-30'
status: banana
---
INVALID FIXTURE: `status: banana` is not in the source-pr status enum
[open, merged, closed]. The validator must reject it (generic enum constraint).
