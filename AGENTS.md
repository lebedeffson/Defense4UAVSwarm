# AGENTS.md

## Core Rule

Use this file as the compact project memory index. Keep it short. Put details in `.codex/notes/` only when they are useful for future tasks.

## Work Style

* Speak briefly.
* Save tokens: avoid long explanations unless asked.
* Work cleanly and quickly: do first, then report briefly.
* Do not print large logs; show only key lines and errors.
* Avoid long plans except for complex tasks.
* Use `rg` and parallel reads; do not scan unrelated files.
* Do not add large `outputs/` or `data/` artifacts to git.
* Final response: changed files, checks run, commit/push if done, and whether project memory was updated.

## Always Follow

* Inspect existing patterns before editing.
* Prefer minimal, targeted changes.
* Fix root causes instead of symptoms.
* Avoid new dependencies unless necessary.
* Run the smallest relevant verification command after code changes.
* Do not store secrets, tokens, credentials, private keys, or sensitive environment values in project memory.

## Notes Index

Read these only when relevant to the current task:

* `.codex/notes/commands.md` — verified build, test, lint, type-check, and dev commands.
* `.codex/notes/debugging.md` — recurring bugs, failed assumptions, and non-obvious fixes.
* `.codex/notes/architecture.md` — module boundaries, naming conventions, and integration constraints.
* `.codex/notes/style.md` — code style conventions not obvious from formatters.

## Memory Update Policy

After each task, update memory only if the lesson is durable and likely to prevent future mistakes.

Use `AGENTS.md` for:

* Rules that apply across most tasks.
* Short project-wide conventions.
* Pointers to detailed notes.

Use `.codex/notes/*.md` for:

* Debugging findings.
* Long explanations.
* Area-specific conventions.
* Commands with caveats.
* Mistakes that should not be repeated.

Do not add:

* Generic programming advice.
* Temporary task notes.
* Duplicates.
* Guesses.
* Secrets or sensitive values.

When adding a lesson, write it as: "When X, do Y because Z."
