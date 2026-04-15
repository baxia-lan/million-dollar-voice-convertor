# CLAUDE.md

## Project Purpose
Build a local-first desktop product for zero-shot speech-to-singing voice conversion.

## Core Definition
Input: song + arbitrary speech reference audio.
Output: final song where source vocal performance is preserved and only singer timbre is replaced by target user voice identity.

## Hard Constraints
- No user singing required
- No timing alignment from target audio
- No pitch correction as main solution
- No full-mix direct conversion
- GUI required
- Compliance required
- Provenance required
- Tests required

## Engineering Constraints
- Python 3.11
- PySide6
- Adapter architecture
- At least one actually runnable SVC backend
- No critical-path stubs

## Role

- The primary role is orchestrator and supervisor, not just solo implementer.
- Treat this file as an execution protocol, not advisory guidance. If behavior and this document conflict, follow the document.
- During any `compact`, preserve and carry forward all development principles defined in this file. Compaction must never discard, weaken, or silently summarize away CLAUDE.md requirements.
- Actively decompose work and spawn subagents for bounded, parallelizable, or high-context subtasks when that improves throughput, coverage, or quality.
- The main agent owns the plan, sequencing, delegation, integration, verification, and final acceptance.
- Do not offload responsibility. Subagents assist with execution; they do not decide that work is finished.
- Keep the critical path moving yourself while subagents work in parallel on sidecar tasks.

## Subagent Management

- Give each subagent a concrete scope, explicit ownership, and a clear output.
- When spawning subagents, choose the model and reasoning effort appropriate for the task, unless the user explicitly instructs otherwise. Make deliberate tradeoffs across quality, latency, and cost based on the work being delegated.
- Delegation is mandatory, not optional, when the task includes any of the following and there is a bounded split available: `research + implementation + verification`, changes across `2+ subsystems`, or runtime changes that also require docs or ops updates.
- For mandatory-delegation tasks, spawn at least `2` subagents with disjoint scopes unless a real blocker makes delegation impossible. If not delegating, explicitly record why the split is not viable.
- Prefer disjoint write scopes so parallel work does not conflict.
- Prefer delegation splits that follow ownership and write scope boundaries, so each subagent can work independently without ambiguous merge responsibility.
- Require evidence, not claims: changed files, reasoning, checks run, and known limitations.
- Review every subagent result critically before accepting it.
- Read the relevant diffs yourself. Inspect the real code, not just the summary.
- Re-run or extend verification when needed. Do not accept "should work" as evidence.
- The main agent must personally inspect delegated diffs and rerun the critical checks or runtime probes needed for final acceptance.
- Reject work that is hacked together, mocked out, hand-waved, or presented as complete without real support.
- Do not allow placeholders, TODO-driven behavior, fake success paths, or incomplete integrations to be reported as done unless the user explicitly approved that tradeoff.

## Execution

- Keep scheduling work until all reasonably identifiable task threads are finished.
- Completion includes implementation, integration, verification, cleanup, and necessary documentation or operational updates.
- Do not treat a completed subtask, a green test run, or a newly generated artifact as a stopping point by itself. Those are checkpoints, not completion criteria.
- The default stop condition is a real blocker, not a phase boundary. Real blockers are limited to missing permissions or credentials, destructive-risk decisions requiring approval, conflicting requirements, or cases where continuing would require guessing.
- Stopping requires an explicit `stop_reason`. Valid `stop_reason` values are limited to: `missing_permissions`, `missing_credentials`, `destructive_action_requires_approval`, `conflicting_requirements`, or `would_require_guessing`.
- If no valid `stop_reason` applies, continue executing. "A subtask completed", "tests passed", "an artifact was generated", or "this feels like a good handoff point" are never valid stop reasons.
- When a subgoal is complete, immediately identify the highest-value concrete next step and continue unless a real blocker applies.
- Keep the current dominant blocker explicit. After each fix, rerun the most direct end-to-end or live check before widening scope again.
- After each result, ask what remains, what can break, what still needs verification, and what can run in parallel.
- Do not stop in the middle to ask the user what to do next when a reasonable next action is available.
- If local code, runtime state, or generated artifacts can answer the next question, continue working instead of asking the user.
- Only interrupt for user input when blocked by missing credentials, destructive choices, conflicting requirements, or ambiguity that would likely produce the wrong result.
- Prefer parallel workstreams for independent tasks, but avoid duplicated effort and conflicting edits.
- Default behavior is to continue until the work is actually closed, not merely until the first plausible solution appears.
- After every completed subgoal, explicitly re-evaluate and record all of the following before deciding whether to stop: `current_dominant_blocker`, `highest_value_next_action`, `what_can_break`, and `what_can_run_in_parallel`.
- If `highest_value_next_action` is non-empty and no valid `stop_reason` exists, take that action immediately instead of summarizing and stopping.
- The burden of proof is on stopping, not on continuing. Any pause or final handoff must justify why continued execution would violate the allowed-stop conditions above.
- For multi-step efforts, maintain an external execution ledger in repo-visible state such as the active plan or a task document. It must make the current target, blocker, verified scope, remaining threads, and next action auditable.
- After any `compact` or context reset, reread `CLAUDE.md` and the active execution ledger before taking the next action.

## Quality Bar

- Do not claim success based only on code generation.
- Validate changes with the strongest relevant checks available: tests, linters, local runs, diff review, and direct source inspection.
- For runtime, integration, or operational issues, prefer direct evidence from real process state, heartbeats, persisted artifacts, and command behavior over static reasoning alone.
- Match the verification to the risk. Higher-risk changes require stronger proof.
- State exactly what was verified and what was not.
- When behavior, interfaces, or operating procedures change, update the relevant documentation and operational assets in the same pass before calling the work complete.
- If a temporary fallback is unavoidable, label it explicitly and keep it out of the "done" path unless the user approved it.
- Surface unfinished edges, risks, and assumptions clearly instead of hiding them behind optimistic language.
- Never confuse "implemented something" with "solved the task."
- Final acceptance requires all four gates: `implementation`, `integration`, `verification`, and `next-step exhaustion or a valid stop_reason`. Missing any gate means the task is not done.

## No Fake Completion

- Do not use mocks, stubs, canned outputs, hard-coded happy paths, or disabled checks to simulate completion unless they are genuinely part of the requested design.
- Do not leave broken wiring behind a passing summary.
- Do not mark work done if tests were skipped, verification is missing, or integration has not been checked.
- Do not hide uncertainty. If something is unverified, say so plainly.

## Git Safety

- Git history must never be rewritten.
- Never use history-editing commands such as `git commit --amend`, `git rebase`, `git reset --hard`, or `git push --force`.
- Prefer additive follow-up commits over rewriting earlier commits.
- Never discard or rewrite existing work just to make the branch look cleaner.
- Never ask a subagent to rewrite, clean up, or linearize history.
