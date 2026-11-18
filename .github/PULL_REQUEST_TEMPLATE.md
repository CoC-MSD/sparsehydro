## Summary

<!-- What does this pull request change, and why? One or two short paragraphs. -->

Closes #

## Type of change

<!-- Tick all that apply. -->

- [ ] Bug fix
- [ ] New model formulation or component
- [ ] New feature (other)
- [ ] Refactor with no change in results
- [ ] Documentation only
- [ ] **Breaking change** (changes a public interface or existing numerical results)

## How it was tested

<!-- Which tests did you add or run? For numerical changes, include a short
before/after comparison (a table or a few numbers is enough). -->

## Checklist

- [ ] The pull request targets `dev`.
- [ ] `pytest` passes locally.
- [ ] New behaviour has tests; no existing test was deleted or weakened for code I did not change.
- [ ] New formulations are registered (`@registry.register`), use physical units (hours, not time steps), and declare bounds and constraints.
- [ ] No study-specific code outside `sparsehydro/studies/`.
- [ ] No `print()`, notebooks, generated outputs, credentials, or files over 1 MiB are added.
- [ ] Public docstrings and `docs/` are updated.
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` (breaking changes marked **BREAKING**).

## Notes for reviewers

<!-- Anything you would like a reviewer to look at closely, open questions,
or follow-up work. Delete this section if not needed. -->
