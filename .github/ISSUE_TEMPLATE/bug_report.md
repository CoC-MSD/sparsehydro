---
name: Bug report
about: Something does not work as documented
title: "[BUG] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what is wrong.

**To reproduce**
A minimal script that shows the problem. Use synthetic data or the bundled
sample (`from sparsehydro.sample_data import load_sample`) where possible, and
remove anything not needed to trigger the bug.

```python
# paste the script here
```

**What happened**
The full error message and traceback, or the incorrect result.

**What you expected**
What you expected to happen instead.

**Environment**
- sparsehydro version (`python -c "import sparsehydro; print(sparsehydro.__version__)"`):
- Python version:
- Operating system:
- Installed extras (for example `speed`, `parallel`):

**Additional context**
Anything else that might help, such as whether it worked in an earlier version.
Please do not attach confidential monitoring data.
