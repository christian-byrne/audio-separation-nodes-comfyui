# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly:

1. **Do not** open a public GitHub issue.
2. Email **christian.byrne@comfy.org** with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. You will receive a response within 7 days.

## Scope

This project is a set of ComfyUI custom nodes for audio separation. Security
concerns most likely involve:

- Arbitrary code execution via crafted audio files
- Path traversal in file handling
- Denial of service through resource exhaustion

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | ✅         |
| < 2.0   | ❌         |
