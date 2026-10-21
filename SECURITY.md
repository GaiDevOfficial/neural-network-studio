# Security Policy

## Supported versions

The latest published release receives security fixes. Older releases may be asked to upgrade.

## Reporting a vulnerability

Please do not disclose security issues in a public issue. Use GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Choose **Advisories** -> **Report a vulnerability**.
3. Include the affected version, a minimal reproduction, and the impact you observed.

If private advisories are unavailable, contact the maintainer through the GitHub profile
[@GaiDevOfficial](https://github.com/GaiDevOfficial) and ask for a private channel before sharing details.

You can expect an acknowledgement within seven days. Valid reports will be fixed in a patch release and
credited unless you ask to remain anonymous.

## Scope notes

- The desktop host binds to `127.0.0.1` and serves the bundled application only.
- The local Data API is read-only and intended for local datasets.
- Node Studio does not upload training data or models.
- The public build is unsigned. Verify the published SHA-256 before running a downloaded ZIP.
