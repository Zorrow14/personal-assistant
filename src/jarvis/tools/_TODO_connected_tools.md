# TODO(phase-5.5): account-connected tools (Gmail, Google Calendar, ...)

**Not implemented.** This is the plan. Connected tools act on the user's online
accounts, so they need OAuth, stored secrets and much stricter safety than the
local tools.

## Approach

1. **Auth.** Use Google's installed-app OAuth flow, run from a one-off
   `python -m jarvis.cli --connect google` command, never from inside a chat
   turn. Request the smallest scopes that work, starting read-only (for example
   `gmail.readonly` and `calendar.readonly`). Add write scopes only per tool,
   when that tool ships.
2. **Secrets.** Keep refresh tokens in the OS keyring (Windows Credential
   Manager), not in `.env` and not in the vault. Never log them. Add a
   credentials provider to `ToolContext` so tools ask for a client and never
   see raw tokens.
3. **Opt-in.** Connected tools stay out of `--list-tools` until the account is
   connected. If connection fails, `from_context` raises and discovery skips the
   tool.
4. **Safety tiers.**
   - Reads (list events, search mail): no confirmation, but results are
     summarised, not dumped wholesale into the LLM context.
   - Writes to the user's own data (create a calendar event, draft an email):
     `requires_confirmation = True`.
   - Sends or acts on other people (send an email, invite attendees): confirm
     with the full recipient list and content shown. These may also need a
     second typed confirmation.
   - Delete or spend: not in this phase.
5. **Isolation.** One module per service (`gmail_tools.py`,
   `calendar_tools.py`). The Google SDK is imported only there, and API errors
   become short messages for the LLM.
6. **Tests.** Mock the service client entirely. No test touches a real account.

## Not planned

Any tool that deletes user data, spends money, or runs arbitrary shell or code.
