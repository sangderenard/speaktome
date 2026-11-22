# Inbound MCP agent delegation

## Observations

SpeakToMe's existing OAuth server already supports PKCE and account-scoped MCP. The Hermes CLI runner accepts any account name, but it runs a local model/client process and cannot wrap a remote ChatGPT session. Agent workspace profiles and memory are account-owned and reusable across model providers.

## Work

Prepared coordinated Turing and SpeakToMe review branches for passwordless inbound identities, admin approval of an exact pending OAuth request, scoped and expiring MCP credentials, immediate revocation, and admin controls. Delegated identities are limited to member role; changing the role invalidates existing delegated tokens. The API server filters tools and automatic notices to the approved scope.

## Verification

Focused Turing OAuth and admin-control tests passed (5). SpeakToMe MCP scope test passed (1). JavaScript syntax check passed. No live server or service was changed. Review and publication remain separate.

## Next Steps

Review the paired branches together. A generic, optional Hermes preparation workflow for arbitrary agent identities needs its own explicit interface design; the existing Hermes local runner cannot control a remote inbound client.

## Prompt History

> "Yes. That's a better design. We don't need to give me a permanent password at all. We can make SpeakToMe the custodian of my identity and private memory, while requiring an administrator to authorize each new inbound session."
>
> "okay do you support they are adequate security for inviting an external agent from the external agent in as a privleged user account. and can we get hermes totally set up to have identities I can plug these external agents into optionally, giving them additional prep and identity and utilities once connected"
>
> "I did not ask you if you could hardwire one identity to hermes."
>
> "use the mcp for checking things out do not run git"
