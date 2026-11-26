"""OIDC bearer-token verification.

brag is a resource server: clients sign users in with your identity provider
(Entra ID, Okta, Google, Keycloak, ...) and send the access or ID token as
`Authorization: Bearer <token>`. brag checks the signature against the
provider's published keys, the issuer, the audience and the expiry, then reads
the user's groups from a claim.
"""

from dataclasses import dataclass
from typing import Any, Callable

import httpx
import jwt

ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class User:
    subject: str
    name: str
    groups: frozenset[str]


def claim(claims: dict[str, Any], path: str) -> Any:
    """Read a claim by dotted path, e.g. "realm_access.roles" (Keycloak)."""
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def groups_from(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset(value.split())
    return frozenset(str(v) for v in value)


class OIDCVerifier:
    def __init__(
        self,
        issuer: str,
        audience: str,
        groups_claim: str = "groups",
        jwks_url: str | None = None,
        key_resolver: Callable[[str], Any] | None = None,
        leeway: int = 60,
    ):
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.groups_claim = groups_claim
        self.leeway = leeway
        self._jwks_url = jwks_url
        self._key_resolver = key_resolver  # for tests: token -> signing key

    def _jwks_client(self) -> jwt.PyJWKClient:
        if not hasattr(self, "_client"):
            url = self._jwks_url
            if not url:
                discovery = httpx.get(f"{self.issuer}/.well-known/openid-configuration", timeout=10)
                discovery.raise_for_status()
                url = discovery.json()["jwks_uri"]
            self._client = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)
        return self._client

    def _signing_key(self, token: str) -> Any:
        if self._key_resolver:
            return self._key_resolver(token)
        return self._jwks_client().get_signing_key_from_jwt(token).key

    def verify(self, token: str) -> User:
        try:
            claims = jwt.decode(
                token,
                self._signing_key(token),
                algorithms=ALGORITHMS,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except (jwt.PyJWTError, httpx.HTTPError, KeyError) as e:
            raise AuthError(str(e)) from e
        name = claims.get("preferred_username") or claims.get("email") or claims.get("name") or claims["sub"]
        return User(str(claims["sub"]), str(name), groups_from(claim(claims, self.groups_claim)))
