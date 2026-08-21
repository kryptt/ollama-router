//! The routing-policy file (`OLLAMA_ROUTER_CONFIG`): backends, the spend
//! boundary, fallbacks, and alias chains, in one TOML document.
//!
//! Everything here is **live** — the discovery loop re-reads and re-validates
//! this file every cycle, backend roster included. Everything in the
//! environment is restart-scoped. That split is the organising rule; see
//! `README.md ## Configuration`.
//!
//! Validation is deliberately all-or-nothing ([`FileConfig::load`] either
//! yields a fully-checked [`Validated`] or nothing at all), because the
//! caller's fallback on error is "keep the previous config entirely". A
//! partial apply would let a reader observe new aliases against an old
//! roster — exactly the atomicity hole this file was created to close.
//!
//! The headline property of validating backends and aliases in one pass
//! against one roster: **every reload re-proves the privacy pin.** Flipping
//! `external = true` on a backend that a `local` alias chains through
//! rejects the whole file, rather than silently un-pinning it.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::Deserialize;

/// The wildcard `allow` entry: publish everything the backend advertises.
const ALLOW_ALL: &str = "*";

/// The raw TOML document, before validation.
///
/// `deny_unknown_fields` throughout: this is a hand-edited file on NFS with
/// no CI in front of it, so `stip_auth = true` must be a rejection rather
/// than a shrug. The cost — additive schema changes become breaking — is
/// accepted deliberately: one file, one operator.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    /// **Document order is routing priority.** An array of tables (not a
    /// map) precisely because `Registry::rebuild_model_map` is
    /// first-writer-wins, so declaration order decides whether a colliding
    /// model name spends metered credits or runs local. A map would lose
    /// the order; a `priority = N` field would be a second source of truth
    /// that can contradict the file's visual order.
    backends: Vec<RawBackend>,
    /// `local-model = "hosted-stand-in"`. Consulted only when no reachable
    /// backend serves the requested model.
    fallbacks: HashMap<String, String>,
    /// `[aliases.<name>]` chains. A map: order is meaningless between
    /// aliases, and TOML's native duplicate-key rejection replaces a
    /// hand-written check.
    aliases: HashMap<String, RawAlias>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBackend {
    name: String,
    url: String,
    /// **Required**, with `["*"]` meaning publish-all. Not optional: under
    /// the old flat-file grammar, deleting a backend's allow line was a
    /// *successful* parse that silently evaporated its spend boundary, so
    /// keep-previous never fired. Absence is now a rejection.
    allow: Vec<String>,
    /// Hosted / off-LAN. Consumed entirely by validation (it is what a
    /// `local` alias is checked against) and never needed at request time,
    /// so it does not survive into [`BackendSpec`].
    #[serde(default)]
    external: bool,
    /// Drop the client's `authorization` header before forwarding.
    #[serde(default)]
    strip_auth: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAlias {
    /// Privacy pin: reject the file if any candidate's backend is
    /// `external`.
    #[serde(default)]
    local: bool,
    chain: Vec<String>,
}

/// One `backend/model` candidate in an alias chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasCandidate {
    pub backend: String,
    pub model: String,
}

/// A named priority chain of concrete `backend/model` candidates. Requests
/// for the alias name walk the chain in order and commit to the first
/// candidate that answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alias {
    /// Privacy pin: a `local` alias is rejected at validation time if any
    /// candidate's backend is marked `external`, so by construction its
    /// traffic can never leave the local backends.
    pub local_only: bool,
    /// Invariant: non-empty, every candidate's backend is a configured
    /// backend name.
    pub candidates: Vec<AliasCandidate>,
}

/// A validated backend declaration. Position in [`Validated::backends`] is
/// routing priority.
///
/// Invariants: `name` non-empty and unique, `url` is `http(s)://…` with no
/// trailing slash, `allow_models` is either `None` or a non-empty set.
///
/// `non_exhaustive` so that only this module can build one: fields are `pub`
/// for readers, but a struct literal outside the crate is refused, keeping
/// [`FileConfig::parse`] the sole constructor and the invariants above true
/// by construction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackendSpec {
    pub name: String,
    pub url: String,
    /// Discovery allowlist. `None` = publish every model the backend
    /// advertises (`allow = ["*"]`); `Some(set)` = keep only these exact
    /// names.
    ///
    /// For a metered backend this is the spend boundary: anything that can
    /// reach the router can request any model the router publishes.
    pub allow_models: Option<HashSet<String>>,
    pub strip_auth: bool,
}

/// A fully validated policy document. The only way to obtain one is
/// [`FileConfig::load`] (or [`FileConfig::parse`]), so holding one is proof
/// that every cross-reference in it checked out.
/// `non_exhaustive` for the same reason as [`BackendSpec`]: readable
/// everywhere, constructible only here.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Validated {
    pub backends: Vec<BackendSpec>,
    pub fallbacks: HashMap<String, String>,
    pub aliases: HashMap<String, Alias>,
}

#[derive(Debug)]
pub enum PolicyError {
    Read {
        path: String,
        source: std::io::Error,
    },
    /// The read did not finish in time. Its own variant because the cause is
    /// categorically different from every other error here: the file is on
    /// NFS, and a hard mount against an unreachable server blocks forever
    /// rather than returning an error.
    ReadTimeout {
        path: String,
        after: std::time::Duration,
    },
    /// Syntax error, unknown field, or type mismatch. `toml`'s own message
    /// already carries the line/column and a source excerpt.
    Parse(toml::de::Error),
    /// Parsed, but failed a semantic or cross-reference check.
    Invalid(String),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "router config: could not read '{path}': {source}")
            }
            Self::ReadTimeout { path, after } => write!(
                f,
                "router config: reading '{path}' timed out after {after:?} \
                 (is the filesystem hung?)"
            ),
            Self::Parse(e) => write!(f, "router config: {e}"),
            Self::Invalid(reason) => write!(f, "router config: {reason}"),
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse(e) => Some(e),
            Self::ReadTimeout { .. } | Self::Invalid(_) => None,
        }
    }
}

fn invalid<T>(reason: impl Into<String>) -> Result<T, PolicyError> {
    Err(PolicyError::Invalid(reason.into()))
}

impl FileConfig {
    /// Read and validate the policy file. Errors are total: nothing is
    /// applied, so a caller holding a previous [`Validated`] keeps it.
    pub fn load(path: &str) -> Result<Validated, PolicyError> {
        let raw = std::fs::read_to_string(path).map_err(|source| PolicyError::Read {
            path: path.to_string(),
            source,
        })?;
        Self::parse(&raw)
    }

    /// Parse and validate an in-memory document. Split out from [`load`] so
    /// tests (and the golden fixture) exercise the exact same path without
    /// touching the filesystem.
    ///
    /// [`load`]: FileConfig::load
    pub fn parse(raw: &str) -> Result<Validated, PolicyError> {
        let file: FileConfig = toml::from_str(raw).map_err(PolicyError::Parse)?;
        file.validate()
    }

    fn validate(self) -> Result<Validated, PolicyError> {
        let FileConfig {
            backends,
            fallbacks,
            aliases,
        } = self;

        if backends.is_empty() {
            return invalid("`backends` must contain at least one entry");
        }

        let mut specs: Vec<BackendSpec> = Vec::with_capacity(backends.len());
        let mut external: HashSet<String> = HashSet::new();
        for raw in backends {
            let name = raw.name.trim().to_string();
            if name.is_empty() {
                return invalid("a backend has an empty `name`");
            }
            if specs.iter().any(|s| s.name == name) {
                return invalid(format!(
                    "duplicate backend name '{name}'; names are the routing key"
                ));
            }
            let url = validate_url(&name, &raw.url)?;
            let allow_models = validate_allow(&name, &raw.allow)?;
            if raw.external {
                external.insert(name.clone());
            }
            specs.push(BackendSpec {
                name,
                url,
                allow_models,
                strip_auth: raw.strip_auth,
            });
        }

        // Sorted iteration over both maps so that a file with several
        // problems always reports the same one — a nondeterministic error
        // message on a hand-edited file is a bad debugging experience.
        let mut fallback_keys: Vec<&String> = fallbacks.keys().collect();
        fallback_keys.sort();
        for from in fallback_keys {
            // `to` is deliberately NOT checked against the published
            // catalogue: its backend may simply be down right now, and it
            // is unprovable at all against an `allow = ["*"]` backend.
            let to = fallbacks.get(from).map(String::as_str).unwrap_or_default();
            if from.trim().is_empty() || to.trim().is_empty() {
                return invalid(format!("fallback '{from}' = '{to}' has an empty side"));
            }
            if from == to {
                return invalid(format!(
                    "fallback '{from}' maps to itself; the hop would be a no-op"
                ));
            }
        }

        let mut alias_names: Vec<&String> = aliases.keys().collect();
        alias_names.sort();
        let mut validated_aliases = HashMap::with_capacity(aliases.len());
        for name in alias_names {
            let Some(raw) = aliases.get(name) else {
                continue;
            };
            let alias = validate_alias(name, raw, &specs, &external)?;
            validated_aliases.insert(name.clone(), alias);
        }

        Ok(Validated {
            backends: specs,
            fallbacks,
            aliases: validated_aliases,
        })
    }
}

/// Check the scheme *and the host*, then strip any trailing slash so `url`
/// concatenates cleanly with the `/api/tags`-style suffixes discovery
/// appends.
///
/// The host check is not pedantry. `http:///llama-swap.ai:8080` (a
/// triple-slash typo) and `http://:8080` are both scheme-correct and
/// host-less: they would be *accepted*, counted as an applied reload, and
/// then fail every probe — so the backend's models would quietly drop out
/// after the grace period instead of the file being rejected outright, with
/// no alert anywhere. Hand-rolled rather than pulling in the `url` crate:
/// this is a scheme plus a non-empty authority, and the dependency would
/// cost more binary than the check is worth.
fn validate_url(name: &str, raw: &str) -> Result<String, PolicyError> {
    let reject = || {
        invalid(format!(
            "backend '{name}': `url` must be http(s)://host[:port], got '{raw}'"
        ))
    };
    let url = raw.trim();
    let Some(rest) = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
    else {
        return reject();
    };
    // Authority is everything before the first '/', hostname everything
    // before the first ':' in that. Both must be non-empty: an empty
    // authority is the triple-slash typo, an empty hostname is ":8080".
    let authority = rest.split('/').next().unwrap_or(rest);
    let hostname = authority.split(':').next().unwrap_or(authority);
    if hostname.is_empty() {
        return reject();
    }
    Ok(url.trim_end_matches('/').to_string())
}

/// Lower an `allow` list to the registry's filter representation.
/// `["*"]` — and only a lone `["*"]` — means "publish everything".
fn validate_allow(name: &str, allow: &[String]) -> Result<Option<HashSet<String>>, PolicyError> {
    if allow.is_empty() {
        return invalid(format!(
            "backend '{name}': `allow` is empty, which would publish nothing and be \
             indistinguishable at runtime from the backend being down; use \
             allow = [\"*\"] to publish everything"
        ));
    }
    if allow.iter().any(|m| m.trim() == ALLOW_ALL) {
        if allow.len() > 1 {
            // "*" plus explicit ids reads as "these, but really everything".
            // Whichever the operator meant, one of the two is wrong.
            return invalid(format!(
                "backend '{name}': `allow` mixes \"*\" with explicit model ids; \
                 use either the wildcard alone or an explicit list"
            ));
        }
        return Ok(None);
    }
    let mut set = HashSet::with_capacity(allow.len());
    for model in allow {
        let model = model.trim();
        if model.is_empty() {
            return invalid(format!("backend '{name}': `allow` contains an empty entry"));
        }
        set.insert(model.to_string());
    }
    Ok(Some(set))
}

fn validate_alias(
    name: &str,
    raw: &RawAlias,
    backends: &[BackendSpec],
    external: &HashSet<String>,
) -> Result<Alias, PolicyError> {
    if name.trim().is_empty() {
        return invalid("an alias has an empty name");
    }
    // Carried over from the flat-file grammar, where `local` was a prefix
    // keyword: an alias by that name reads as a privacy marker at every
    // call site that mentions it.
    if name == "local" {
        return invalid("alias may not be named 'local' (reserved keyword)");
    }
    if raw.chain.is_empty() {
        return invalid(format!("alias '{name}': `chain` is empty"));
    }

    let mut candidates = Vec::with_capacity(raw.chain.len());
    for entry in &raw.chain {
        let entry = entry.trim();
        // Split on the FIRST '/' only: model ids routinely contain '/' and
        // ':' themselves (`freellmapi/groq/llama-3.3-70b` is backend
        // `freellmapi`, model `groq/llama-3.3-70b`).
        let Some((backend, model)) = entry.split_once('/') else {
            return invalid(format!(
                "alias '{name}': candidate '{entry}' is not backend/model"
            ));
        };
        let (backend, model) = (backend.trim(), model.trim());
        if backend.is_empty() || model.is_empty() {
            return invalid(format!(
                "alias '{name}': candidate '{entry}' has an empty backend or model"
            ));
        }
        // Covers removing a backend that an alias still needs: the removal
        // and the alias edit have to land in the same save, which one file
        // makes atomic.
        if !backends.iter().any(|b| b.name == backend) {
            return invalid(format!(
                "alias '{name}' names backend '{backend}', which is not declared in [[backends]]"
            ));
        }
        // NOT checked: whether `model` is in that backend's allow list. A
        // chain skips non-serving candidates by design, and placeholder
        // candidates for a not-yet-provisioned provider are deliberate.
        candidates.push(AliasCandidate {
            backend: backend.to_string(),
            model: model.to_string(),
        });
    }

    if raw.local
        && let Some(c) = candidates.iter().find(|c| external.contains(&c.backend))
    {
        return invalid(format!(
            "alias '{name}' is marked local but candidate backend '{}' is external",
            c.backend
        ));
    }

    Ok(Alias {
        local_only: raw.local,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest valid document: one backend, both maps present but
    /// empty. `[fallbacks]` and `[aliases]` are required tables — the only
    /// optional fields in the schema are the three whose absence cannot
    /// widen anything (`external`, `strip_auth`, alias `local`).
    const MINIMAL: &str = r#"
        [[backends]]
        name = "a"
        url = "http://a:1"
        allow = ["*"]
        [fallbacks]
        [aliases]
    "#;

    fn parse(raw: &str) -> Validated {
        FileConfig::parse(raw).expect("expected a valid policy document")
    }

    #[track_caller]
    fn assert_rejects(raw: &str, expected_substring: &str) {
        let err = FileConfig::parse(raw).expect_err("expected a rejection");
        assert!(
            err.to_string().contains(expected_substring),
            "expected error containing {expected_substring:?}, got: {err}"
        );
    }

    /// A two-backend roster (one local, one external) plus both maps, with
    /// `extra` appended — the substrate for the alias/validation tests.
    fn with_roster(extra: &str) -> String {
        format!(
            r#"
            [[backends]]
            name = "swap"
            url = "http://s:1"
            allow = ["*"]

            [[backends]]
            name = "nous"
            url = "http://n:2"
            external = true
            allow = ["m"]

            [fallbacks]

            {extra}
            "#
        )
    }

    fn aliases(extra: &str) -> String {
        format!("{}\n[aliases]\n{extra}", with_roster(""))
    }

    // ── the golden fixture: a byte copy of the live file ─────────────────
    //
    // This doubles as migration verification. If it stops matching
    // /swarm/main/ai/ollama-router/router.toml, one of the two is stale.

    #[test]
    fn golden_fixture_parses_to_the_live_policy() {
        let v = parse(include_str!("../tests/fixtures/router.toml"));

        // Order is the security boundary: local backends first, metered
        // last. Assert the exact sequence, not just membership.
        let names: Vec<&str> = v.backends.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["llama-swap-cuda", "llama-swap", "nous", "freellmapi"]
        );

        // Both llama-swaps and freellmapi publish everything...
        for name in ["llama-swap-cuda", "llama-swap", "freellmapi"] {
            let b = v
                .backends
                .iter()
                .find(|b| b.name == name)
                .expect("backend present");
            assert!(b.allow_models.is_none(), "{name} should publish everything");
            assert!(!b.strip_auth, "{name} must forward auth");
        }

        // ...nous is the spend boundary.
        let nous = v
            .backends
            .iter()
            .find(|b| b.name == "nous")
            .expect("nous present");
        let allow = nous.allow_models.as_ref().expect("nous is allowlisted");
        assert_eq!(allow.len(), 13);
        assert!(allow.contains("nousresearch/hermes-4-405b"));
        assert!(allow.contains("z-ai/glm-4.6v"));
        assert!(
            !allow.contains("*"),
            "the wildcard must not survive as an id"
        );
        assert!(nous.strip_auth);

        assert_eq!(v.fallbacks.len(), 14);
        assert_eq!(
            v.fallbacks.get("gemma4:e2b").map(String::as_str),
            Some("google/gemma-4-26b-a4b-it")
        );

        assert_eq!(v.aliases.len(), 6);
        let mut local: Vec<&str> = v
            .aliases
            .iter()
            .filter(|(_, a)| a.local_only)
            .map(|(n, _)| n.as_str())
            .collect();
        local.sort();
        assert_eq!(local, vec!["financial", "memory"]);

        // The vision chain exercises the first-'/' split on a model id that
        // contains both '/' and ':'.
        let vision = v.aliases.get("vision").expect("vision alias");
        assert_eq!(vision.candidates[0].backend, "llama-swap-cuda");
        assert_eq!(
            vision.candidates[0].model,
            "haervwe/GLM-4.6V-Flash-9B:latest"
        );
        assert_eq!(vision.candidates[1].backend, "nous");
        assert_eq!(vision.candidates[1].model, "z-ai/glm-4.6v");
    }

    // ── schema ───────────────────────────────────────────────────────────

    #[test]
    fn minimal_document_parses() {
        let v = parse(MINIMAL);
        assert_eq!(v.backends.len(), 1);
        assert!(v.fallbacks.is_empty());
        assert!(v.aliases.is_empty());
        assert!(!v.backends[0].strip_auth);
    }

    #[test]
    fn backend_order_is_preserved() {
        let v = parse(
            r#"
            [[backends]]
            name = "third"
            url = "http://c:3"
            allow = ["*"]
            [[backends]]
            name = "first"
            url = "http://a:1"
            allow = ["*"]
            [[backends]]
            name = "second"
            url = "http://b:2"
            allow = ["*"]
            [fallbacks]
            [aliases]
            "#,
        );
        let names: Vec<&str> = v.backends.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["third", "first", "second"]);
    }

    #[test]
    fn unknown_field_is_rejected() {
        // The motivating typo: `stip_auth = true` silently no-ops under a
        // permissive schema and leaks the inbound token.
        assert_rejects(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*"]
            stip_auth = true
            [fallbacks]
            [aliases]
            "#,
            "stip_auth",
        );
        assert_rejects(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*"]
            [fallbacks]
            [aliases]
            [aliases.x]
            chian = ["a/m"]
            "#,
            "chian",
        );
        assert_rejects(&format!("{MINIMAL}\n[extras]\nk = 1\n"), "extras");
    }

    #[test]
    fn syntax_error_is_rejected() {
        assert_rejects("[[backends]\nname = \"a\"\n", "expected");
    }

    #[test]
    fn absent_backends_or_maps_are_rejected() {
        assert_rejects("[fallbacks]\n[aliases]\n", "backends");
        assert_rejects("backends = []\n[fallbacks]\n[aliases]\n", "at least one");
        assert_rejects(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"*\"]\n[aliases]\n",
            "fallbacks",
        );
        assert_rejects(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"*\"]\n[fallbacks]\n",
            "aliases",
        );
    }

    // ── the spend boundary ───────────────────────────────────────────────

    #[test]
    fn absent_allow_is_rejected() {
        // The bug this schema exists to kill: under the old grammar,
        // deleting a backend's allow line was a *successful* parse that
        // silently removed its spend boundary.
        assert_rejects(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\n[fallbacks]\n[aliases]\n",
            "allow",
        );
    }

    #[test]
    fn empty_allow_is_rejected() {
        assert_rejects(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = []\n[fallbacks]\n[aliases]\n",
            "publish nothing",
        );
    }

    #[test]
    fn wildcard_mixed_with_ids_is_rejected() {
        assert_rejects(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*", "m"]
            [fallbacks]
            [aliases]
            "#,
            "mixes",
        );
    }

    #[test]
    fn wildcard_lowers_to_no_filter_and_ids_lower_to_a_set() {
        let v = parse(
            r#"
            [[backends]]
            name = "open"
            url = "http://a:1"
            allow = ["*"]
            [[backends]]
            name = "metered"
            url = "http://b:2"
            allow = ["x/one", "x/two"]
            [fallbacks]
            [aliases]
            "#,
        );
        assert!(v.backends[0].allow_models.is_none());
        assert_eq!(
            v.backends[1].allow_models.as_ref().map(HashSet::len),
            Some(2)
        );
    }

    // ── backend identity ─────────────────────────────────────────────────

    #[test]
    fn duplicate_backend_name_is_rejected() {
        assert_rejects(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*"]
            [[backends]]
            name = "a"
            url = "http://b:2"
            allow = ["*"]
            [fallbacks]
            [aliases]
            "#,
            "duplicate backend name",
        );
    }

    #[test]
    fn bad_name_or_url_is_rejected() {
        let doc = |name: &str, url: &str| {
            format!(
                "[[backends]]\nname = \"{name}\"\nurl = \"{url}\"\nallow = [\"*\"]\n\
                 [fallbacks]\n[aliases]\n"
            )
        };
        assert_rejects(&doc("", "http://a:1"), "empty `name`");
        assert_rejects(&doc("a", ""), "http(s)://");
        assert_rejects(&doc("a", "ftp://a:1"), "http(s)://");
        assert_rejects(&doc("a", "a:1"), "http(s)://");
        assert_rejects(&doc("a", "http://"), "http(s)://");
    }

    #[test]
    fn hostless_url_is_rejected() {
        // A triple-slash typo is scheme-correct but host-less. Accepted, it
        // would count as an applied reload and then fail every probe, so the
        // backend's models drop out after the grace period with no alert —
        // the exact opposite of the wholesale rejection promised above.
        let doc = |url: &str| {
            format!(
                "[[backends]]\nname = \"a\"\nurl = \"{url}\"\nallow = [\"*\"]\n\
                 [fallbacks]\n[aliases]\n"
            )
        };
        for url in [
            "http:///llama-swap.ai:8080",
            "https:///a",
            "http://:8080",
            "http:///",
        ] {
            assert_rejects(&doc(url), "http(s)://");
        }
        // ...while real hosts, with and without port or path, still pass.
        for url in [
            "http://a",
            "http://a:1",
            "https://inference-api.nousresearch.com",
            "http://a:1/v1",
        ] {
            assert!(
                FileConfig::parse(&doc(url)).is_ok(),
                "{url} should be accepted"
            );
        }
    }

    #[test]
    fn trailing_slash_is_stripped() {
        let v = parse(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1/\"\nallow = [\"*\"]\n\
             [fallbacks]\n[aliases]\n",
        );
        assert_eq!(v.backends[0].url, "http://a:1");
    }

    // ── fallbacks, and the dotted-key trap ───────────────────────────────

    #[test]
    fn fallbacks_parse_with_quoted_keys() {
        let v = parse(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*"]
            [fallbacks]
            "qwen3.6-low" = "qwen/qwen3.8-27b"
            "gemma4:26b" = "google/gemma-4-26b-a4b-it"
            [aliases]
            "#,
        );
        assert_eq!(
            v.fallbacks.get("qwen3.6-low").map(String::as_str),
            Some("qwen/qwen3.8-27b")
        );
        assert_eq!(v.fallbacks.len(), 2);
    }

    #[test]
    fn unquoted_dotted_fallback_key_is_rejected_loudly() {
        // Every model name contains a '.', and a bare `qwen3.6-low = "x"` is
        // a valid TOML *dotted key* parsing as `{qwen3 = {"6-low" = "x"}}`.
        // The target type is HashMap<String, String>, so the nested table
        // fails deserialisation — never a silent mis-key.
        assert_rejects(
            r#"
            [[backends]]
            name = "a"
            url = "http://a:1"
            allow = ["*"]
            [fallbacks]
            qwen3.6-low = "qwen/qwen3.8-27b"
            [aliases]
            "#,
            "string",
        );
    }

    #[test]
    fn unquoted_key_with_colon_or_slash_is_a_syntax_error() {
        for key in ["gemma4:26b", "haervwe/GLM-4.6V"] {
            assert_rejects(
                &format!(
                    "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"*\"]\n\
                     [fallbacks]\n{key} = \"x\"\n[aliases]\n"
                ),
                "expected",
            );
        }
    }

    #[test]
    fn fallback_self_map_or_empty_side_is_rejected() {
        let doc = |body: &str| {
            format!(
                "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"*\"]\n\
                 [fallbacks]\n{body}\n[aliases]\n"
            )
        };
        assert_rejects(&doc(r#""m" = "m""#), "maps to itself");
        assert_rejects(&doc(r#""m" = """#), "empty side");
        assert_rejects(&doc(r#""" = "m""#), "empty side");
    }

    #[test]
    fn duplicate_fallback_key_is_rejected_by_toml() {
        assert_rejects(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"*\"]\n\
             [fallbacks]\n\"m\" = \"x\"\n\"m\" = \"y\"\n[aliases]\n",
            "duplicate",
        );
    }

    #[test]
    fn fallback_target_need_not_be_published() {
        // Its backend may simply be down, and it is unprovable at all
        // against an `allow = ["*"]` backend — so this must be accepted.
        let v = parse(
            "[[backends]]\nname = \"a\"\nurl = \"http://a:1\"\nallow = [\"m\"]\n\
             [fallbacks]\n\"local\" = \"never-published\"\n[aliases]\n",
        );
        assert_eq!(v.fallbacks.len(), 1);
    }

    // ── aliases ──────────────────────────────────────────────────────────

    #[test]
    fn alias_chain_splits_on_first_slash() {
        let v = parse(&aliases(
            "[aliases.fast]\nchain = [\"swap/qwen3.6:latest\", \"nous/groq/llama-3.3-70b\"]\n",
        ));
        let fast = v.aliases.get("fast").expect("alias present");
        assert!(!fast.local_only);
        assert_eq!(fast.candidates[0].backend, "swap");
        assert_eq!(fast.candidates[0].model, "qwen3.6:latest");
        assert_eq!(fast.candidates[1].backend, "nous");
        assert_eq!(fast.candidates[1].model, "groq/llama-3.3-70b");
    }

    #[test]
    fn local_alias_through_external_backend_is_rejected() {
        assert_rejects(
            &aliases("[aliases.secret]\nlocal = true\nchain = [\"swap/m\", \"nous/m\"]\n"),
            "marked local but candidate backend 'nous' is external",
        );
    }

    #[test]
    fn flipping_a_chained_backend_to_external_rejects_the_whole_file() {
        // THE headline property. Same alias, same chain — only the
        // backend's `external` flag moves, and the file stops validating.
        let doc = |external: bool| {
            format!(
                r#"
                [[backends]]
                name = "swap"
                url = "http://s:1"
                external = {external}
                allow = ["*"]
                [fallbacks]
                [aliases.financial]
                local = true
                chain = ["swap/m"]
                "#
            )
        };
        assert!(FileConfig::parse(&doc(false)).is_ok());
        assert_rejects(&doc(true), "marked local");
    }

    #[test]
    fn alias_naming_an_unconfigured_backend_is_rejected() {
        // Also the "removed a backend an alias still needs" case: with one
        // file, the removal and the alias edit must land in the same save.
        assert_rejects(
            &aliases("[aliases.fast]\nchain = [\"typo/m\"]\n"),
            "backend 'typo'",
        );
    }

    #[test]
    fn alias_named_local_is_rejected() {
        assert_rejects(
            &aliases("[aliases.local]\nchain = [\"swap/m\"]\n"),
            "may not be named 'local'",
        );
    }

    #[test]
    fn malformed_chain_is_rejected() {
        assert_rejects(&aliases("[aliases.f]\nchain = []\n"), "`chain` is empty");
        assert_rejects(
            &aliases("[aliases.f]\nchain = [\"noslash\"]\n"),
            "backend/model",
        );
        assert_rejects(
            &aliases("[aliases.f]\nchain = [\"swap/\"]\n"),
            "empty backend or model",
        );
        assert_rejects(
            &aliases("[aliases.f]\nchain = [\"/m\"]\n"),
            "empty backend or model",
        );
    }

    #[test]
    fn alias_candidate_need_not_be_in_the_backends_allow_list() {
        // `freellmapi/auto:fast` is a deliberate placeholder until the
        // provider keys land; chains skip non-serving candidates by design.
        let v = parse(&aliases(
            "[aliases.chat]\nchain = [\"nous/not-in-allow\", \"swap/m\"]\n",
        ));
        assert_eq!(v.aliases.get("chat").map(|a| a.candidates.len()), Some(2));
    }

    #[test]
    fn duplicate_alias_is_rejected_by_toml() {
        assert_rejects(
            &aliases("[aliases.f]\nchain = [\"swap/a\"]\n[aliases.f]\nchain = [\"swap/b\"]\n"),
            "duplicate",
        );
    }

    #[test]
    fn missing_file_is_a_read_error() {
        let err = FileConfig::load("/nonexistent/router.toml").expect_err("expected a read error");
        assert!(err.to_string().contains("could not read"), "{err}");
    }
}
