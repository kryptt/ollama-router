//! Per-hop health: bench an alias-chain hop after a classified failure so
//! the requests behind it skip straight to a hop that can answer.
//!
//! A hop is `(backend, model)`. Its state machine:
//!
//! ```text
//! HEALTHY --failure(reason)--> BENCHED{until, strikes}
//! BENCHED --now >= until-----> one request is admitted as the probe; the
//!                              bench is extended by PROBE_LEASE meanwhile,
//!                              so concurrent requests keep skipping
//! probe success -------------> HEALTHY (entry removed, strikes reset)
//! probe failure -------------> BENCHED, strikes+1,
//!                              delay = min(base * 2^(strikes-1), cap)
//! failure while still benched -> no change
//! ```
//!
//! Only a probe (or a failure after the bench ran out) adds a strike. A
//! failure while the bench is live comes from a request admitted anyway (the
//! all-benched "try everyone" pass, or one already in flight) and says
//! nothing new; counting it would double the bench on every request.
//!
//! The lease doubles as the abandoned-probe guard: a probe whose client
//! hung up never reports an outcome, and the next request after the lease
//! simply becomes the new probe. No flag to get stuck.
//!
//! `auth` benches the whole backend (one credential per backend: every
//! model behind it fails the same way). Everything else benches one hop.
//!
//! In memory only: a restart costs one failure per broken hop.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::http::{HeaderMap, StatusCode};

/// How long a granted probe holds the bench before another request may
/// probe. Matches the router's default per-attempt request timeout, so a
/// probe that is merely slow is not raced by a second one.
const PROBE_LEASE: Duration = Duration::from_secs(600);

/// Why a hop was benched. Each reason has its own re-check delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// 402: the account behind the backend is out of credit.
    Payment,
    /// 401 / 403: the backend's credential is bad. Benches the backend.
    Auth,
    /// 404: the backend does not (or no longer) serve this model.
    ModelMissing,
    /// 429: rate limited. A `Retry-After` header overrides the delay.
    RateLimited,
    /// 5xx: upstream error, usually transient.
    Upstream5xx,
    /// No response at all: connect, timeout or transport failure.
    Transport,
}

impl Reason {
    pub const ALL: [Reason; 6] = [
        Reason::Payment,
        Reason::Auth,
        Reason::ModelMissing,
        Reason::RateLimited,
        Reason::Upstream5xx,
        Reason::Transport,
    ];

    /// Stable label for metrics, logs and the `[health]` config table.
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Payment => "payment",
            Reason::Auth => "auth",
            Reason::ModelMissing => "model_missing",
            Reason::RateLimited => "rate_limited",
            Reason::Upstream5xx => "upstream_5xx",
            Reason::Transport => "transport",
        }
    }

    pub fn from_name(name: &str) -> Option<Reason> {
        Reason::ALL.into_iter().find(|r| r.as_str() == name)
    }

    /// The bench reason for an upstream status, or `None` when the status
    /// commits (2xx, and 4xx that are the client's own fault).
    pub fn from_status(status: StatusCode) -> Option<Reason> {
        match status {
            StatusCode::PAYMENT_REQUIRED => Some(Reason::Payment),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Some(Reason::Auth),
            StatusCode::NOT_FOUND => Some(Reason::ModelMissing),
            StatusCode::TOO_MANY_REQUESTS => Some(Reason::RateLimited),
            s if s.is_server_error() => Some(Reason::Upstream5xx),
            _ => None,
        }
    }

    fn backend_wide(self) -> bool {
        matches!(self, Reason::Auth)
    }
}

/// Re-check delay for one reason: the first bench lasts `base`, each
/// consecutive failure doubles it, never past `cap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delay {
    pub base: Duration,
    pub cap: Duration,
}

impl Delay {
    const fn secs(base: u64, cap: u64) -> Delay {
        Delay {
            base: Duration::from_secs(base),
            cap: Duration::from_secs(cap),
        }
    }

    fn for_strike(self, strikes: u32) -> Duration {
        let factor = 1u32 << strikes.saturating_sub(1).min(20);
        self.base.saturating_mul(factor).min(self.cap)
    }
}

/// The delay table. Defaults below; `[health.<reason>]` in router.toml
/// overrides any of them (live-reloaded with the rest of the file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delays(HashMap<Reason, Delay>);

impl Default for Delays {
    fn default() -> Self {
        Delays(HashMap::from([
            (Reason::Payment, Delay::secs(15 * 60, 24 * 3600)),
            (Reason::Auth, Delay::secs(5 * 60, 12 * 3600)),
            (Reason::ModelMissing, Delay::secs(10 * 60, 12 * 3600)),
            (Reason::RateLimited, Delay::secs(30, 3600)),
            (Reason::Upstream5xx, Delay::secs(15, 3600)),
            (Reason::Transport, Delay::secs(60, 2 * 3600)),
        ]))
    }
}

impl Delays {
    pub fn get(&self, reason: Reason) -> Delay {
        // Every reason is seeded by Default and overrides only replace.
        self.0
            .get(&reason)
            .copied()
            .unwrap_or_else(|| Delay::secs(60, 3600))
    }

    pub fn set(&mut self, reason: Reason, delay: Delay) {
        self.0.insert(reason, delay);
    }
}

/// Seconds from a `Retry-After` header. The HTTP-date form is ignored
/// (rare from LLM APIs); the reason's own delay applies instead.
pub fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let secs: u64 = headers
        .get(axum::http::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(secs))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    backend: String,
    /// `None` = the whole backend.
    model: Option<String>,
}

impl Key {
    fn hop(backend: &str, model: &str) -> Key {
        Key {
            backend: backend.to_string(),
            model: Some(model.to_string()),
        }
    }
    fn backend(backend: &str) -> Key {
        Key {
            backend: backend.to_string(),
            model: None,
        }
    }
}

#[derive(Debug, Clone)]
struct Bench {
    reason: Reason,
    until: Instant,
    strikes: u32,
    /// A probe was admitted and `until` is its lease, not a real bench.
    probing: bool,
}

/// One benched entry, for `/status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchedHop {
    pub backend: String,
    /// `None` when the whole backend is benched.
    pub model: Option<String>,
    pub reason: Reason,
    pub strikes: u32,
    pub remaining: Duration,
}

/// Admission decision for one hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// Healthy, or this request is the probe.
    Go,
    /// Benched; skip it.
    Skip(Reason),
}

#[derive(Debug, Default)]
pub struct Health {
    benches: Mutex<HashMap<Key, Bench>>,
}

impl Health {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Bench>> {
        // A panic while holding this lock leaves only plain data behind,
        // and the worst a stale entry does is bench a hop a little longer.
        self.benches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Read-only: is this hop benched right now (an expired bench counts as
    /// healthy, since its next request would be admitted as the probe)?
    pub fn is_benched(&self, backend: &str, model: &str, now: Instant) -> bool {
        let map = self.lock();
        [Key::backend(backend), Key::hop(backend, model)]
            .iter()
            .any(|k| map.get(k).is_some_and(|b| now < b.until))
    }

    /// Admit or skip, granting the probe when a bench has expired.
    pub fn admit(&self, backend: &str, model: &str, now: Instant) -> Admit {
        let mut map = self.lock();
        let keys = [Key::backend(backend), Key::hop(backend, model)];
        // Decide first, then lease: leasing the backend's expired bench and
        // then skipping on the hop's live one would lock every other model
        // on the backend out for a whole lease with no probe in flight.
        if let Some(b) = keys
            .iter()
            .filter_map(|k| map.get(k))
            .find(|b| now < b.until)
        {
            return Admit::Skip(b.reason);
        }
        for key in &keys {
            if let Some(b) = map.get_mut(key) {
                b.until = now + PROBE_LEASE;
                b.probing = true;
            }
        }
        Admit::Go
    }

    /// Bench after a failure; returns how long.
    pub fn record_failure(
        &self,
        backend: &str,
        model: &str,
        reason: Reason,
        retry_after: Option<Duration>,
        delays: &Delays,
        now: Instant,
    ) -> Duration {
        let key = if reason.backend_wide() {
            Key::backend(backend)
        } else {
            Key::hop(backend, model)
        };
        let delay = delays.get(reason);
        let mut map = self.lock();
        if let Some(b) = map.get(&key)
            && now < b.until
            && !b.probing
        {
            return b.until - now;
        }
        let strikes = map.get(&key).map_or(0, |b| b.strikes) + 1;
        let wait = retry_after.map_or_else(|| delay.for_strike(strikes), |d| d.min(delay.cap));
        map.insert(
            key,
            Bench {
                reason,
                until: now + wait,
                strikes,
                probing: false,
            },
        );
        wait
    }

    /// The hop answered: clear its bench and its backend's.
    pub fn record_success(&self, backend: &str, model: &str) {
        let mut map = self.lock();
        map.remove(&Key::hop(backend, model));
        map.remove(&Key::backend(backend));
    }

    /// Every entry still benched (expired ones awaiting a probe excluded).
    pub fn benched(&self, now: Instant) -> Vec<BenchedHop> {
        let mut out: Vec<BenchedHop> = self
            .lock()
            .iter()
            .filter(|(_, b)| now < b.until)
            .map(|(k, b)| BenchedHop {
                backend: k.backend.clone(),
                model: k.model.clone(),
                reason: b.reason,
                strikes: b.strikes,
                remaining: b.until - now,
            })
            .collect();
        out.sort_by(|a, b| (&a.backend, &a.model).cmp(&(&b.backend, &b.model)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    #[test]
    fn status_classification() {
        assert_eq!(
            Reason::from_status(StatusCode::PAYMENT_REQUIRED),
            Some(Reason::Payment)
        );
        assert_eq!(
            Reason::from_status(StatusCode::FORBIDDEN),
            Some(Reason::Auth)
        );
        assert_eq!(
            Reason::from_status(StatusCode::BAD_GATEWAY),
            Some(Reason::Upstream5xx)
        );
        assert_eq!(Reason::from_status(StatusCode::BAD_REQUEST), None);
        assert_eq!(Reason::from_status(StatusCode::OK), None);
        for r in Reason::ALL {
            assert_eq!(Reason::from_name(r.as_str()), Some(r));
        }
    }

    #[test]
    fn bench_probe_escalate_recover() {
        let h = Health::default();
        let d = Delays::default();
        let t0 = Instant::now();
        assert_eq!(h.admit("nous", "m", t0), Admit::Go);

        let w1 = h.record_failure("nous", "m", Reason::Payment, None, &d, t0);
        assert_eq!(w1, 15 * 60 * S);
        assert_eq!(h.admit("nous", "m", t0 + S), Admit::Skip(Reason::Payment));
        assert_eq!(
            h.admit("nous", "other", t0 + S),
            Admit::Go,
            "402 is hop-scoped"
        );

        // Expired: first request probes, a concurrent one still skips.
        let t1 = t0 + w1;
        assert!(!h.is_benched("nous", "m", t1));
        assert_eq!(h.admit("nous", "m", t1), Admit::Go);
        assert_eq!(h.admit("nous", "m", t1), Admit::Skip(Reason::Payment));

        // Probe fails again: delay doubles.
        assert_eq!(
            h.record_failure("nous", "m", Reason::Payment, None, &d, t1),
            30 * 60 * S
        );

        h.record_success("nous", "m");
        assert_eq!(h.admit("nous", "m", t1), Admit::Go);
        assert!(h.benched(t1).is_empty());
    }

    #[test]
    fn failures_while_benched_do_not_escalate() {
        let h = Health::default();
        let d = Delays::default();
        let t = Instant::now();
        let w = h.record_failure("b", "m", Reason::Payment, None, &d, t);
        for i in 1..10 {
            // e.g. the all-benched pass hitting the hop on every request
            let left = h.record_failure("b", "m", Reason::Payment, None, &d, t + i * S);
            assert_eq!(left, w - i * S);
        }
        assert_eq!(h.benched(t)[0].strikes, 1);
    }

    #[test]
    fn cap_and_retry_after() {
        let h = Health::default();
        let d = Delays::default();
        let mut t = Instant::now();
        let mut last = Duration::ZERO;
        for _ in 0..30 {
            last = h.record_failure("b", "m", Reason::Upstream5xx, None, &d, t);
            t += last; // each failure lands after the previous bench ran out
        }
        assert_eq!(last, 3600 * S, "capped");
        assert_eq!(
            h.record_failure("b", "x", Reason::RateLimited, Some(7 * S), &d, t),
            7 * S
        );
        assert_eq!(
            h.record_failure("b", "y", Reason::RateLimited, Some(99_999 * S), &d, t),
            3600 * S,
            "Retry-After is capped too"
        );
    }

    #[test]
    fn auth_benches_whole_backend_and_success_clears_it() {
        let h = Health::default();
        let t = Instant::now();
        h.record_failure("nous", "a", Reason::Auth, None, &Delays::default(), t);
        assert_eq!(h.admit("nous", "b", t), Admit::Skip(Reason::Auth));
        assert_eq!(h.admit("freellmapi", "a", t), Admit::Go);
        h.record_success("nous", "b");
        assert_eq!(h.admit("nous", "a", t), Admit::Go);
    }

    #[test]
    fn retry_after_header() {
        let mut hm = HeaderMap::new();
        assert_eq!(retry_after(&hm), None);
        hm.insert(axum::http::header::RETRY_AFTER, "12".parse().unwrap());
        assert_eq!(retry_after(&hm), Some(12 * S));
        hm.insert(
            axum::http::header::RETRY_AFTER,
            "Wed, 21 Oct 2015 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(retry_after(&hm), None);
    }
}
