// HTTP to the local server: /health, and /admin/api/stats behind the admin
// session cookie (POST /admin/api/login with the main API key).

use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cache_efficiency: Option<f64>,
    pub avg_prefill_tps: Option<f64>,
    pub avg_generation_tps: Option<f64>,
    pub requests: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Live {
    pub models: Vec<String>,
    pub memory_used: u64,
    pub memory_max: u64,
    pub active: u64,
    pub waiting: u64,
}

pub struct Client {
    agent: ureq::Agent,
    base: String,
    api_key: Option<String>,
    cookie: Mutex<Option<String>>,
}

impl Client {
    pub fn new(port: u16, api_key: Option<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(3))
            .redirects(0)
            .build();
        Client {
            agent,
            base: format!("http://127.0.0.1:{port}"),
            api_key,
            cookie: Mutex::new(None),
        }
    }

    pub fn health(&self) -> bool {
        match self.agent.get(&format!("{}/health", self.base)).call() {
            Ok(resp) => resp
                .into_string()
                .ok()
                .and_then(|body| serde_json::from_str::<Value>(&body).ok())
                .is_some_and(|v| v.get("status").and_then(Value::as_str) == Some("healthy")),
            Err(_) => false,
        }
    }

    pub fn has_key(&self) -> bool {
        self.api_key.is_some()
    }

    fn login(&self) -> Option<String> {
        let key = self.api_key.as_ref()?;
        let body = serde_json::json!({ "api_key": key, "remember": true }).to_string();
        let resp = self
            .agent
            .post(&format!("{}/admin/api/login", self.base))
            .set("Content-Type", "application/json")
            .send_string(&body)
            .ok()?;
        let cookie = resp
            .all("set-cookie")
            .into_iter()
            .find(|c| c.starts_with("omlx_admin_session="))?
            .split(';')
            .next()?
            .to_owned();
        *self.cookie.lock().unwrap() = Some(cookie.clone());
        Some(cookie)
    }

    fn get_admin(&self, path: &str) -> Option<Value> {
        let url = format!("{}{}", self.base, path);
        let cached = self.cookie.lock().unwrap().clone();
        let cookie = match cached {
            Some(c) => c,
            None => self.login()?,
        };
        match self.agent.get(&url).set("Cookie", &cookie).call() {
            Ok(resp) => serde_json::from_str(&resp.into_string().ok()?).ok(),
            Err(ureq::Error::Status(401, _)) => {
                let fresh = self.login()?;
                let resp = self.agent.get(&url).set("Cookie", &fresh).call().ok()?;
                serde_json::from_str(&resp.into_string().ok()?).ok()
            }
            Err(_) => None,
        }
    }

    /// Session totals plus live model / request numbers.
    pub fn session(&self) -> Option<(Totals, Live)> {
        let v = self.get_admin("/admin/api/stats")?;
        Some((totals(&v), live(&v)))
    }

    pub fn alltime(&self) -> Option<Totals> {
        self.get_admin("/admin/api/stats?scope=alltime").map(|v| totals(&v))
    }
}

fn totals(v: &Value) -> Totals {
    Totals {
        prompt_tokens: v.get("total_prompt_tokens").and_then(Value::as_u64),
        cached_tokens: v.get("total_cached_tokens").and_then(Value::as_u64),
        cache_efficiency: v.get("cache_efficiency").and_then(Value::as_f64),
        avg_prefill_tps: v.get("avg_prefill_tps").and_then(Value::as_f64),
        avg_generation_tps: v.get("avg_generation_tps").and_then(Value::as_f64),
        requests: v.get("total_requests").and_then(Value::as_u64),
    }
}

fn live(v: &Value) -> Live {
    let am = v.get("active_models").cloned().unwrap_or(Value::Null);
    let models = am
        .get("models")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    let n = |k: &str| am.get(k).and_then(Value::as_u64).unwrap_or(0);
    Live {
        models,
        memory_used: n("model_memory_used"),
        memory_max: n("model_memory_max"),
        active: n("total_active_requests"),
        waiting: n("total_waiting_requests"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_stats_payload() {
        let v: Value = serde_json::from_str(
            r#"{"total_prompt_tokens":1703063,"total_cached_tokens":1636352,"cache_efficiency":96.1,
                "avg_prefill_tps":699.0,"avg_generation_tps":46.1,"total_requests":20,
                "active_models":{"models":[{"id":"qwen3.6-35b","active_requests":1}],
                "model_memory_used":39775275993,"model_memory_max":70503064752,
                "total_active_requests":1,"total_waiting_requests":0}}"#,
        )
        .unwrap();
        let t = totals(&v);
        assert_eq!(t.requests, Some(20));
        assert_eq!(t.cache_efficiency, Some(96.1));
        let l = live(&v);
        assert_eq!(l.models, vec!["qwen3.6-35b".to_string()]);
        assert_eq!(l.active, 1);
        assert_eq!(l.memory_max, 70503064752);
    }
}
