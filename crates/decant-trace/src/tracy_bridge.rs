//! The hook that forwards spans, plots and frame marks to the Tracy profiler.
//!
//! Without the `tracy` feature every function here is empty and `Zone` has no size, so the rest of the
//! crate calls them without any `cfg`.

#[cfg(feature = "tracy")]
mod real {
    use std::collections::HashMap;
    use std::sync::{Mutex, PoisonError};

    use tracy_client::{Client, PlotName, Span};

    /// A Tracy zone that ends when it is dropped. Tracy zones belong to the thread that began them.
    pub struct Zone(#[expect(dead_code, reason = "held so the zone ends on drop")] Option<Span>);

    impl Zone {
        pub fn enter(cat: &'static str, name: &'static str) -> Self {
            // The category stands in for the function and the file of the zone, as Tracy has no category.
            Zone(Client::running().map(|c| c.span_alloc(Some(name), cat, "decant-trace", 0, 0)))
        }
    }

    static PLOTS: Mutex<Option<HashMap<(&'static str, &'static str), PlotName>>> = Mutex::new(None);

    /// Plots `value` on the plot `prefix` followed by `name`. Each plot name is leaked once, as Tracy requires.
    pub fn plot(prefix: &'static str, name: &'static str, value: f64) {
        let Some(client) = Client::running() else { return };
        let plot = *PLOTS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(HashMap::new)
            .entry((prefix, name))
            .or_insert_with(|| PlotName::new_leak(format!("{prefix}{name}")));
        client.plot(plot, value);
    }

    pub fn frame() {
        if let Some(client) = Client::running() {
            client.frame_mark();
        }
    }

    pub fn start() {
        let _ = Client::start();
    }
}

#[cfg(not(feature = "tracy"))]
mod real {
    /// Stands in for the Tracy zone when the feature is off.
    pub struct Zone;

    impl Zone {
        #[inline]
        pub fn enter(_cat: &'static str, _name: &'static str) -> Self {
            Zone
        }
    }

    #[inline]
    pub fn plot(_prefix: &'static str, _name: &'static str, _value: f64) {}

    #[inline]
    pub fn frame() {}

    #[inline]
    pub fn start() {}
}

pub(crate) use real::{Zone, frame, plot, start};

/// True when the crate was built with the `tracy` feature.
pub const ENABLED: bool = cfg!(feature = "tracy");
