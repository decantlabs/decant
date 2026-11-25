//! A value made once for each device, found again by the device pointer. The pure part of
//! `fallback`: the zeroed buffer is a resource of one device, so its GPU address is no address on another.

use std::collections::HashMap;
use std::sync::Mutex;

/// One value for each device key; `make` runs once for a key (also when it made 0 or `None`).
pub struct PerDevice<T: Copy> {
    map: Mutex<Option<HashMap<usize, T>>>,
}

impl<T: Copy> PerDevice<T> {
    pub const fn new() -> Self {
        Self { map: Mutex::new(None) }
    }

    /// The value of `device`, made by `make` the first time. `make` runs under the lock, so two
    /// threads asking for one device make one value.
    pub fn get_or_make(&self, device: usize, make: impl FnOnce() -> T) -> T {
        let mut g = self.map.lock().unwrap_or_else(|e| e.into_inner());
        *g.get_or_insert_with(HashMap::new).entry(device).or_insert_with(make)
    }
}

impl<T: Copy> Default for PerDevice<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_device_gets_its_own_value_made_once() {
        let v = PerDevice::<u64>::new();
        let mut made = 0;
        assert_eq!(v.get_or_make(1, || { made += 1; 0x1000 }), 0x1000);
        assert_eq!(v.get_or_make(1, || { made += 1; 0x9999 }), 0x1000, "the same device keeps its value");
        assert_eq!(v.get_or_make(2, || { made += 1; 0x2000 }), 0x2000, "another device does not get the first one's address");
        assert_eq!(made, 2);
        assert_eq!(v.get_or_make(3, || 0), 0);
        assert_eq!(v.get_or_make(3, || 7), 0, "a refusal is kept too");
    }
}
