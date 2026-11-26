//! A value made once for each device, found again by the device pointer. The pure part of
//! `fallback`: the zeroed buffer is a resource of one device, so its GPU address is no address on another.

use std::sync::Mutex;

enum DeviceMap<T: Copy> {
    Empty,
    One(usize, T),
    Many(Vec<(usize, T)>),
}

/// One value for each device key; `make` runs once for a key (also when it made 0 or `None`).
pub struct PerDevice<T: Copy> {
    map: Mutex<DeviceMap<T>>,
}

impl<T: Copy> PerDevice<T> {
    pub const fn new() -> Self {
        Self { map: Mutex::new(DeviceMap::Empty) }
    }

    /// The value of `device`, made by `make` the first time. `make` runs under the lock, so two
    /// threads asking for one device make one value.
    pub fn get_or_make(&self, device: usize, make: impl FnOnce() -> T) -> T {
        let mut g = self.map.lock().unwrap_or_else(|e| e.into_inner());
        match &mut *g {
            DeviceMap::Empty => {
                let value = make();
                *g = DeviceMap::One(device, value);
                value
            }
            DeviceMap::One(k0, v0) => {
                if *k0 == device {
                    return *v0;
                }
                let k0 = *k0;
                let v0 = *v0;
                let value = make();
                *g = DeviceMap::Many(vec![(k0, v0), (device, value)]);
                value
            }
            DeviceMap::Many(entries) => {
                for &(k, v) in entries.iter() {
                    if k == device {
                        return v;
                    }
                }
                let value = make();
                entries.push((device, value));
                value
            }
        }
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

    #[test]
    fn many_devices_and_concurrent_threads_keep_distinct_values() {
        let v = std::sync::Arc::new(PerDevice::<usize>::new());
        for id in 10..26 {
            assert_eq!(v.get_or_make(id, || id * 3), id * 3);
        }
        for id in 10..26 {
            assert_eq!(v.get_or_make(id, || 0), id * 3);
        }
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let v = &v;
                scope.spawn(move || {
                    for id in 10..26 {
                        assert_eq!(v.get_or_make(id, || 999), id * 3);
                    }
                });
            }
        });
    }
}
