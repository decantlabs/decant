//! A D3D12 resource whose diagnostic charge is removed when the resource is dropped.

#![allow(unsafe_code)]

use std::ops::Deref;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::mem_owners::{self, Charge, Kind, Segment};

pub(crate) struct TrackedResource {
    pub(crate) resource: ID3D12Resource,
    charge: Charge,
}

impl TrackedResource {
    pub(crate) fn readback(resource: ID3D12Resource, bytes: u64) -> Self {
        let charge = Charge {
            kind: Kind::Readback,
            segment: Segment::System,
            bytes,
        };
        mem_owners::add_charge(charge);
        Self { resource, charge }
    }
}

impl Deref for TrackedResource {
    type Target = ID3D12Resource;

    fn deref(&self) -> &Self::Target {
        &self.resource
    }
}

impl Drop for TrackedResource {
    fn drop(&mut self) {
        mem_owners::remove_charge(self.charge);
    }
}
