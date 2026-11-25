//! Every PC vtable slot the layer calls by number, checked against the public D3D12 headers (the
//! vtables of the `windows` crate). A wrong number calls another method without any error: console
//! list slot 96 called `ID3D12GraphicsCommandList8::OMSetFrontAndBackStencilRef` (81) with three
//! floats, where `RSSetDepthBias` is 82.
//!
//! Three groups: the relayed rows of `abi` (every `Fwd` slot must have an entry here), the numbers
//! written in `method::<…>(pc, n)` calls, and the named constants of the newer interfaces.

use std::mem::offset_of;

use windows::Win32::Graphics::Direct3D12::*;

use super::*;

/// The PC slot of method `$m` in vtable `$vt`.
macro_rules! at {
    ($vt:ty, $m:ident) => {
        offset_of!($vt, $m) / std::mem::size_of::<usize>()
    };
}

/// The PC slot each relayed console slot must reach, by object kind.
fn relayed(kind: Kind) -> Vec<(u32, usize, &'static str)> {
    let object = vec![
        (3, at!(ID3D12Object_Vtbl, GetPrivateData), "GetPrivateData"),
        (4, at!(ID3D12Object_Vtbl, SetPrivateData), "SetPrivateData"),
        (5, at!(ID3D12Object_Vtbl, SetPrivateDataInterface), "SetPrivateDataInterface"),
        (6, at!(ID3D12Object_Vtbl, SetName), "SetName"),
    ];
    let child = |mut v: Vec<(u32, usize, &'static str)>| {
        v.push((7, at!(ID3D12DeviceChild_Vtbl, GetDevice), "GetDevice"));
        v
    };
    let mut rows = match kind {
        Kind::Device => {
            let mut v = object.clone();
            v.extend([
                (7, at!(ID3D12Device_Vtbl, GetNodeCount), "GetNodeCount"),
                (13, at!(ID3D12Device_Vtbl, CheckFeatureSupport), "CheckFeatureSupport"),
                (28, at!(ID3D12Device_Vtbl, CreateHeap), "CreateHeap"),
                (30, at!(ID3D12Device_Vtbl, CreateReservedResource), "CreateReservedResource"),
                (32, at!(ID3D12Device_Vtbl, OpenSharedHandle), "OpenSharedHandle"),
                (33, at!(ID3D12Device_Vtbl, OpenSharedHandleByName), "OpenSharedHandleByName"),
                (37, at!(ID3D12Device_Vtbl, GetDeviceRemovedReason), "GetDeviceRemovedReason"),
                (38, at!(ID3D12Device_Vtbl, GetCopyableFootprints), "GetCopyableFootprints"),
                (39, at!(ID3D12Device_Vtbl, CreateQueryHeap), "CreateQueryHeap"),
                (40, at!(ID3D12Device_Vtbl, SetStablePowerState), "SetStablePowerState"),
                (43, at!(ID3D12Device_Vtbl, GetAdapterLuid), "GetAdapterLuid"),
            ]);
            v
        }
        Kind::List | Kind::DmaList => {
            let mut v = child(object.clone());
            v.push((8, at!(ID3D12CommandList_Vtbl, GetType), "GetType"));
            v.extend([
                (11, at!(ID3D12GraphicsCommandList_Vtbl, ClearState), "ClearState"),
                (21, at!(ID3D12GraphicsCommandList_Vtbl, RSSetViewports), "RSSetViewports"),
                (22, at!(ID3D12GraphicsCommandList_Vtbl, RSSetScissorRects), "RSSetScissorRects"),
                (23, at!(ID3D12GraphicsCommandList_Vtbl, OMSetBlendFactor), "OMSetBlendFactor"),
                (24, at!(ID3D12GraphicsCommandList_Vtbl, OMSetStencilRef), "OMSetStencilRef"),
                (28, at!(ID3D12GraphicsCommandList_Vtbl, BeginQuery), "BeginQuery"),
                (29, at!(ID3D12GraphicsCommandList_Vtbl, EndQuery), "EndQuery"),
                (49, at!(ID3D12GraphicsCommandList_Vtbl, SOSetTargets), "SOSetTargets"),
                (56, at!(ID3D12GraphicsCommandList_Vtbl, SetMarker), "SetMarker"),
                (57, at!(ID3D12GraphicsCommandList_Vtbl, BeginEvent), "BeginEvent"),
                (58, at!(ID3D12GraphicsCommandList_Vtbl, EndEvent), "EndEvent"),
                (68, at!(ID3D12GraphicsCommandList1_Vtbl, OMSetDepthBounds), "OMSetDepthBounds"),
            ]);
            v
        }
        Kind::Queue => {
            let mut v = child(object.clone());
            v.extend([
                (11, at!(ID3D12CommandQueue_Vtbl, SetMarker), "SetMarker"),
                (12, at!(ID3D12CommandQueue_Vtbl, BeginEvent), "BeginEvent"),
                (13, at!(ID3D12CommandQueue_Vtbl, EndEvent), "EndEvent"),
            ]);
            v
        }
        Kind::Heap => {
            let mut v = child(object.clone());
            v.push((8, at!(ID3D12DescriptorHeap_Vtbl, GetDesc), "GetDesc"));
            v
        }
        Kind::Allocator => child(object.clone()),
        Kind::Resource => {
            let mut v = child(object.clone());
            v.extend([
                (10, at!(ID3D12Resource_Vtbl, GetDesc), "GetDesc"),
                (12, at!(ID3D12Resource_Vtbl, WriteToSubresource), "WriteToSubresource"),
                (13, at!(ID3D12Resource_Vtbl, ReadFromSubresource), "ReadFromSubresource"),
                (14, at!(ID3D12Resource_Vtbl, GetHeapProperties), "GetHeapProperties"),
            ]);
            v
        }
        Kind::Fence => {
            let mut v = child(object.clone());
            v.extend([
                (8, at!(ID3D12Fence_Vtbl, GetCompletedValue), "GetCompletedValue"),
                (9, at!(ID3D12Fence_Vtbl, SetEventOnCompletion), "SetEventOnCompletion"),
                (11, at!(ID3D12Fence1_Vtbl, GetCreationFlags), "GetCreationFlags"),
            ]);
            v
        }
    };
    rows.sort_by_key(|r| r.0);
    rows
}

#[test]
fn every_relayed_console_slot_reaches_the_pc_method_its_row_names() {
    for kind in Kind::ALL {
        let expected = relayed(kind);
        for rule in kind.rules().iter().filter(|r| r.kind == RuleKind::Fwd) {
            for slot in u32::from(rule.first)..=u32::from(rule.last) {
                let pc = usize::from(rule.pc) + (slot - u32::from(rule.first)) as usize;
                let Some(&(_, want, name)) = expected.iter().find(|e| e.0 == slot) else {
                    panic!("{} slot {slot} is relayed to PC {pc} without a check here ({})", kind.name(), rule.what);
                };
                assert_eq!(pc, want, "{} slot {slot} must reach {name}", kind.name());
            }
        }
    }
}

#[test]
fn the_pc_slots_written_in_method_calls_are_those_of_the_headers() {
    // The list methods the proxies call by number.
    for (n, want, name) in [
        (9, at!(ID3D12GraphicsCommandList_Vtbl, Close), "Close"),
        (10, at!(ID3D12GraphicsCommandList_Vtbl, Reset), "Reset"),
        (12, at!(ID3D12GraphicsCommandList_Vtbl, DrawInstanced), "DrawInstanced"),
        (13, at!(ID3D12GraphicsCommandList_Vtbl, DrawIndexedInstanced), "DrawIndexedInstanced"),
        (14, at!(ID3D12GraphicsCommandList_Vtbl, Dispatch), "Dispatch"),
        (15, at!(ID3D12GraphicsCommandList_Vtbl, CopyBufferRegion), "CopyBufferRegion"),
        (16, at!(ID3D12GraphicsCommandList_Vtbl, CopyTextureRegion), "CopyTextureRegion"),
        (17, at!(ID3D12GraphicsCommandList_Vtbl, CopyResource), "CopyResource"),
        (18, at!(ID3D12GraphicsCommandList_Vtbl, CopyTiles), "CopyTiles"),
        (19, at!(ID3D12GraphicsCommandList_Vtbl, ResolveSubresource), "ResolveSubresource"),
        (20, at!(ID3D12GraphicsCommandList_Vtbl, IASetPrimitiveTopology), "IASetPrimitiveTopology"),
        (25, at!(ID3D12GraphicsCommandList_Vtbl, SetPipelineState), "SetPipelineState"),
        (26, at!(ID3D12GraphicsCommandList_Vtbl, ResourceBarrier), "ResourceBarrier"),
        (27, at!(ID3D12GraphicsCommandList_Vtbl, ExecuteBundle), "ExecuteBundle"),
        (28, at!(ID3D12GraphicsCommandList_Vtbl, SetDescriptorHeaps), "SetDescriptorHeaps"),
        (43, at!(ID3D12GraphicsCommandList_Vtbl, IASetIndexBuffer), "IASetIndexBuffer"),
        (44, at!(ID3D12GraphicsCommandList_Vtbl, IASetVertexBuffers), "IASetVertexBuffers"),
        (46, at!(ID3D12GraphicsCommandList_Vtbl, OMSetRenderTargets), "OMSetRenderTargets"),
        (47, at!(ID3D12GraphicsCommandList_Vtbl, ClearDepthStencilView), "ClearDepthStencilView"),
        (48, at!(ID3D12GraphicsCommandList_Vtbl, ClearRenderTargetView), "ClearRenderTargetView"),
        (49, at!(ID3D12GraphicsCommandList_Vtbl, ClearUnorderedAccessViewUint), "ClearUnorderedAccessViewUint"),
        (50, at!(ID3D12GraphicsCommandList_Vtbl, ClearUnorderedAccessViewFloat), "ClearUnorderedAccessViewFloat"),
        (51, at!(ID3D12GraphicsCommandList_Vtbl, DiscardResource), "DiscardResource"),
        (54, at!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData), "ResolveQueryData"),
        (55, at!(ID3D12GraphicsCommandList_Vtbl, SetPredication), "SetPredication"),
        (59, at!(ID3D12GraphicsCommandList_Vtbl, ExecuteIndirect), "ExecuteIndirect"),
        (62, at!(ID3D12GraphicsCommandList1_Vtbl, OMSetDepthBounds), "OMSetDepthBounds"),
    ] {
        assert_eq!(n, want, "list method {name}");
    }
    // The device methods the proxies call by number.
    for (n, want, name) in [
        (8, at!(ID3D12Device_Vtbl, CreateCommandQueue), "CreateCommandQueue"),
        (9, at!(ID3D12Device_Vtbl, CreateCommandAllocator), "CreateCommandAllocator"),
        (10, at!(ID3D12Device_Vtbl, CreateGraphicsPipelineState), "CreateGraphicsPipelineState"),
        (11, at!(ID3D12Device_Vtbl, CreateComputePipelineState), "CreateComputePipelineState"),
        (12, at!(ID3D12Device_Vtbl, CreateCommandList), "CreateCommandList"),
        (13, at!(ID3D12Device_Vtbl, CheckFeatureSupport), "CheckFeatureSupport"),
        (14, at!(ID3D12Device_Vtbl, CreateDescriptorHeap), "CreateDescriptorHeap"),
        (15, at!(ID3D12Device_Vtbl, GetDescriptorHandleIncrementSize), "GetDescriptorHandleIncrementSize"),
        (16, at!(ID3D12Device_Vtbl, CreateRootSignature), "CreateRootSignature"),
        (17, at!(ID3D12Device_Vtbl, CreateConstantBufferView), "CreateConstantBufferView"),
        (18, at!(ID3D12Device_Vtbl, CreateShaderResourceView), "CreateShaderResourceView"),
        (19, at!(ID3D12Device_Vtbl, CreateUnorderedAccessView), "CreateUnorderedAccessView"),
        (20, at!(ID3D12Device_Vtbl, CreateRenderTargetView), "CreateRenderTargetView"),
        (21, at!(ID3D12Device_Vtbl, CreateDepthStencilView), "CreateDepthStencilView"),
        (22, at!(ID3D12Device_Vtbl, CreateSampler), "CreateSampler"),
        (23, at!(ID3D12Device_Vtbl, CopyDescriptors), "CopyDescriptors"),
        (24, at!(ID3D12Device_Vtbl, CopyDescriptorsSimple), "CopyDescriptorsSimple"),
        (25, at!(ID3D12Device_Vtbl, GetResourceAllocationInfo), "GetResourceAllocationInfo"),
        (26, at!(ID3D12Device_Vtbl, GetCustomHeapProperties), "GetCustomHeapProperties"),
        (27, at!(ID3D12Device_Vtbl, CreateCommittedResource), "CreateCommittedResource"),
        (29, at!(ID3D12Device_Vtbl, CreatePlacedResource), "CreatePlacedResource"),
        (31, at!(ID3D12Device_Vtbl, CreateSharedHandle), "CreateSharedHandle"),
        (41, at!(ID3D12Device_Vtbl, CreateCommandSignature), "CreateCommandSignature"),
        (42, at!(ID3D12Device_Vtbl, GetResourceTiling), "GetResourceTiling"),
        (43, at!(ID3D12Device_Vtbl, GetAdapterLuid), "GetAdapterLuid"),
    ] {
        assert_eq!(n, want, "device method {name}");
    }
    // Queue, allocator, heap and resource methods called by number.
    for (n, want, name) in [
        (8, at!(ID3D12CommandQueue_Vtbl, UpdateTileMappings), "UpdateTileMappings"),
        (9, at!(ID3D12CommandQueue_Vtbl, CopyTileMappings), "CopyTileMappings"),
        (10, at!(ID3D12CommandQueue_Vtbl, ExecuteCommandLists), "ExecuteCommandLists"),
        (8, at!(ID3D12CommandAllocator_Vtbl, Reset), "allocator Reset"),
        (9, at!(ID3D12DescriptorHeap_Vtbl, GetCPUDescriptorHandleForHeapStart), "GetCPUDescriptorHandleForHeapStart"),
        (10, at!(ID3D12DescriptorHeap_Vtbl, GetGPUDescriptorHandleForHeapStart), "GetGPUDescriptorHandleForHeapStart"),
        (8, at!(ID3D12Resource_Vtbl, Map), "Map"),
        (11, at!(ID3D12Resource_Vtbl, GetGPUVirtualAddress), "GetGPUVirtualAddress"),
    ] {
        assert_eq!(n, want, "{name}");
    }
}

#[test]
fn the_newer_interface_slots_are_those_of_the_headers() {
    // ID3D12GraphicsCommandList7 to 10: Barrier 80, OMSetFrontAndBackStencilRef 81, RSSetDepthBias
    // 82 (not 81), IASetIndexBufferStripCutValue 83, then the work graph methods.
    assert_eq!(list_misc::PC_RS_SET_DEPTH_BIAS, at!(ID3D12GraphicsCommandList9_Vtbl, RSSetDepthBias), "RSSetDepthBias");
    assert_ne!(list_misc::PC_RS_SET_DEPTH_BIAS, at!(ID3D12GraphicsCommandList8_Vtbl, OMSetFrontAndBackStencilRef));
    assert_eq!(mesh_pipeline::PC_DISPATCH_MESH, at!(ID3D12GraphicsCommandList6_Vtbl, DispatchMesh), "DispatchMesh");
    assert_eq!(mesh_pipeline::PC_CREATE_PIPELINE_STATE, at!(ID3D12Device2_Vtbl, CreatePipelineState), "CreatePipelineState");
    let same = |a: windows::core::GUID, b: GUID| (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4);
    use windows::core::Interface;
    assert!(same(ID3D12GraphicsCommandList9::IID, list_misc::IID_GRAPHICS_COMMAND_LIST9));
}
